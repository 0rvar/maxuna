use std::sync::Arc;

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::RmsNorm;

use crate::attention::{AttnBlock, AttnWeights, PrefillMask};
use crate::config::LagunaConfig;
use crate::gguf::{GgufFile, QLinear, Weights};
use crate::kv_cache::{LayerCache, MaskKind};
use crate::moe::{DenseMlp, MoeBlock};
use crate::ops::ExpertRunner;
use crate::rope::Rope;

/// Warn at load if the resident footprint (weights mmap-uploaded to the device
/// plus the preallocated KV cache) exceeds this. The Q4_K_M checkpoint is ~70GB
/// on its own, so this only fires on an over-large `--max-ctx`.
const MEMORY_WARN_BYTES: u64 = 90 * 1024 * 1024 * 1024;

/// The per-layer FFN: layer 0 is a dense SwiGLU MLP, layers >= dense_layers are
/// the sigmoid-routed MoE block (routed experts + always-on shared expert).
enum Ffn {
    Dense(DenseMlp),
    Moe(MoeBlock),
}

/// One transformer layer: pre-attention norm + attention, pre-FFN norm + FFN.
/// Both residual adds are owned by `LagunaModel::forward`, not here.
struct Layer {
    attn_norm: RmsNorm,
    attn: AttnBlock,
    ffn_norm: RmsNorm,
    ffn: Ffn,
}

/// The assembled model: embeddings, 48 layers (attn + dense/MoE FFN), final
/// norm, lm_head. Holds the per-layer KV caches; batch=1 by design. Exposes
/// per-layer residual taps (fork cb() names) for the parity harness and,
/// later, the DFlash drafter.
pub struct LagunaModel {
    cfg: LagunaConfig,
    device: Device,
    /// Dequantized token embeddings `[vocab, hidden]`, f16 on Metal (halves the
    /// 1.2GB f32 footprint) or f32 elsewhere. Rows are gathered per forward.
    embed: Tensor,
    layers: Vec<Layer>,
    caches: Vec<LayerCache>,
    output_norm: RmsNorm,
    lm_head: QLinear,
    /// Retained handle to the lm_head weight's Metal buffer, shared with
    /// `lm_head`'s QTensor (zero-copy). Present only on Metal for the vendored
    /// q4_K/q6_K plain mat-vec; `None` off Metal or for an unsupported dtype, in
    /// which case the decode path stays on `lm_head.forward`.
    lm_head_buffer: Option<Arc<candle_metal_kernels::metal::Buffer>>,
    lm_head_dtype: candle_core::quantized::GgmlDType,
    /// Attention weight dtype resolved once at load (F16 default, F32 under
    /// LAGUNA_ATTN_F32); activations are f32 either way. Surfaced so dump
    /// provenance can record which path ran.
    attn_dtype: DType,
    max_ctx: usize,
    tap_enabled: bool,
    taps: Vec<(String, Tensor)>,
}

impl LagunaModel {
    pub fn load(gguf: Arc<GgufFile>, runner: ExpertRunner, max_ctx: usize) -> Result<Self> {
        let cfg = LagunaConfig::from_gguf(&gguf.content)?;
        let device = gguf.device.clone();
        let w = Weights::from_gguf(gguf.clone());

        // Attention WEIGHT dtype, read ONCE at load: f16 by default — the GGUF
        // stores the attention weights as F16, so the default keeps them dense
        // f16 and runs each projection through the vendored mixed-dtype kernels
        // (ops::matmul_f16: f16 weights x f32 activations, f32 accumulate and
        // output — the fork's exact mul_mat precision structure, with the
        // stored weights as the only f16 rounding). LAGUNA_ATTN_F32
        // (presence-based, like the other LAGUNA_* switches) dequantizes them
        // to dense f32 instead — the fully legacy path, which the strict
        // parity tier gates.
        let attn_dtype = if std::env::var_os("LAGUNA_ATTN_F32").is_some() {
            DType::F32
        } else {
            DType::F16
        };
        let attn_weights = match attn_dtype {
            DType::F32 => AttnWeights::DequantF32,
            _ => AttnWeights::F16,
        };

        // Two RoPE tables shared across all layers of each type (Arc-shared into
        // every AttnBlock): YaRN for full-attention layers, plain for SWA. Built
        // to the runtime context budget, not n_ctx_train.
        let rope_full = Arc::new(Rope::new(&cfg.rope_full, max_ctx, &device)?);
        let rope_swa = Arc::new(Rope::new(&cfg.rope_swa, max_ctx, &device)?);

        // Token embeddings: dequantize once. On Metal the f32 table is 1.2GB, so
        // keep it as f16 (halved) and upcast the gathered rows to f32 per forward.
        let embed = w.qtensor("token_embd")?.dequantize(&device)?;
        let embed = if matches!(device, Device::Metal(_)) {
            embed.to_dtype(DType::F16)?
        } else {
            embed.to_dtype(DType::F32)?
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        let mut caches = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let lw = w.pp(format!("blk.{il}"));
            let rope = if cfg.is_full_attn(il) { rope_full.clone() } else { rope_swa.clone() };
            let attn = AttnBlock::new(&lw, &cfg, il, rope, attn_weights)?;
            let ffn = if il < cfg.dense_layers {
                Ffn::Dense(DenseMlp::new(&lw)?)
            } else {
                Ffn::Moe(MoeBlock::new(&lw, &cfg, runner)?)
            };
            layers.push(Layer {
                attn_norm: lw.rms_norm("attn_norm", cfg.rms_eps)?,
                attn,
                ffn_norm: lw.rms_norm("ffn_norm", cfg.rms_eps)?,
                ffn,
            });
            caches.push(LayerCache::new(&cfg, il, max_ctx, &device)?);
        }

        let output_norm = w.rms_norm("output_norm", cfg.rms_eps)?;
        let (lm_head, lm_head_buffer, lm_head_dtype) = w.qlinear_with_buffer("output")?;

        warn_if_over_budget(&gguf, &cfg, max_ctx);

        Ok(Self {
            cfg,
            device,
            embed,
            layers,
            caches,
            output_norm,
            lm_head,
            lm_head_buffer,
            lm_head_dtype,
            attn_dtype,
            max_ctx,
            tap_enabled: false,
            taps: Vec::new(),
        })
    }

    pub fn config(&self) -> &LagunaConfig {
        &self.cfg
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// The attention WEIGHT dtype `load` resolved: `F16` (GGUF-stored f16
    /// weights through the vendored mixed-dtype kernels — the shipped default)
    /// or `F32` (dequantized dense f32, the legacy path selected by
    /// `LAGUNA_ATTN_F32`). Activations are f32 in both modes.
    pub fn attn_dtype(&self) -> DType {
        self.attn_dtype
    }

    /// Run the transformer stack (embedding → 48 layers → final norm) and return
    /// the post-final-norm hidden states `[seq, hidden]` for EVERY position,
    /// together with the per-layer taps collected when capture is enabled.
    /// Shared by `forward` (which narrows to the last position for the lm head)
    /// and `forward_all_logits` (which keeps every position). Advances the KV
    /// caches, so callers feeding chunks must pass a monotonically increasing
    /// `pos`.
    fn run_stack(&mut self, tokens: &Tensor, pos: usize) -> Result<(Tensor, Vec<(String, Tensor)>)> {
        let seq = tokens.elem_count();
        ensure!(
            pos + seq <= self.max_ctx,
            "context overflow: position {pos} + {seq} tokens exceeds max_ctx {} \
             (raise --max-ctx or shorten the prompt)",
            self.max_ctx
        );

        // Embedding lookup, upcast to the f32 residual stream.
        let tokens = tokens.to_dtype(DType::U32)?;
        let mut x = self.embed.index_select(&tokens, 0)?.to_dtype(DType::F32)?; // [seq, hidden]

        // Taps are collected into a local vec (no self-borrow tangle with the
        // per-layer cache mutation) and published by the caller when enabled.
        let mut taps: Vec<(String, Tensor)> = Vec::new();
        macro_rules! tap {
            ($name:expr, $il:expr, $t:expr) => {
                if self.tap_enabled {
                    taps.push((format!("{}-{}", $name, $il), $t.clone()));
                }
            };
        }

        // Hoist the two distinct prefill masks (full-attn causal, SWA windowed)
        // out of the per-layer loop: the mask is a pure function of (kind, pos,
        // seq_len), so every full-attn layer shares one mask and every SWA layer
        // another. Building the two materialized sdpa masks once here replaces 48
        // per-layer builds — the dominant per-layer attention cost at seq=512.
        // Only at prefill (seq>1); decode builds no mask, so the layers see None.
        // Built lazily per kind so an all-full or all-SWA model never constructs
        // a mask it won't use.
        let (mut full_mask, mut swa_mask) = (None, None);
        if seq > 1 {
            for il in 0..self.layers.len() {
                if full_mask.is_none() && self.cfg.is_full_attn(il) {
                    full_mask = PrefillMask::build(MaskKind::Full, self.cfg.n_head(il), seq, pos, &self.device)?;
                } else if swa_mask.is_none() && !self.cfg.is_full_attn(il) {
                    swa_mask = PrefillMask::build(
                        MaskKind::Swa { window: self.cfg.sliding_window },
                        self.cfg.n_head(il),
                        seq,
                        pos,
                        &self.device,
                    )?;
                }
                if full_mask.is_some() && swa_mask.is_some() {
                    break;
                }
            }
        }

        for il in 0..self.layers.len() {
            let layer = &self.layers[il];
            let mask = if self.cfg.is_full_attn(il) { full_mask.as_ref() } else { swa_mask.as_ref() };
            let cache = &mut self.caches[il];

            let normed = layer.attn_norm.forward(&x)?;
            tap!("attn_norm", il, normed);

            // x += attn(attn_norm(x)); the AttnBlock output is the fork's
            // post-o_proj "attn_o_proj" node.
            let attn = layer.attn.forward(&normed, cache, pos, mask)?;
            tap!("attn_o_proj", il, attn);
            let ffn_inp = (&x + &attn)?;
            tap!("ffn_inp", il, ffn_inp);

            let ffn_normed = layer.ffn_norm.forward(&ffn_inp)?;
            tap!("ffn_norm", il, ffn_normed);

            // x += ffn(ffn_norm(x)).
            let ffn_out = match &layer.ffn {
                Ffn::Dense(mlp) => mlp.forward(&ffn_normed)?,
                Ffn::Moe(moe) => moe.forward(&ffn_normed)?,
            };
            tap!("ffn_out", il, ffn_out);

            x = (&ffn_inp + &ffn_out)?;
            tap!("l_out", il, x);
        }

        // Pre-final-norm residual stream (DFlash drafter's last capture point).
        if self.tap_enabled {
            taps.push(("h_nextn".to_string(), x.clone()));
        }

        let normed = self.output_norm.forward(&x)?; // [seq, hidden]
        Ok((normed, taps))
    }

    /// tokens: [seq] u32 at absolute position pos. Returns last-position
    /// logits [vocab] f32.
    pub fn forward(&mut self, tokens: &Tensor, pos: usize) -> Result<Tensor> {
        let seq = tokens.elem_count();
        let (normed, mut taps) = self.run_stack(tokens, pos)?;

        // Final norm over the full sequence, then the lm head on the LAST
        // position only — never run the vocab matmul over the whole prefill
        // chunk. `result_norm` matches the fork, which captures it after the
        // last-position gather, so it is last-position-only too.
        let last = normed.narrow(0, seq - 1, 1)?.contiguous()?; // [1, hidden]
        // Decode bypass: at one query position, run the vendored ggml-geometry
        // plain mat-vec over the shared lm_head buffer (candle's baked Q6_K mv
        // runs ~15x under bandwidth). Falls back to QMatMul for prefill (seq > 1),
        // off Metal, an unsupported dtype, or under LAGUNA_MV_CLASSIC.
        let logits = match &self.lm_head_buffer {
            Some(buf)
                if seq == 1
                    && !crate::ops::mv_classic()
                    && crate::ops::mv_vendored_supported(self.lm_head_dtype) =>
            {
                crate::ops::mul_mv(
                    buf,
                    self.lm_head_dtype,
                    self.lm_head.out_dim,
                    self.lm_head.in_dim,
                    &last,
                )? // [1, vocab]
            }
            _ => self.lm_head.forward(&last)?, // [1, vocab]
        };
        let logits = logits.flatten_all()?; // [vocab]
        if self.tap_enabled {
            taps.push(("result_norm".to_string(), last));
            taps.push(("result_output".to_string(), logits.clone()));
            self.taps = taps;
        }

        Ok(logits)
    }

    /// All-position logits `[seq, vocab]` f32 for offline scoring (perplexity
    /// parity). Runs the identical transformer stack as `forward` but keeps the
    /// lm head over EVERY position instead of narrowing to the last, so the
    /// caller can gather a next-token log-probability at every prefill position.
    ///
    /// The lm head runs through the plain QMatMul path (`lm_head.forward`) over
    /// the full `[seq, hidden]` — the same path `forward` uses for a seq > 1
    /// prefill chunk — so this shares the default prefill numerics and never
    /// touches the decode-only vendored mat-vec bypass. Offline tooling only;
    /// `forward`/`generate` are unaffected. Advances the KV caches like
    /// `forward`, so a chunked continuous pass must feed a monotonic `pos`.
    pub fn forward_all_logits(&mut self, tokens: &Tensor, pos: usize) -> Result<Tensor> {
        let (normed, _taps) = self.run_stack(tokens, pos)?;
        Ok(self.lm_head.forward(&normed)?) // [seq, vocab]
    }

    /// Enable capture of named intermediate tensors (parity bisection).
    pub fn set_tap_capture(&mut self, enabled: bool) {
        self.tap_enabled = enabled;
        if !enabled {
            self.taps.clear();
        }
    }

    pub fn take_taps(&mut self) -> Vec<(String, Tensor)> {
        std::mem::take(&mut self.taps)
    }

    pub fn reset_cache(&mut self) -> Result<()> {
        for cache in &mut self.caches {
            cache.reset();
        }
        Ok(())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Sum the resident bytes (weights + KV cache) and warn if it clears the budget.
/// Only full-attention layers preallocate to `max_ctx`; SWA layers keep a fixed
/// `sliding_window` ring, so they contribute negligibly and are ignored here.
fn warn_if_over_budget(gguf: &GgufFile, cfg: &LagunaConfig, max_ctx: usize) {
    let weight_bytes: u64 = gguf
        .content
        .tensor_infos
        .values()
        .map(|info| {
            let elems = info.shape.elem_count() as u64;
            let dt = info.ggml_dtype;
            elems / dt.block_size() as u64 * dt.type_size() as u64
        })
        .sum();

    let n_full = (0..cfg.n_layer).filter(|&il| cfg.is_full_attn(il)).count() as u64;
    // k and v, f16 (2 bytes), [n_kv_head, max_ctx, head_dim] per full layer.
    let kv_bytes =
        n_full * 2 * cfg.n_kv_head as u64 * max_ctx as u64 * cfg.head_dim as u64 * 2;

    let total = weight_bytes + kv_bytes;
    let gb = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "laguna: weights {:.1}GB + KV {:.1}GB = {:.1}GB resident (max_ctx {max_ctx})",
        gb(weight_bytes),
        gb(kv_bytes),
        gb(total)
    );
    if total > MEMORY_WARN_BYTES {
        eprintln!(
            "laguna: WARNING resident footprint {:.1}GB exceeds {:.0}GB budget",
            gb(total),
            gb(MEMORY_WARN_BYTES)
        );
    }
}
