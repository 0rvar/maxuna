//! DFlash drafter model (WP-D1) — the speculative-decoding drafter for the
//! Laguna S 2.1 target. This is a self-contained module: the drafter is a pure
//! function of its OWN weights and its OWN KV cache and never references
//! `LagunaModel`. Token embeddings and the lm_head live in the target and are
//! applied by the driver, not here (the drafter returns pre-lm_head hidden
//! states).
//!
//! Ground truth: `reference/llama.cpp-laguna-branch/src/models/dflash.cpp`
//! (`decoder_laguna == true`, i.e. the Laguna drafter variant). The architecture
//! is "dflash" with a "laguna" decoder: 6 dense layers, hidden 3072, 72 Q heads
//! / 8 KV heads, head_dim 128, QK-norm before rope, a per-head softplus attention
//! output gate, SwiGLU FFN, plain NEOX rope (theta 500k, all 128 dims, no YaRN),
//! full causal attention on every layer (no SWA).
//!
//! Weights are materialized DENSE f32 (BF16 -> f32 is exact; norms are stored
//! f32). The whole forward uses plain candle ops — no custom Metal kernels, no
//! `src/ops` dispatch. The drafter is tiny (~1.1B, <= block_size query rows per
//! forward), so this favors clarity over speed. (An f16 weight downcast would
//! roughly halve the ~4.4GB footprint; deferred.)

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use anyhow::{Context, Result, bail, ensure};
use candle_core::quantized::GgmlDType;
use candle_core::quantized::gguf_file::{Content, Value};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::RmsNorm;

use crate::gguf::GgufFile;

/// Static architecture facts parsed from the `dflash.*` GGUF metadata. Errors
/// unless the file is a Laguna-decoder DFlash drafter (arch "dflash",
/// decoder_arch "laguna").
#[derive(Debug, Clone)]
pub struct DflashConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub rms_eps: f64,
    pub rope_theta: f32,
    /// Noise-block width the drafter drafts per step (`dflash.block_size`).
    pub block_size: usize,
    /// Target-model layer indices whose residual taps feed the encoder, in the
    /// order the encoder concatenates them (`dflash.target_layers`).
    pub target_layers: Vec<usize>,
    /// Token id used to fill the masked positions of a noise block
    /// (`tokenizer.ggml.mask_token_id`).
    pub mask_token_id: u32,
    pub context_length: usize,
}

impl DflashConfig {
    pub fn from_gguf(content: &Content) -> Result<Self> {
        let get = |k: &str| content.metadata.get(k).with_context(|| format!("missing GGUF key {k}"));

        let arch = get("general.architecture")?.to_string()?.as_str();
        ensure!(arch == "dflash", "expected a dflash GGUF, got architecture {arch:?}");
        let decoder = get("dflash.decoder_arch")?.to_string()?.as_str();
        ensure!(decoder == "laguna", "expected decoder_arch \"laguna\", got {decoder:?}");

        let target_layers = get("dflash.target_layers")?
            .to_vec()
            .context("dflash.target_layers is not an array")?
            .iter()
            .map(value_usize)
            .collect::<Result<Vec<_>>>()
            .context("dflash.target_layers has non-integer entries")?;
        ensure!(!target_layers.is_empty(), "dflash.target_layers is empty");

        Ok(Self {
            n_layer: value_usize(get("dflash.block_count")?)?,
            n_embd: value_usize(get("dflash.embedding_length")?)?,
            n_head: value_usize(get("dflash.attention.head_count")?)?,
            n_head_kv: value_usize(get("dflash.attention.head_count_kv")?)?,
            head_dim: value_usize(get("dflash.attention.key_length")?)?,
            n_ff: value_usize(get("dflash.feed_forward_length")?)?,
            rms_eps: get("dflash.attention.layer_norm_rms_epsilon")?.to_f32()? as f64,
            rope_theta: get("dflash.rope.freq_base")?.to_f32()?,
            block_size: value_usize(get("dflash.block_size")?)?,
            target_layers,
            mask_token_id: get("tokenizer.ggml.mask_token_id")?.to_u32()?,
            context_length: value_usize(get("dflash.context_length")?)?,
        })
    }

    /// The encoder's concatenated feature width: `target_layers.len() * n_embd`.
    fn n_embd_inp(&self) -> usize {
        self.target_layers.len() * self.n_embd
    }

    /// The target-model `l_out` indices whose residuals feed the encoder, in
    /// tap order — i.e. what to pass to `LagunaModel::set_spec_taps`.
    ///
    /// `dflash.target_layers` entries name the residual ENTERING target layer
    /// `t` (the fork's `t_layer_inp[t]`), which is the residual LEAVING layer
    /// `t - 1`; the sentinel `t == n_layer_tgt` names the pre-final-norm
    /// state, which is likewise `l_out` of the last layer. So the uniform
    /// translation to our post-FFN `l_out` capture points is `t - 1`.
    pub fn spec_tap_layers(&self) -> Result<Vec<usize>> {
        self.target_layers
            .iter()
            .map(|&t| {
                t.checked_sub(1).with_context(|| {
                    format!("target_layers entry {t} taps the raw embedding, which has no l_out capture point")
                })
            })
            .collect()
    }
}

/// Accept any GGUF integer width/signedness (target_layers is i32[]).
fn value_usize(v: &Value) -> Result<usize> {
    if let Ok(u) = v.to_u64() {
        return Ok(u as usize);
    }
    if let Ok(u) = v.to_u32() {
        return Ok(u as usize);
    }
    let i = v.to_i64().or_else(|_| v.to_i32().map(i64::from))?;
    ensure!(i >= 0, "negative integer {i}");
    Ok(i as usize)
}

/// One decoder layer's weights, all dense f32. Matmul weights are `[out, in]`
/// (GGUF layout); `y = x @ wᵀ`.
struct LayerWeights {
    attn_norm: RmsNorm,
    wq: Tensor,   // [n_head*head_dim, n_embd]
    wk: Tensor,   // [n_head_kv*head_dim, n_embd]
    wv: Tensor,   // [n_head_kv*head_dim, n_embd]
    wo: Tensor,   // [n_embd, n_head*head_dim]
    q_norm: RmsNorm, // [head_dim]
    k_norm: RmsNorm, // [head_dim]
    gate: Tensor, // [n_head, n_embd]
    ffn_norm: RmsNorm,
    ffn_gate: Tensor, // [n_ff, n_embd]
    ffn_up: Tensor,   // [n_ff, n_embd]
    ffn_down: Tensor, // [n_embd, n_ff]
}

/// One layer's full-attention KV cache: `[n_head_kv, max_ctx, head_dim]` f32.
/// Chosen f32 (not the target's f16) for exactness — the drafter is tiny, so the
/// extra bytes are negligible and it keeps the naive-reference tests bit-tight.
struct LayerCache {
    k: Tensor,
    v: Tensor,
}

/// The DFlash drafter: encoder (feature fusion of target taps), per-layer KV
/// injection of the fused context, and noise-block draft forwards. Batch=1.
pub struct DflashDrafter {
    cfg: DflashConfig,
    device: Device,

    // Encoder.
    aux_norm: Tensor,       // [n_aux, n_embd] f32
    enc_output_norm: RmsNorm,
    fc: Tensor,             // [n_embd, n_aux*n_embd]

    // Decoder.
    output_norm: RmsNorm,   // final norm (decoder), NOT the encoder's
    layers: Vec<LayerWeights>,

    caches: Vec<LayerCache>,
    /// Committed (injected) context length, shared across layers — every layer
    /// advances together on `inject`.
    committed: usize,
    max_ctx: usize,

    // Plain NEOX rope tables (theta from cfg, all head_dim dims rotary, no YaRN
    // scaling): cos/sin `[max_ctx, head_dim/2]` f32.
    rope_cos: Tensor,
    rope_sin: Tensor,
}

impl DflashDrafter {
    /// Load the drafter from an opened GGUF. Reads the raw tensor bytes directly
    /// (BF16 -> f32, F32 -> f32); the target's mmap/residency machinery on
    /// `GgufFile` is ignored — the drafter is small enough to materialize dense.
    /// `max_ctx` sizes the per-layer KV caches and the rope tables.
    pub fn load(gguf: &GgufFile, device: &Device, max_ctx: usize) -> Result<Self> {
        let cfg = DflashConfig::from_gguf(&gguf.content)?;
        let mut file =
            File::open(&gguf.path).with_context(|| format!("opening {} for drafter load", gguf.path.display()))?;
        let content = &gguf.content;
        Self::build(cfg, device, max_ctx, |name| read_tensor_f32(&mut file, content, name, device))
    }

    /// Assemble the drafter from a tensor provider yielding dense f32 tensors by
    /// GGUF name (with the `.weight` suffix). Shared by `load` (reads the GGUF)
    /// and the tests (an in-memory synthetic map), so both exercise identical
    /// assembly.
    fn build(
        cfg: DflashConfig,
        device: &Device,
        max_ctx: usize,
        mut get: impl FnMut(&str) -> Result<Tensor>,
    ) -> Result<Self> {
        ensure!(max_ctx >= 1, "max_ctx must be >= 1");
        ensure!(cfg.head_dim % 2 == 0, "head_dim {} must be even for rope", cfg.head_dim);
        ensure!(
            cfg.n_head % cfg.n_head_kv == 0,
            "n_head {} must be a multiple of n_head_kv {}",
            cfg.n_head,
            cfg.n_head_kv
        );

        let eps = cfg.rms_eps;
        let norm = |t: Tensor| RmsNorm::new(t, eps);

        let aux_norm = get("enc.aux_norm")?;
        let n_aux = cfg.target_layers.len();
        ensure!(
            aux_norm.dims() == [n_aux, cfg.n_embd],
            "enc.aux_norm shape {:?} != [{n_aux}, {}]",
            aux_norm.dims(),
            cfg.n_embd
        );
        let enc_output_norm = norm(get("enc.output_norm")?);
        let fc = get("fc")?;
        ensure!(
            fc.dims() == [cfg.n_embd, cfg.n_embd_inp()],
            "fc shape {:?} != [{}, {}]",
            fc.dims(),
            cfg.n_embd,
            cfg.n_embd_inp()
        );
        let output_norm = norm(get("output_norm")?);

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let p = |n: &str| format!("blk.{il}.{n}");
            layers.push(LayerWeights {
                attn_norm: norm(get(&p("attn_norm"))?),
                wq: get(&p("attn_q"))?,
                wk: get(&p("attn_k"))?,
                wv: get(&p("attn_v"))?,
                wo: get(&p("attn_output"))?,
                q_norm: norm(get(&p("attn_q_norm"))?),
                k_norm: norm(get(&p("attn_k_norm"))?),
                gate: get(&p("attn_gate"))?,
                ffn_norm: norm(get(&p("ffn_norm"))?),
                ffn_gate: get(&p("ffn_gate"))?,
                ffn_up: get(&p("ffn_up"))?,
                ffn_down: get(&p("ffn_down"))?,
            });
        }

        let (n_kv, hd) = (cfg.n_head_kv, cfg.head_dim);
        let caches = (0..cfg.n_layer)
            .map(|_| {
                Ok(LayerCache {
                    k: Tensor::zeros((n_kv, max_ctx, hd), DType::F32, device)?,
                    v: Tensor::zeros((n_kv, max_ctx, hd), DType::F32, device)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let (rope_cos, rope_sin) = rope_tables(cfg.rope_theta, hd, max_ctx, device)?;

        Ok(Self {
            cfg,
            device: device.clone(),
            aux_norm,
            enc_output_norm,
            fc,
            output_norm,
            layers,
            caches,
            committed: 0,
            max_ctx,
            rope_cos,
            rope_sin,
        })
    }

    pub fn config(&self) -> &DflashConfig {
        &self.cfg
    }

    pub fn committed_len(&self) -> usize {
        self.committed
    }

    /// Roll the committed cache back to `len` (<= current committed length). The
    /// bytes beyond `len` stay in the cache tensors but are ignored; the next
    /// `inject` overwrites them.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        ensure!(
            len <= self.committed,
            "truncate({len}) beyond committed length {}",
            self.committed
        );
        self.committed = len;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.committed = 0;
    }

    /// Encoder: fuse the target-model residual taps into the drafter's hidden
    /// space. `taps` are `n_aux` (= `target_layers.len()`) f32 tensors, each
    /// `[seq, n_embd]`, ordered as `target_layers`.
    ///
    /// Per the Laguna drafter (dflash.cpp:128-143): view the concatenated taps as
    /// `[seq, n_aux, n_embd]`, RMS-norm over the last dim (eps `rms_eps`, no
    /// weight), multiply by the stacked per-aux weights `aux_norm [n_aux, n_embd]`,
    /// flatten to `[seq, n_aux*n_embd]`, apply `fc`, then RMS-norm with
    /// `enc.output_norm`. Returns fused `[seq, n_embd]` f32.
    pub fn encode(&self, taps: &[Tensor]) -> Result<Tensor> {
        let n_aux = self.cfg.target_layers.len();
        ensure!(
            taps.len() == n_aux,
            "encode expects {n_aux} taps (one per target layer), got {}",
            taps.len()
        );
        let (seq, n_embd) = taps[0].dims2().context("tap 0 is not rank-2 [seq, n_embd]")?;
        ensure!(n_embd == self.cfg.n_embd, "tap 0 hidden {n_embd} != n_embd {}", self.cfg.n_embd);
        for (i, t) in taps.iter().enumerate() {
            let (s, e) = t.dims2().with_context(|| format!("tap {i} is not rank-2"))?;
            ensure!(s == seq && e == n_embd, "tap {i} shape [{s}, {e}] != [{seq}, {n_embd}]");
        }

        // [seq, n_aux, n_embd].
        let refs: Vec<&Tensor> = taps.iter().collect();
        let x = Tensor::stack(&refs, 1)?.to_dtype(DType::F32)?.contiguous()?;

        // ggml_rms_norm over the last dim (no weight), then multiply aux_norm.
        let ms = x.sqr()?.mean_keepdim(2)?; // [seq, n_aux, 1]
        let denom = (ms + self.cfg.rms_eps)?.sqrt()?;
        let normed = x.broadcast_div(&denom)?;
        let aux = self.aux_norm.reshape((1, n_aux, n_embd))?;
        let scaled = normed.broadcast_mul(&aux)?;

        let flat = scaled.reshape((seq, n_aux * n_embd))?.contiguous()?;
        let fused = linear(&flat, &self.fc)?; // [seq, n_embd]
        Ok(self.enc_output_norm.forward(&fused)?)
    }

    /// KV-injection: project the fused context into every layer's K/V and append
    /// to the committed cache. `pos0` must equal the current committed length
    /// (positions are sequential). K is QK-normed then roped; V is neither.
    /// (dflash.cpp:189-227, `decoder_laguna` branch: K/V come from the
    /// `attn_norm`-normed fused input, matching the query path.)
    pub fn inject(&mut self, fused: &Tensor, pos0: usize) -> Result<()> {
        ensure!(
            pos0 == self.committed,
            "inject pos0 {pos0} must equal committed length {}",
            self.committed
        );
        let (seq, n_embd) = fused.dims2().context("fused is not rank-2 [seq, n_embd]")?;
        ensure!(n_embd == self.cfg.n_embd, "fused hidden {n_embd} != n_embd {}", self.cfg.n_embd);
        ensure!(
            self.committed + seq <= self.max_ctx,
            "inject overflows cache: committed {} + {seq} > max_ctx {}",
            self.committed,
            self.max_ctx
        );
        let fused = fused.to_dtype(DType::F32)?;
        let (n_kv, hd) = (self.cfg.n_head_kv, self.cfg.head_dim);

        for il in 0..self.layers.len() {
            let layer = &self.layers[il];
            let kv_inp = layer.attn_norm.forward(&fused)?; // [seq, n_embd]

            let k = linear(&kv_inp, &layer.wk)?.reshape((seq, n_kv, hd))?;
            let v = linear(&kv_inp, &layer.wv)?.reshape((seq, n_kv, hd))?;

            let k = layer.k_norm.forward(&k)?; // QK-norm over head_dim, pre-rope
            let k = k.transpose(0, 1)?.contiguous()?; // [n_kv, seq, hd]
            let k = self.rope(&k, pos0)?;
            let v = v.transpose(0, 1)?.contiguous()?; // [n_kv, seq, hd]

            self.caches[il].k.slice_set(&k, 1, self.committed)?;
            self.caches[il].v.slice_set(&v, 1, self.committed)?;
        }
        self.committed += seq;
        Ok(())
    }

    /// Noise-block draft forward. `noise_embd` is `[n_block, n_embd]` f32 —
    /// target-side token embeddings of `[id_last, MASK * (n_block-1)]` at
    /// positions `pos0..pos0+n_block` (pos0 must equal the committed length). Runs
    /// the full causal decoder attending over `[committed cache | noise block]`;
    /// the noise K/V is used for this forward only and never written to the cache
    /// (committed is unchanged on return). Returns final-normed hidden states
    /// `[n_block, n_embd]` f32 (pre-lm_head; the caller applies the target's
    /// shared lm_head).
    pub fn draft_forward(&mut self, noise_embd: &Tensor, pos0: usize) -> Result<Tensor> {
        ensure!(
            pos0 == self.committed,
            "draft_forward pos0 {pos0} must equal committed length {}",
            self.committed
        );
        let (n_block, n_embd) = noise_embd.dims2().context("noise_embd is not rank-2")?;
        ensure!(n_embd == self.cfg.n_embd, "noise hidden {n_embd} != n_embd {}", self.cfg.n_embd);
        ensure!(n_block >= 1, "noise block must have >= 1 row");
        ensure!(
            pos0 + n_block <= self.max_ctx,
            "noise block [{pos0}, {}) exceeds max_ctx {}",
            pos0 + n_block,
            self.max_ctx
        );

        let committed = self.committed;
        let (n_kv, hd, n_head) = (self.cfg.n_head_kv, self.cfg.head_dim, self.cfg.n_head);
        let scale = 1.0f32 / (hd as f32).sqrt();

        let mut inp = noise_embd.to_dtype(DType::F32)?;

        for il in 0..self.layers.len() {
            let layer = &self.layers[il];
            let noise_norm = layer.attn_norm.forward(&inp)?; // [nb, n_embd]

            let q = linear(&noise_norm, &layer.wq)?.reshape((n_block, n_head, hd))?;
            let k = linear(&noise_norm, &layer.wk)?.reshape((n_block, n_kv, hd))?;
            let v = linear(&noise_norm, &layer.wv)?.reshape((n_block, n_kv, hd))?;

            let q = layer.q_norm.forward(&q)?;
            let k = layer.k_norm.forward(&k)?;

            let q = self.rope(&q.transpose(0, 1)?.contiguous()?, pos0)?; // [n_head, nb, hd]
            let k = self.rope(&k.transpose(0, 1)?.contiguous()?, pos0)?; // [n_kv, nb, hd]
            let v = v.transpose(0, 1)?.contiguous()?; // [n_kv, nb, hd]

            // Effective K/V = committed context ++ this block's noise K/V.
            let (k_all, v_all) = if committed > 0 {
                let k_ctx = self.caches[il].k.narrow(1, 0, committed)?;
                let v_ctx = self.caches[il].v.narrow(1, 0, committed)?;
                (Tensor::cat(&[&k_ctx, &k], 1)?, Tensor::cat(&[&v_ctx, &v], 1)?)
            } else {
                (k, v)
            };

            let attn = self.attention(&q, &k_all, &v_all, committed, n_block, scale)?; // [n_head, nb, hd]

            // Softplus output gate, per-head, broadcast over head_dim.
            let gate = softplus(&linear(&noise_norm, &layer.gate)?)?; // [nb, n_head]
            let gate = gate.transpose(0, 1)?.reshape((n_head, n_block, 1))?;
            let attn = attn.broadcast_mul(&gate)?;

            let attn = attn.transpose(0, 1)?.contiguous()?.reshape((n_block, n_head * hd))?;
            let cur = linear(&attn, &layer.wo)?; // [nb, n_embd]

            let ffn_inp = (&cur + &inp)?;
            let ffn_normed = layer.ffn_norm.forward(&ffn_inp)?;
            let ffn_out = self.swiglu(layer, &ffn_normed)?;
            inp = (&ffn_out + &ffn_inp)?;
        }

        Ok(self.output_norm.forward(&inp)?)
    }

    /// Causal GQA attention of `n_block` queries over `committed + n_block` keys.
    /// `q [n_head, nb, hd]`, `k_all`/`v_all [n_kv, committed+nb, hd]`, all f32.
    /// Committed keys are always visible; the noise keys are causally masked
    /// within the block (key i' visible to query i iff i' <= i).
    fn attention(
        &self,
        q: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
        committed: usize,
        n_block: usize,
        scale: f32,
    ) -> Result<Tensor> {
        let (n_kv, hd, n_head) = (self.cfg.n_head_kv, self.cfg.head_dim, self.cfg.n_head);
        let g = n_head / n_kv;
        let k_seq = committed + n_block;

        let q4 = q.reshape((n_kv, g, n_block, hd))?;
        let k4 = k_all.reshape((n_kv, 1, k_seq, hd))?;
        let v4 = v_all.reshape((n_kv, 1, k_seq, hd))?;

        let scores = q4.broadcast_matmul(&k4.transpose(2, 3)?)?.affine(scale as f64, 0.0)?;
        let mask = self.causal_mask(committed, n_block, k_seq)?; // [1, 1, nb, k_seq]
        let scores = scores.broadcast_add(&mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.broadcast_matmul(&v4)?; // [n_kv, g, nb, hd]
        Ok(out.reshape((n_head, n_block, hd))?)
    }

    /// Additive mask `[1, 1, n_block, k_seq]` (0 attend / -inf block): committed
    /// columns visible to every query, noise column c (>= committed) visible to
    /// query i iff `c - committed <= i`.
    fn causal_mask(&self, committed: usize, n_block: usize, k_seq: usize) -> Result<Tensor> {
        let mut data = vec![0f32; n_block * k_seq];
        for i in 0..n_block {
            for c in 0..k_seq {
                let blocked = c >= committed && (c - committed) > i;
                if blocked {
                    data[i * k_seq + c] = f32::NEG_INFINITY;
                }
            }
        }
        Ok(Tensor::from_vec(data, (1, 1, n_block, k_seq), &self.device)?)
    }

    /// SwiGLU: `down(silu(gate(x)) * up(x))`, x `[nb, n_embd]` -> `[nb, n_embd]`.
    fn swiglu(&self, layer: &LayerWeights, x: &Tensor) -> Result<Tensor> {
        let gate = linear(x, &layer.ffn_gate)?;
        let up = linear(x, &layer.ffn_up)?;
        let hidden = (candle_nn::ops::silu(&gate)? * up)?;
        linear(&hidden, &layer.ffn_down)
    }

    /// Plain NEOX rope over the full head_dim (rotate-half: dim i pairs with
    /// i + head_dim/2), for `x [n_head_or_kv, seq, head_dim]` at absolute
    /// positions `pos..pos+seq`. Bit-mirrors the engine's NEOX convention
    /// (src/rope.rs), computed with plain candle ops.
    fn rope(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let (_, seq, hd) = x.dims3()?;
        let half = hd / 2;
        let cos = self.rope_cos.narrow(0, pos, seq)?.reshape((1, seq, half))?;
        let sin = self.rope_sin.narrow(0, pos, seq)?.reshape((1, seq, half))?;
        let x1 = x.narrow(2, 0, half)?;
        let x2 = x.narrow(2, half, half)?;
        let o1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let o2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        Ok(Tensor::cat(&[&o1, &o2], 2)?.contiguous()?)
    }
}

/// `y = x @ wᵀ` for a dense `[out, in]` weight and `x [.., in]`.
fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    Ok(x.matmul(&w.t()?)?)
}

/// Numerically stable softplus, ln(1 + exp(x)) = relu(x) + ln(1 + exp(-|x|)).
fn softplus(x: &Tensor) -> Result<Tensor> {
    let ax = x.abs()?;
    let relu = x.broadcast_add(&ax)?.affine(0.5, 0.0)?;
    let tail = ax.neg()?.exp()?.affine(1.0, 1.0)?.log()?;
    Ok(relu.broadcast_add(&tail)?)
}

/// Plain NEOX rope cos/sin tables `[max_ctx, head_dim/2]` f32, mscale 1 (no
/// YaRN): `inv_freq[j] = theta^(-2j/head_dim)`, angle `p * inv_freq[j]`.
fn rope_tables(theta: f32, head_dim: usize, max_ctx: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let base = theta as f64;
    let inv_freq: Vec<f64> = (0..half).map(|j| base.powf(-(2.0 * j as f64) / head_dim as f64)).collect();
    let mut cos = vec![0f32; max_ctx * half];
    let mut sin = vec![0f32; max_ctx * half];
    for p in 0..max_ctx {
        for j in 0..half {
            let angle = p as f64 * inv_freq[j];
            cos[p * half + j] = angle.cos() as f32;
            sin[p * half + j] = angle.sin() as f32;
        }
    }
    Ok((
        Tensor::from_vec(cos, (max_ctx, half), device)?,
        Tensor::from_vec(sin, (max_ctx, half), device)?,
    ))
}

/// Read a GGUF tensor by fully-qualified name (with `.weight` suffix) into a
/// dense f32 tensor. BF16 widens exactly to f32; F32 passes through. The GGUF
/// stores matmul weights `[out, in]` and norms `[dim]`, which this preserves.
fn read_tensor_f32(file: &mut File, content: &Content, name: &str, device: &Device) -> Result<Tensor> {
    let full = format!("{name}.weight");
    let info = content
        .tensor_infos
        .get(&full)
        .with_context(|| format!("drafter tensor {full} not found"))?;
    let dims = info.shape.dims().to_vec();
    let elems: usize = dims.iter().product();
    let start = content.tensor_data_offset + info.offset;
    file.seek(SeekFrom::Start(start))
        .with_context(|| format!("seeking to {full}"))?;

    let values: Vec<f32> = match info.ggml_dtype {
        GgmlDType::F32 => {
            let mut raw = vec![0u8; elems * 4];
            file.read_exact(&mut raw).with_context(|| format!("reading {full}"))?;
            raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
        GgmlDType::BF16 => {
            let mut raw = vec![0u8; elems * 2];
            file.read_exact(&mut raw).with_context(|| format!("reading {full}"))?;
            raw.chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect()
        }
        other => bail!("drafter tensor {full} has unsupported dtype {other:?} (expected F32 or BF16)"),
    };
    Ok(Tensor::from_vec(values, dims, device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DRAFTER_PATH: &str = "models/laguna-s-2.1-DFlash-BF16.gguf";

    // --- synthetic weight plumbing -------------------------------------------

    /// Deterministic pseudo-random f32s in roughly [-0.5, 0.5] (LCG, no deps).
    fn seeded(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / u32::MAX as f32) - 0.5
            })
            .collect()
    }

    fn tiny_cfg(n_layer: usize, n_head: usize, n_kv: usize, hd: usize, n_embd: usize, n_ff: usize) -> DflashConfig {
        DflashConfig {
            n_layer,
            n_embd,
            n_head,
            n_head_kv: n_kv,
            head_dim: hd,
            n_ff,
            rms_eps: 1e-6,
            rope_theta: 500_000.0,
            block_size: 16,
            target_layers: vec![2, 11],
            mask_token_id: 12,
            context_length: 1024,
        }
    }

    /// A synthetic weight map for `cfg`: every tensor filled with seeded pseudo-
    /// random f32 (norm weights near 1.0 so the RMSNorms stay well-conditioned).
    fn synth_weights(cfg: &DflashConfig, dev: &Device) -> HashMap<String, Tensor> {
        let mut m = HashMap::new();
        let seed = std::cell::Cell::new(1u64);
        let next = || {
            seed.set(seed.get() + 1);
            seed.get()
        };
        let mat = |rows: usize, cols: usize| {
            Tensor::from_vec(seeded(rows * cols, next()), (rows, cols), dev).unwrap()
        };
        let (n_embd, hd, n_head, n_kv, n_ff) = (cfg.n_embd, cfg.head_dim, cfg.n_head, cfg.n_head_kv, cfg.n_ff);
        let n_aux = cfg.target_layers.len();

        m.insert("fc".to_string(), mat(n_embd, n_aux * n_embd));
        // Norm-like weights: 1.0 + small perturbation.
        let norm_w = |dim: usize| {
            let v: Vec<f32> = seeded(dim, next()).iter().map(|x| 1.0 + 0.1 * x).collect();
            Tensor::from_vec(v, dim, dev).unwrap()
        };
        m.insert("enc.aux_norm".to_string(), {
            let v: Vec<f32> = seeded(n_aux * n_embd, next()).iter().map(|x| 1.0 + 0.1 * x).collect();
            Tensor::from_vec(v, (n_aux, n_embd), dev).unwrap()
        });
        m.insert("enc.output_norm".to_string(), norm_w(n_embd));
        m.insert("output_norm".to_string(), norm_w(n_embd));

        for il in 0..cfg.n_layer {
            let p = |n: &str| format!("blk.{il}.{n}");
            m.insert(p("attn_norm"), norm_w(n_embd));
            m.insert(p("attn_q"), mat(n_head * hd, n_embd));
            m.insert(p("attn_k"), mat(n_kv * hd, n_embd));
            m.insert(p("attn_v"), mat(n_kv * hd, n_embd));
            m.insert(p("attn_output"), mat(n_embd, n_head * hd));
            m.insert(p("attn_q_norm"), norm_w(hd));
            m.insert(p("attn_k_norm"), norm_w(hd));
            m.insert(p("attn_gate"), mat(n_head, n_embd));
            m.insert(p("ffn_norm"), norm_w(n_embd));
            m.insert(p("ffn_gate"), mat(n_ff, n_embd));
            m.insert(p("ffn_up"), mat(n_ff, n_embd));
            m.insert(p("ffn_down"), mat(n_embd, n_ff));
        }
        m
    }

    fn drafter_from_map(cfg: DflashConfig, m: &HashMap<String, Tensor>, dev: &Device, max_ctx: usize) -> DflashDrafter {
        DflashDrafter::build(cfg, dev, max_ctx, |name| {
            m.get(name).cloned().with_context(|| format!("synthetic tensor {name} missing"))
        })
        .unwrap()
    }

    fn to_vec(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        let num: f64 = got.iter().zip(want).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>().sqrt();
        let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
        (num / den.max(1e-30)) as f32
    }

    // --- scalar f64 reference (fully independent of the tensor impl) ----------

    /// Host-side view of one layer's weights, row-major f64.
    struct RefLayer {
        attn_norm: Vec<f64>,
        wq: Vec<f64>,
        wk: Vec<f64>,
        wv: Vec<f64>,
        wo: Vec<f64>,
        q_norm: Vec<f64>,
        k_norm: Vec<f64>,
        gate: Vec<f64>,
        ffn_norm: Vec<f64>,
        ffn_gate: Vec<f64>,
        ffn_up: Vec<f64>,
        ffn_down: Vec<f64>,
    }

    fn host(m: &HashMap<String, Tensor>, name: &str) -> Vec<f64> {
        to_vec(m.get(name).unwrap()).iter().map(|x| *x as f64).collect()
    }

    /// `y[o] = Σ_i w[o*in + i] * x[i]`, w row-major `[out, in]`.
    fn matvec(w: &[f64], x: &[f64], out: usize, inn: usize) -> Vec<f64> {
        (0..out).map(|o| (0..inn).map(|i| w[o * inn + i] * x[i]).sum()).collect()
    }

    fn rmsnorm(x: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
        let ms: f64 = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let s = 1.0 / (ms + eps).sqrt();
        x.iter().zip(weight).map(|(v, w)| v * s * w).collect()
    }

    /// Rotate-half NEOX rope on a single head vector `[hd]` at absolute `pos`.
    fn rope_vec(v: &[f64], pos: usize, hd: usize, theta: f64) -> Vec<f64> {
        let half = hd / 2;
        let mut out = vec![0.0; hd];
        for j in 0..half {
            let inv = theta.powf(-(2.0 * j as f64) / hd as f64);
            let (c, s) = ((pos as f64 * inv).cos(), (pos as f64 * inv).sin());
            let (x1, x2) = (v[j], v[j + half]);
            out[j] = x1 * c - x2 * s;
            out[j + half] = x1 * s + x2 * c;
        }
        out
    }

    fn softplus_s(x: f64) -> f64 {
        // ln(1+exp(x)), stable.
        x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()
    }

    fn silu_s(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    /// A from-scratch f64 draft forward: inject `ctx` context tokens, then run
    /// the noise block through the decoder. Returns `[n_block][n_embd]`.
    #[allow(clippy::too_many_arguments)]
    fn naive_draft(
        cfg: &DflashConfig,
        m: &HashMap<String, Tensor>,
        ctx: &[Vec<f64>],   // [committed][n_embd]
        noise: &[Vec<f64>], // [n_block][n_embd]
    ) -> Vec<Vec<f64>> {
        let (n_embd, hd, n_head, n_kv, n_ff) = (cfg.n_embd, cfg.head_dim, cfg.n_head, cfg.n_head_kv, cfg.n_ff);
        let (eps, theta) = (cfg.rms_eps, cfg.rope_theta as f64);
        let g = n_head / n_kv;
        let scale = 1.0 / (hd as f64).sqrt();
        let committed = ctx.len();
        let n_block = noise.len();

        let layers: Vec<RefLayer> = (0..cfg.n_layer)
            .map(|il| {
                let p = |n: &str| host(m, &format!("blk.{il}.{n}"));
                RefLayer {
                    attn_norm: p("attn_norm"),
                    wq: p("attn_q"),
                    wk: p("attn_k"),
                    wv: p("attn_v"),
                    wo: p("attn_output"),
                    q_norm: p("attn_q_norm"),
                    k_norm: p("attn_k_norm"),
                    gate: p("attn_gate"),
                    ffn_norm: p("ffn_norm"),
                    ffn_gate: p("ffn_gate"),
                    ffn_up: p("ffn_up"),
                    ffn_down: p("ffn_down"),
                }
            })
            .collect();

        // Per-layer context K/V from the fused ctx (attn_norm -> wk/wv, K normed+roped).
        // ctxk[il][kv] = Vec of [hd] per context token.
        let mut ctxk = vec![vec![Vec::<Vec<f64>>::new(); n_kv]; cfg.n_layer];
        let mut ctxv = vec![vec![Vec::<Vec<f64>>::new(); n_kv]; cfg.n_layer];
        for (il, layer) in layers.iter().enumerate() {
            for (t, tok) in ctx.iter().enumerate() {
                let nn = rmsnorm(tok, &layer.attn_norm, eps);
                let kf = matvec(&layer.wk, &nn, n_kv * hd, n_embd);
                let vf = matvec(&layer.wv, &nn, n_kv * hd, n_embd);
                for kv in 0..n_kv {
                    let kh = &kf[kv * hd..(kv + 1) * hd];
                    let kn = rmsnorm(kh, &layer.k_norm, eps);
                    ctxk[il][kv].push(rope_vec(&kn, t, hd, theta));
                    ctxv[il][kv].push(vf[kv * hd..(kv + 1) * hd].to_vec());
                }
            }
        }

        let mut hs: Vec<Vec<f64>> = noise.to_vec();
        for (il, layer) in layers.iter().enumerate() {
            // Per-token projections for the block.
            let mut qh = vec![vec![Vec::<f64>::new(); n_head]; n_block]; // [b][head] -> [hd]
            let mut nk = vec![vec![Vec::<f64>::new(); n_kv]; n_block];
            let mut nv = vec![vec![Vec::<f64>::new(); n_kv]; n_block];
            let mut gates = vec![vec![0.0f64; n_head]; n_block];
            let mut norms = Vec::with_capacity(n_block);
            for b in 0..n_block {
                let pos = committed + b;
                let nn = rmsnorm(&hs[b], &layer.attn_norm, eps);
                let qf = matvec(&layer.wq, &nn, n_head * hd, n_embd);
                let kf = matvec(&layer.wk, &nn, n_kv * hd, n_embd);
                let vf = matvec(&layer.wv, &nn, n_kv * hd, n_embd);
                for h in 0..n_head {
                    let qn = rmsnorm(&qf[h * hd..(h + 1) * hd], &layer.q_norm, eps);
                    qh[b][h] = rope_vec(&qn, pos, hd, theta);
                }
                for kv in 0..n_kv {
                    let kn = rmsnorm(&kf[kv * hd..(kv + 1) * hd], &layer.k_norm, eps);
                    nk[b][kv] = rope_vec(&kn, pos, hd, theta);
                    nv[b][kv] = vf[kv * hd..(kv + 1) * hd].to_vec();
                }
                let gl = matvec(&layer.gate, &nn, n_head, n_embd);
                for h in 0..n_head {
                    gates[b][h] = softplus_s(gl[h]);
                }
                norms.push(nn);
            }
            let _ = norms;

            // Attention + gate + o_proj + residual + FFN, per block token.
            for b in 0..n_block {
                let mut attn_cat = vec![0.0f64; n_head * hd];
                for h in 0..n_head {
                    let kv = h / g;
                    // Keys: committed context (all), then noise 0..=b (causal).
                    let mut scores = Vec::with_capacity(committed + b + 1);
                    for t in 0..committed {
                        let s: f64 = qh[b][h].iter().zip(&ctxk[il][kv][t]).map(|(a, c)| a * c).sum();
                        scores.push(s * scale);
                    }
                    for bp in 0..=b {
                        let s: f64 = qh[b][h].iter().zip(&nk[bp][kv]).map(|(a, c)| a * c).sum();
                        scores.push(s * scale);
                    }
                    // Softmax.
                    let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
                    let sum: f64 = exps.iter().sum();
                    let mut ov = vec![0.0f64; hd];
                    for (idx, w) in exps.iter().map(|e| e / sum).enumerate() {
                        let vv = if idx < committed { &ctxv[il][kv][idx] } else { &nv[idx - committed][kv] };
                        for d in 0..hd {
                            ov[d] += w * vv[d];
                        }
                    }
                    let gate = gates[b][h];
                    for d in 0..hd {
                        attn_cat[h * hd + d] = ov[d] * gate;
                    }
                }
                let cur = matvec(&layer.wo, &attn_cat, n_embd, n_head * hd);
                let ffn_inp: Vec<f64> = cur.iter().zip(&hs[b]).map(|(a, b)| a + b).collect();
                let fn_ = rmsnorm(&ffn_inp, &layer.ffn_norm, eps);
                let gate = matvec(&layer.ffn_gate, &fn_, n_ff, n_embd);
                let up = matvec(&layer.ffn_up, &fn_, n_ff, n_embd);
                let hidden: Vec<f64> = gate.iter().zip(&up).map(|(gg, uu)| silu_s(*gg) * uu).collect();
                let down = matvec(&layer.ffn_down, &hidden, n_embd, n_ff);
                hs[b] = down.iter().zip(&ffn_inp).map(|(a, b)| a + b).collect();
            }
        }

        hs.iter().map(|h| rmsnorm(h, &host(m, "output_norm"), eps)).collect()
    }

    // --- tests ----------------------------------------------------------------

    /// Test 1: load the real drafter GGUF and check the parsed config plus the
    /// on-disk tensor shapes/dtypes (BF16 matmuls, F32 norms). Reads the 2.2GB
    /// file only — no target model process, safe to run.
    #[test]
    fn real_file_load_and_shapes() {
        if !std::path::Path::new(DRAFTER_PATH).exists() {
            eprintln!("skipping real_file_load_and_shapes: {DRAFTER_PATH} not present");
            return;
        }
        let dev = Device::Cpu;
        let gguf = crate::gguf::open(DRAFTER_PATH, &dev).unwrap();

        let cfg = DflashConfig::from_gguf(&gguf.content).unwrap();
        assert_eq!(cfg.n_layer, 6);
        assert_eq!(cfg.n_embd, 3072);
        assert_eq!(cfg.n_head, 72);
        assert_eq!(cfg.n_head_kv, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.n_ff, 12288);
        assert_eq!(cfg.block_size, 16);
        assert_eq!(cfg.target_layers, vec![2, 11, 20, 30, 39, 48]);
        // Layer-input ids translate to post-FFN l_out capture indices (t - 1).
        assert_eq!(cfg.spec_tap_layers().unwrap(), vec![1, 10, 19, 29, 38, 47]);
        assert_eq!(cfg.mask_token_id, 12);
        assert!((cfg.rms_eps - 1e-6).abs() < 1e-12);
        assert_eq!(cfg.rope_theta, 500_000.0);

        // On-disk tensor shapes and dtypes.
        let info = |n: &str| gguf.content.tensor_infos.get(n).unwrap_or_else(|| panic!("missing {n}"));
        let shape = |n: &str| info(n).shape.dims().to_vec();
        let dtype = |n: &str| info(n).ggml_dtype;

        assert_eq!(shape("fc.weight"), vec![3072, 18432]);
        assert_eq!(dtype("fc.weight"), GgmlDType::BF16);
        assert_eq!(shape("enc.aux_norm.weight"), vec![6, 3072]);
        assert_eq!(dtype("enc.aux_norm.weight"), GgmlDType::F32);
        assert_eq!(dtype("enc.output_norm.weight"), GgmlDType::F32);
        assert_eq!(dtype("output_norm.weight"), GgmlDType::F32);
        for il in 0..cfg.n_layer {
            let p = |n: &str| format!("blk.{il}.{n}.weight");
            assert_eq!(shape(&p("attn_q")), vec![9216, 3072]);
            assert_eq!(shape(&p("attn_k")), vec![1024, 3072]);
            assert_eq!(shape(&p("attn_v")), vec![1024, 3072]);
            assert_eq!(shape(&p("attn_output")), vec![3072, 9216]);
            assert_eq!(shape(&p("attn_gate")), vec![72, 3072]);
            assert_eq!(shape(&p("attn_q_norm")), vec![128]);
            assert_eq!(shape(&p("attn_k_norm")), vec![128]);
            assert_eq!(shape(&p("ffn_gate")), vec![12288, 3072]);
            assert_eq!(shape(&p("ffn_up")), vec![12288, 3072]);
            assert_eq!(shape(&p("ffn_down")), vec![3072, 12288]);
            assert_eq!(dtype(&p("attn_q")), GgmlDType::BF16);
            assert_eq!(dtype(&p("attn_q_norm")), GgmlDType::F32);
            assert_eq!(dtype(&p("attn_norm")), GgmlDType::F32);
        }

        // Full load materializes every weight dense f32 and builds the caches.
        let drafter = DflashDrafter::load(&gguf, &dev, 64).unwrap();
        assert_eq!(drafter.config().n_layer, 6);
        assert_eq!(drafter.committed_len(), 0);
    }

    /// Test 2: the encoder matches a hand-rolled scalar re-implementation.
    #[test]
    fn encoder_matches_scalar() {
        let dev = Device::Cpu;
        let (n_embd, seq) = (8usize, 3usize);
        let cfg = tiny_cfg(1, 2, 1, 4, n_embd, 16); // 2 target layers -> n_aux 2
        let n_aux = cfg.target_layers.len();
        let m = synth_weights(&cfg, &dev);

        let taps: Vec<Tensor> = (0..n_aux)
            .map(|a| Tensor::from_vec(seeded(seq * n_embd, 100 + a as u64), (seq, n_embd), &dev).unwrap())
            .collect();
        let drafter = drafter_from_map(cfg.clone(), &m, &dev, 32);
        let got = to_vec(&drafter.encode(&taps).unwrap());

        // Scalar reference.
        let aux_norm = host(&m, "enc.aux_norm");
        let enc_out = host(&m, "enc.output_norm");
        let fc = host(&m, "fc");
        let taps_h: Vec<Vec<f64>> = taps.iter().map(|t| to_vec(t).iter().map(|x| *x as f64).collect()).collect();
        let eps = cfg.rms_eps;
        let mut want = Vec::new();
        for s in 0..seq {
            // Per-aux: rms-norm (no weight) over n_embd, then * aux_norm[aux].
            let mut concat = vec![0.0f64; n_aux * n_embd];
            for a in 0..n_aux {
                let feat: Vec<f64> = (0..n_embd).map(|e| taps_h[a][s * n_embd + e]).collect();
                let ms: f64 = feat.iter().map(|v| v * v).sum::<f64>() / n_embd as f64;
                let sc = 1.0 / (ms + eps).sqrt();
                for e in 0..n_embd {
                    concat[a * n_embd + e] = feat[e] * sc * aux_norm[a * n_embd + e];
                }
            }
            let fused = matvec(&fc, &concat, n_embd, n_aux * n_embd);
            let normed = rmsnorm(&fused, &enc_out, eps);
            want.extend(normed.iter().map(|x| *x as f32));
        }
        let rel = rel_l2(&got, &want);
        assert!(rel < 1e-5, "encoder rel_l2 {rel} too high");
    }

    /// Test 3: draft_forward matches the independent scalar reference — verifies
    /// cache-vs-block equivalence, GQA, rope positions, causal masking, gate.
    #[test]
    fn draft_forward_matches_scalar() {
        let dev = Device::Cpu;
        let (n_embd, hd) = (8usize, 4usize);
        let cfg = tiny_cfg(2, 2, 1, hd, n_embd, 16);
        let m = synth_weights(&cfg, &dev);
        let mut drafter = drafter_from_map(cfg.clone(), &m, &dev, 32);

        let committed = 4usize;
        let n_block = 3usize;
        // Fused context (post-encoder residuals) and the noise block, at residual
        // scale so the block stays well-conditioned.
        let scale_in = 0.25f32;
        let fused = Tensor::from_vec(seeded(committed * n_embd, 7), (committed, n_embd), &dev)
            .unwrap()
            .affine(scale_in as f64, 0.0)
            .unwrap();
        let noise = Tensor::from_vec(seeded(n_block * n_embd, 8), (n_block, n_embd), &dev)
            .unwrap()
            .affine(scale_in as f64, 0.0)
            .unwrap();

        drafter.inject(&fused, 0).unwrap();
        assert_eq!(drafter.committed_len(), committed);
        let got = to_vec(&drafter.draft_forward(&noise, committed).unwrap());
        // draft_forward must not advance the committed cache.
        assert_eq!(drafter.committed_len(), committed);

        let to_rows = |t: &Tensor, rows: usize| -> Vec<Vec<f64>> {
            let flat = to_vec(t);
            (0..rows).map(|r| (0..n_embd).map(|c| flat[r * n_embd + c] as f64).collect()).collect()
        };
        let ctx = to_rows(&fused, committed);
        let nz = to_rows(&noise, n_block);
        let want_rows = naive_draft(&cfg, &m, &ctx, &nz);
        let want: Vec<f32> = want_rows.iter().flatten().map(|x| *x as f32).collect();

        let rel = rel_l2(&got, &want);
        assert!(rel < 1e-4, "draft_forward rel_l2 {rel} too high");
    }

    /// Test 4: cache hygiene — inject/draft/truncate roundtrips leave the cache
    /// in a clean state. A drafter that injects 5, drafts, injects 2 more must
    /// produce identical draft logits to a fresh drafter given the SAME final
    /// injections, proving draft_forward never mutates the committed cache and
    /// truncate rolls back cleanly.
    #[test]
    fn truncate_inject_roundtrip() {
        let dev = Device::Cpu;
        let (n_embd, hd) = (8usize, 4usize);
        let cfg = tiny_cfg(2, 2, 1, hd, n_embd, 16);
        let m = synth_weights(&cfg, &dev);

        let mk = |n: usize, seed: u64| {
            Tensor::from_vec(seeded(n * n_embd, seed), (n, n_embd), &dev).unwrap().affine(0.25, 0.0).unwrap()
        };
        let inj_a = mk(5, 1); // first injection (5 tokens)
        let inj_b = mk(2, 2); // second injection (2 tokens)
        let noise = mk(3, 3);

        // Path 1: inject 5, draft 3 (cache back to 5), truncate is implicit
        // (draft never advances), inject 2 more -> committed 7, then draft.
        let mut d1 = drafter_from_map(cfg.clone(), &m, &dev, 32);
        d1.inject(&inj_a, 0).unwrap();
        let _ = d1.draft_forward(&noise, 5).unwrap();
        assert_eq!(d1.committed_len(), 5);
        d1.inject(&inj_b, 5).unwrap();
        assert_eq!(d1.committed_len(), 7);
        let out1 = to_vec(&d1.draft_forward(&noise, 7).unwrap());

        // Path 2: a fresh drafter given the same two injections back to back.
        let mut d2 = drafter_from_map(cfg.clone(), &m, &dev, 32);
        d2.inject(&inj_a, 0).unwrap();
        d2.inject(&inj_b, 5).unwrap();
        let out2 = to_vec(&d2.draft_forward(&noise, 7).unwrap());

        assert_eq!(out1.len(), out2.len());
        for (i, (a, b)) in out1.iter().zip(&out2).enumerate() {
            assert!((a - b).abs() < 1e-6, "logit {i} differs: {a} vs {b}");
        }

        // Explicit truncate: roll back to 5 and re-inject 2 -> identical to d2.
        let mut d3 = drafter_from_map(cfg.clone(), &m, &dev, 32);
        d3.inject(&inj_a, 0).unwrap();
        d3.inject(&inj_b, 5).unwrap();
        d3.truncate(5).unwrap();
        assert_eq!(d3.committed_len(), 5);
        d3.inject(&inj_b, 5).unwrap();
        let out3 = to_vec(&d3.draft_forward(&noise, 7).unwrap());
        for (a, b) in out3.iter().zip(&out2) {
            assert!((a - b).abs() < 1e-6, "post-truncate logit differs: {a} vs {b}");
        }
    }
}
