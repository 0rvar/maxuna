use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};

use crate::config::LagunaConfig;
use crate::gguf::{QLinear, Weights};
use crate::kv_cache::LayerCache;
use crate::rope::Rope;

/// How the attention projection weights are held, decided at load (model.rs,
/// `LAGUNA_ATTN_F32`). Activations are f32 either way — the `F16` mode streams
/// the weights at their GGUF-stored width through the vendored mixed-dtype
/// kernels (ops::matmul_f16), so the stored f16 weights are the only f16 in
/// the projection math (the fork's exact precision structure).
#[derive(Clone, Copy)]
pub enum AttnWeights {
    /// GGUF-stored f16 weights kept dense f16 (shipped default; Metal only).
    F16,
    /// Weights dequantized to dense f32 behind `QMatMul` (fully legacy).
    DequantF32,
}

/// One attention projection. `DenseF16` holds the weight as a dense f16 tensor
/// consumed by the vendored ggml-geometry f16-weight kernels: the f32
/// activation is never cast and the output is written f32 — f16 weight
/// streaming (the bandwidth win) with zero non-weight rounding. `Quant` keeps
/// the GGUF tensor behind candle's `QMatMul`, which dequantizes the F16-stored
/// attention weights to dense f32 at load.
enum Proj {
    Quant(QLinear),
    /// `[out_dim, in_dim]` f16.
    DenseF16(Tensor),
}

impl Proj {
    /// f32 in, f32 out on both variants; on `DenseF16` the stored f16 weights
    /// are the only f16 the matmul ever sees.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Proj::Quant(q) => Ok(q.forward(x)?),
            Proj::DenseF16(w) => crate::ops::matmul_f16(w, x),
        }
    }
}

/// One Laguna attention block. Order (fork laguna.cpp:200-265):
/// gate logits come from the pre-attention normed input; q/k get per-head-dim
/// RMSNorm before rope; sdpa at scale 1/sqrt(head_dim) in f16; the softplus
/// gate multiplies attention output per-head (broadcast over head_dim) BEFORE
/// o_proj. Q-head count is per-layer (48 full / 72 SWA); KV heads always 8.
///
/// Activations are f32 end-to-end (matching the fork's precision structure);
/// with `AttnWeights::F16` each projection runs the vendored mixed-dtype
/// kernels (f16 weights x f32 activations, f32 accumulate/output), so the
/// GGUF-stored f16 weights stream at their stored width with no other
/// rounding. f16 otherwise appears only where both weight modes share it: the
/// KV cache and the sdpa kernel.
pub struct AttnBlock {
    q_proj: Proj,
    k_proj: Proj,
    v_proj: Proj,
    g_proj: Proj,
    o_proj: Proj,
    /// QK-norm weights, f32 (candle's rms_norm requires weight dtype == x dtype).
    q_norm: candle_nn::RmsNorm,
    k_norm: candle_nn::RmsNorm,
    rope: Arc<Rope>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
}

impl AttnBlock {
    /// `w` is positioned at the block prefix (e.g. `blk.7`).
    pub fn new(
        w: &Weights,
        cfg: &LagunaConfig,
        il: usize,
        rope: Arc<Rope>,
        weights: AttnWeights,
    ) -> Result<Self> {
        let proj = |name: &str| -> Result<Proj> {
            Ok(match weights {
                AttnWeights::F16 => Proj::DenseF16(w.dense_f16(name)?),
                AttnWeights::DequantF32 => Proj::Quant(w.qlinear(name)?),
            })
        };
        let norm = |name: &str| -> Result<candle_nn::RmsNorm> {
            Ok(candle_nn::RmsNorm::new(w.dense_f32(name)?, cfg.rms_eps))
        };
        Ok(Self {
            q_proj: proj("attn_q")?,
            k_proj: proj("attn_k")?,
            v_proj: proj("attn_v")?,
            g_proj: proj("attn_gate")?,
            o_proj: proj("attn_output")?,
            q_norm: norm("attn_q_norm")?,
            k_norm: norm("attn_k_norm")?,
            rope,
            n_head: cfg.n_head(il),
            n_kv_head: cfg.n_kv_head,
            head_dim: cfg.head_dim,
        })
    }

    /// x_normed: [seq, hidden] f32 (already attn_norm'ed by the caller).
    /// Returns [seq, hidden] f32.
    pub fn forward(&self, x_normed: &Tensor, cache: &mut LayerCache, pos: usize) -> Result<Tensor> {
        let (seq, _hidden) = x_normed.dims2()?;

        // Gate logits from the *pre-attention* normed input (not the attn output).
        let gate_logits = self.g_proj.forward(x_normed)?; // [seq, n_head] f32

        let q = self.q_proj.forward(x_normed)?.reshape((seq, self.n_head, self.head_dim))?;
        let k = self.k_proj.forward(x_normed)?.reshape((seq, self.n_kv_head, self.head_dim))?;
        let v = self.v_proj.forward(x_normed)?.reshape((seq, self.n_kv_head, self.head_dim))?;

        // QK-norm: RMSNorm over head_dim before rope, in [seq, head, dim]
        // layout where head_dim is contiguous last.
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // To [head, seq, head_dim] for rope + attention. At seq==1 (decode) the
        // [1, head, dim] and [head, 1, dim] layouts share byte order, so a reshape
        // (metadata only) is bit-identical to transpose+contiguous and drops three
        // copy dispatches per layer on the hot decode path. seq>1 (prefill) is a
        // real permutation, so it keeps the transpose+contiguous.
        let (q, k, v) = if seq == 1 {
            (
                q.reshape((self.n_head, 1, self.head_dim))?,
                k.reshape((self.n_kv_head, 1, self.head_dim))?,
                v.reshape((self.n_kv_head, 1, self.head_dim))?,
            )
        } else {
            (
                q.transpose(0, 1)?.contiguous()?,
                k.transpose(0, 1)?.contiguous()?,
                v.transpose(0, 1)?.contiguous()?,
            )
        };

        let (q, k) = self.rope.apply(&q, &k, pos)?;

        // Cache in f16; sdpa runs in f16.
        let (k_all, v_all) = cache.append(&k.to_dtype(DType::F16)?, &v.to_dtype(DType::F16)?)?;
        let mask = cache.attn_mask(seq, pos)?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();

        // sdpa is a Metal-only kernel; fall back to an explicit f32 attention on
        // other devices (used by the CPU tests and any non-Metal run).
        let attn = if matches!(x_normed.device(), Device::Metal(_)) {
            self.sdpa_attention(&q, &k_all, &v_all, mask.as_ref(), scale)?
        } else {
            self.manual_attention(&q, &k_all, &v_all, mask.as_ref(), scale, seq)?
        }; // [n_head, seq, head_dim] f32

        // Softplus output gate, per-head, broadcast over head_dim, in f32.
        let gate = softplus(&gate_logits)?.transpose(0, 1)?.reshape((self.n_head, seq, 1))?;
        let attn = attn.broadcast_mul(&gate)?;

        // Back to [seq, n_head*head_dim] then o_proj. Same seq==1 shortcut: the
        // [head, 1, dim] -> [1, head*dim] regroup is byte-identical to
        // transpose+contiguous+reshape, so decode skips the copy.
        let out = if seq == 1 {
            attn.reshape((seq, self.n_head * self.head_dim))?
        } else {
            attn.transpose(0, 1)?.contiguous()?.reshape((seq, self.n_head * self.head_dim))?
        };
        self.o_proj.forward(&out)
    }

    /// Metal MLX fused attention. q [n_head, seq, hd] f32, k/v
    /// [n_kv_head, K, hd] f16. GQA (n_head multiple of n_kv_head) is handled by
    /// the kernel; k/v are not pre-tiled. The kernel runs in f16 (q and the
    /// mask are cast below). Returns [n_head, seq, hd] f32.
    fn sdpa_attention(
        &self,
        q: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
        mask: Option<&Tensor>,
        scale: f32,
    ) -> Result<Tensor> {
        let q = q.to_dtype(DType::F16)?.unsqueeze(0)?.contiguous()?; // [1, n_head, seq, hd]
        // k/v stay as the cache's narrowed views: rows within a head are
        // contiguous and only the head dimension carries the max_ctx gap, which
        // sdpa handles via the per-head k/v stride it is passed. Materializing a
        // packed copy here would grow with context for no benefit.
        let k = k_all.unsqueeze(0)?; // [1, n_kv_head, K, hd], head-strided
        let v = v_all.unsqueeze(0)?;

        // Full-kernel masks must match q dtype and carry the query-head dim.
        let mask = match mask {
            Some(m) => {
                let (s, kk) = m.dims2()?;
                Some(
                    m.reshape((1, 1, s, kk))?
                        .broadcast_as((1, self.n_head, s, kk))?
                        .to_dtype(DType::F16)?
                        .contiguous()?,
                )
            }
            None => None,
        };

        let out = candle_nn::ops::sdpa(&q, &k, &v, mask.as_ref(), false, scale, 1.0)?;
        Ok(out.squeeze(0)?.to_dtype(DType::F32)?)
    }

    /// Explicit softmax(q·kᵀ·scale + mask)·v in q's dtype (f32), GQA via a
    /// broadcast over the query group dim. q [n_head, seq, hd] f32, k/v
    /// [n_kv_head, K, hd] f16. Non-Metal fallback (CPU tests, Reference oracle).
    fn manual_attention(
        &self,
        q: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
        mask: Option<&Tensor>,
        scale: f32,
        seq: usize,
    ) -> Result<Tensor> {
        let g = self.n_head / self.n_kv_head;
        let k_seq = k_all.dim(1)?;
        let k = k_all.to_dtype(q.dtype())?;
        let v = v_all.to_dtype(q.dtype())?;

        let q4 = q.reshape((self.n_kv_head, g, seq, self.head_dim))?;
        let k4 = k.reshape((self.n_kv_head, 1, k_seq, self.head_dim))?;
        let v4 = v.reshape((self.n_kv_head, 1, k_seq, self.head_dim))?;

        let scores = q4.broadcast_matmul(&k4.transpose(2, 3)?)?.affine(scale as f64, 0.0)?;
        let scores = match mask {
            // The additive mask is built f32, matching the scores' dtype.
            Some(m) => scores.broadcast_add(&m.to_dtype(scores.dtype())?.reshape((1, 1, seq, k_seq))?)?,
            None => scores,
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.broadcast_matmul(&v4)?; // [n_kv_head, g, seq, hd]
        Ok(out.reshape((self.n_head, seq, self.head_dim))?)
    }
}

/// Numerically stable softplus, ln(1 + exp(x)) = relu(x) + ln(1 + exp(-|x|)).
fn softplus(x: &Tensor) -> Result<Tensor> {
    let ax = x.abs()?;
    let relu = x.broadcast_add(&ax)?.affine(0.5, 0.0)?;
    let tail = ax.neg()?.exp()?.affine(1.0, 1.0)?.log()?;
    Ok(relu.broadcast_add(&tail)?)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::config::RopeKind;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::Device;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    struct RawWeights {
        wq: Tensor, // [n_head*hd, hidden]
        wk: Tensor, // [n_kv*hd, hidden]
        wv: Tensor,
        wg: Tensor, // [n_head, hidden]
        wo: Tensor, // [hidden, n_head*hd]
        qn: Tensor, // [hd]
        kn: Tensor,
    }

    fn dense(rows: usize, cols: usize, seed: u64, dev: &Device) -> Tensor {
        Tensor::from_vec(seeded(rows * cols, seed), (rows, cols), dev).unwrap()
    }

    fn raw_weights(n_head: usize, n_kv: usize, hd: usize, hidden: usize, dev: &Device) -> RawWeights {
        RawWeights {
            wq: dense(n_head * hd, hidden, 1, dev),
            wk: dense(n_kv * hd, hidden, 2, dev),
            wv: dense(n_kv * hd, hidden, 3, dev),
            wg: dense(n_head, hidden, 4, dev),
            wo: dense(hidden, n_head * hd, 5, dev),
            // Norm weights near 1.0 so the RMSNorm stays well-conditioned.
            qn: Tensor::from_vec(seeded(hd, 6).iter().map(|x| 1.0 + 0.1 * x).collect(), hd, dev).unwrap(),
            kn: Tensor::from_vec(seeded(hd, 7).iter().map(|x| 1.0 + 0.1 * x).collect(), hd, dev).unwrap(),
        }
    }

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    /// Write the weights to a throwaway GGUF (F32 quant) and load an AttnBlock,
    /// exercising the real gguf.rs loading seam rather than a test-only shortcut.
    fn build_block(
        w: &RawWeights,
        cfg: &LagunaConfig,
        il: usize,
        rope: Arc<Rope>,
        dev: &Device,
        weights: AttnWeights,
    ) -> AttnBlock {
        let q = |t: &Tensor| QTensor::quantize(&t.to_device(&Device::Cpu).unwrap(), GgmlDType::F32).unwrap();
        let (wq, wk, wv, wg, wo, qn, kn) =
            (q(&w.wq), q(&w.wk), q(&w.wv), q(&w.wg), q(&w.wo), q(&w.qn), q(&w.kn));
        let tensors: Vec<(&str, &QTensor)> = vec![
            ("blk.0.attn_q.weight", &wq),
            ("blk.0.attn_k.weight", &wk),
            ("blk.0.attn_v.weight", &wv),
            ("blk.0.attn_gate.weight", &wg),
            ("blk.0.attn_output.weight", &wo),
            ("blk.0.attn_q_norm.weight", &qn),
            ("blk.0.attn_k_norm.weight", &kn),
        ];
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path: PathBuf = std::env::temp_dir().join(format!("laguna_attn_test_{id}.gguf"));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            candle_core::quantized::gguf_file::write(&mut f, &[], &tensors).unwrap();
        }
        let src = crate::gguf::open(&path, dev).unwrap();
        let loaded = Weights::from_gguf(src).pp("blk.0");
        let block = AttnBlock::new(&loaded, cfg, il, rope, weights).unwrap();
        let _ = std::fs::remove_file(&path);
        block
    }

    fn test_cfg(n_head: usize, n_kv: usize, hd: usize, hidden: usize, swa_period: usize, window: usize) -> LagunaConfig {
        LagunaConfig {
            n_layer: 4,
            hidden,
            vocab: 32,
            n_head: vec![n_head; 4],
            n_kv_head: n_kv,
            head_dim: hd,
            dense_layers: 1,
            dense_ff: 32,
            n_expert: 1,
            n_expert_used: 1,
            expert_ff: 32,
            shared_expert_ff: 32,
            expert_weights_scale: 1.0,
            expert_weights_norm: true,
            rms_eps: 1e-6,
            sliding_window: window,
            swa_period,
            n_ctx_train: 4096,
            rope_full: RopeKind::Plain { freq_base: 10_000.0, n_rot: hd },
            rope_swa: RopeKind::Plain { freq_base: 10_000.0, n_rot: hd },
            bos_token: 1,
            eog_tokens: vec![2],
        }
    }

    // --- independent naive reference -----------------------------------------

    /// A from-scratch f32 attention over the full token sequence, causal or with
    /// the SWA window, replicating qk-norm + rope + f16 cache + softplus gate.
    /// Returns [total, hidden].
    fn naive_forward(
        w: &RawWeights,
        rope: &Rope,
        x: &Tensor,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        window: Option<usize>,
    ) -> Tensor {
        let dev = x.device();
        let (total, _hidden) = x.dims2().unwrap();
        let eps = 1e-6f64;

        let lin = |x: &Tensor, wt: &Tensor| x.matmul(&wt.t().unwrap()).unwrap();
        let rms = |t: &Tensor, weight: &Tensor| {
            // t: [head, seq, hd]; normalize over hd.
            let ms = (t.sqr().unwrap().sum_keepdim(2).unwrap() / hd as f64).unwrap();
            let normed = t.broadcast_div(&(ms + eps).unwrap().sqrt().unwrap()).unwrap();
            normed.broadcast_mul(&weight.reshape((1, 1, hd)).unwrap()).unwrap()
        };

        let gate = softplus(&lin(x, &w.wg)).unwrap(); // [total, n_head]
        let q = lin(x, &w.wq).reshape((total, n_head, hd)).unwrap();
        let k = lin(x, &w.wk).reshape((total, n_kv, hd)).unwrap();
        let v = lin(x, &w.wv).reshape((total, n_kv, hd)).unwrap();

        let q = rms(&q.transpose(0, 1).unwrap().contiguous().unwrap(), &w.qn);
        let k = rms(&k.transpose(0, 1).unwrap().contiguous().unwrap(), &w.kn);
        let v = v.transpose(0, 1).unwrap().contiguous().unwrap();
        let (q, k) = rope.apply(&q, &k, 0).unwrap();
        // Round-trip k/v through f16 exactly as the cache does.
        let k = k.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap();
        let v = v.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap();

        let g = n_head / n_kv;
        let scale = 1.0 / (hd as f64).sqrt();
        let q4 = q.reshape((n_kv, g, total, hd)).unwrap();
        let k4 = k.reshape((n_kv, 1, total, hd)).unwrap();
        let v4 = v.reshape((n_kv, 1, total, hd)).unwrap();
        let scores = q4.broadcast_matmul(&k4.transpose(2, 3).unwrap()).unwrap();
        let scores = (scores * scale).unwrap();

        let mut mask = vec![0f32; total * total];
        for qi in 0..total {
            for kj in 0..total {
                let blocked = kj > qi || window.is_some_and(|wsz| qi - kj >= wsz);
                if blocked {
                    mask[qi * total + kj] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(mask, (1, 1, total, total), dev).unwrap();
        let scores = scores.broadcast_add(&mask).unwrap();
        let probs = candle_nn::ops::softmax_last_dim(&scores).unwrap();
        let attn = probs.broadcast_matmul(&v4).unwrap().reshape((n_head, total, hd)).unwrap();

        let gate = gate.transpose(0, 1).unwrap().reshape((n_head, total, 1)).unwrap();
        let attn = attn.broadcast_mul(&gate).unwrap();
        let out = attn.transpose(0, 1).unwrap().contiguous().unwrap().reshape((total, n_head * hd)).unwrap();
        lin(&out, &w.wo)
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    /// ||a - b||2 / ||b||2, with b the reference.
    fn rel_l2(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let num: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
        let den: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
        num / den.max(1e-30)
    }

    // --- tests ----------------------------------------------------------------

    /// Test 1: prefill (seq 4) then a decode step (pos 4) match a single full
    /// causal reference pass over all 5 tokens — verifying cache append+mask.
    #[test]
    fn forward_matches_naive_prefill_and_decode() {
        let dev = Device::Cpu;
        let (n_head, n_kv, hd, hidden) = (2, 1, 8, 16);
        let cfg = test_cfg(n_head, n_kv, hd, hidden, 4, 512); // il 0 => full attn
        let rope = Arc::new(Rope::new(cfg.rope(0), 64, &dev).unwrap());
        let w = raw_weights(n_head, n_kv, hd, hidden, &dev);
        let block = build_block(&w, &cfg, 0, rope.clone(), &dev, AttnWeights::DequantF32);
        let mut cache = LayerCache::new(&cfg, 0, 64, &dev).unwrap();

        let x = dense(5, hidden, 42, &dev);
        let reference = naive_forward(&w, &rope, &x, n_head, n_kv, hd, None);

        let prefill = block.forward(&x.narrow(0, 0, 4).unwrap(), &mut cache, 0).unwrap();
        assert!(max_abs_diff(&prefill, &reference.narrow(0, 0, 4).unwrap()) < 1e-3);

        let decode = block.forward(&x.narrow(0, 4, 1).unwrap(), &mut cache, 4).unwrap();
        assert!(max_abs_diff(&decode, &reference.narrow(0, 4, 1).unwrap()) < 1e-3);
    }

    /// Test 2: SWA ring — feed 20 tokens one at a time (window 8) and check each
    /// step equals the full-cache reference with the window mask, across both the
    /// first wrap (pos 8..16) and the second (16..20).
    #[test]
    fn swa_ring_matches_windowed_full_cache() {
        let dev = Device::Cpu;
        let (n_head, n_kv, hd, hidden, window) = (2, 1, 8, 16, 8);
        let cfg = test_cfg(n_head, n_kv, hd, hidden, 4, window);
        let rope = Arc::new(Rope::new(cfg.rope(1), 64, &dev).unwrap());
        let w = raw_weights(n_head, n_kv, hd, hidden, &dev);
        let block = build_block(&w, &cfg, 1, rope.clone(), &dev, AttnWeights::DequantF32); // il 1 => SWA
        let mut cache = LayerCache::new(&cfg, 1, 64, &dev).unwrap();

        let total = 20;
        let x = dense(total, hidden, 99, &dev);
        for t in 0..total {
            let step = block.forward(&x.narrow(0, t, 1).unwrap(), &mut cache, t).unwrap();
            let reference = naive_forward(&w, &rope, &x.narrow(0, 0, t + 1).unwrap(), n_head, n_kv, hd, Some(window));
            let diff = max_abs_diff(&step, &reference.narrow(0, t, 1).unwrap());
            assert!(diff < 1e-3, "step {t} diff {diff}");
        }
    }

    /// Test 3: exact mask matrices for a Full cache and a boundary-crossing SWA.
    #[test]
    fn attn_mask_content() {
        let dev = Device::Cpu;
        let inf = f32::NEG_INFINITY;

        let cfg_full = test_cfg(2, 1, 8, 16, 4, 8);
        let full = LayerCache::new(&cfg_full, 0, 32, &dev).unwrap();
        let m = full.attn_mask(3, 2).unwrap().unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(
            m,
            vec![
                vec![0.0, 0.0, 0.0, inf, inf],
                vec![0.0, 0.0, 0.0, 0.0, inf],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        );

        // SWA window 4, 3 queries at pos 3 => columns are abs positions 0..6.
        let cfg_swa = test_cfg(2, 1, 8, 16, 4, 4);
        let swa = LayerCache::new(&cfg_swa, 1, 32, &dev).unwrap();
        let m = swa.attn_mask(3, 3).unwrap().unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(
            m,
            vec![
                vec![0.0, 0.0, 0.0, 0.0, inf, inf],
                vec![inf, 0.0, 0.0, 0.0, 0.0, inf],
                vec![inf, inf, 0.0, 0.0, 0.0, 0.0],
            ]
        );

        // Single decode token never needs a mask.
        assert!(full.attn_mask(1, 10).unwrap().is_none());
        assert!(swa.attn_mask(1, 10).unwrap().is_none());
    }

    /// Test 4: the Metal sdpa path matches the f32 CPU reference (f16 tolerance),
    /// for full-attn prefill, a decode step, and an SWA windowed prefill.
    #[test]
    fn metal_sdpa_matches_reference() {
        let dev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return, // no Metal device: nothing to compare
        };
        let (n_head, n_kv, hd, hidden, window) = (48, 8, 128, 256, 512);

        // Full-attention prefill (seq 6) and a follow-up decode step.
        let cfg = test_cfg(n_head, n_kv, hd, hidden, 4, window);
        let rope = Arc::new(Rope::new(cfg.rope(0), 128, &dev).unwrap());
        let w = raw_weights(n_head, n_kv, hd, hidden, &dev);
        let block = build_block(&w, &cfg, 0, rope.clone(), &dev, AttnWeights::DequantF32);
        let mut cache = LayerCache::new(&cfg, 0, 128, &dev).unwrap();

        let x = dense(7, hidden, 7, &dev);
        let reference = naive_forward(&w, &rope, &x, n_head, n_kv, hd, None);

        let prefill = block.forward(&x.narrow(0, 0, 6).unwrap(), &mut cache, 0).unwrap();
        let ref_prefill = reference.narrow(0, 0, 6).unwrap();
        assert!(cosine(&prefill, &ref_prefill) > 0.999, "prefill cos {}", cosine(&prefill, &ref_prefill));

        let decode = block.forward(&x.narrow(0, 6, 1).unwrap(), &mut cache, 6).unwrap();
        let ref_decode = reference.narrow(0, 6, 1).unwrap();
        assert!(cosine(&decode, &ref_decode) > 0.999, "decode cos {}", cosine(&decode, &ref_decode));

        // SWA windowed prefill exercises the full sdpa kernel with a real mask.
        let win = 4;
        let cfg_swa = test_cfg(n_head, n_kv, hd, hidden, 4, win);
        let rope_swa = Arc::new(Rope::new(cfg_swa.rope(1), 128, &dev).unwrap());
        let block_swa = build_block(&w, &cfg_swa, 1, rope_swa.clone(), &dev, AttnWeights::DequantF32);
        let mut cache_swa = LayerCache::new(&cfg_swa, 1, 128, &dev).unwrap();
        let xs = dense(6, hidden, 8, &dev);
        let ref_swa = naive_forward(&w, &rope_swa, &xs, n_head, n_kv, hd, Some(win));
        let swa = block_swa.forward(&xs, &mut cache_swa, 0).unwrap();
        assert!(cosine(&swa, &ref_swa) > 0.999, "swa cos {}", cosine(&swa, &ref_swa));
    }

    /// Test 4b: the f16-weight path (f16 weights through the vendored
    /// mixed-dtype kernels; f32 activations end-to-end) matches the f32
    /// reference within f16 WEIGHT rounding — the only rounding the projections
    /// carry — for full-attn prefill, a decode step, and an SWA windowed
    /// prefill (which exercises the mask through sdpa). The prefill seqs sit
    /// above the mv/mm dispatch threshold (8) so the tiled gemm runs; decode
    /// exercises the gemv. Metal only — the f16-weight path targets the Metal
    /// kernels. Model-level acceptance is the parity gates.
    #[test]
    fn f16_block_matches_reference() {
        let dev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return, // no Metal device: nothing to compare
        };
        let (n_head, n_kv, hd, hidden, window) = (48, 8, 128, 256, 512);

        let cfg = test_cfg(n_head, n_kv, hd, hidden, 4, window);
        let rope = Arc::new(Rope::new(cfg.rope(0), 128, &dev).unwrap());
        let w = raw_weights(n_head, n_kv, hd, hidden, &dev);
        let block = build_block(&w, &cfg, 0, rope.clone(), &dev, AttnWeights::F16);
        let mut cache = LayerCache::new(&cfg, 0, 128, &dev).unwrap();

        // The synthetic weights at this geometry drive gate logits to ~±19 and
        // the gated block output to ~±4e5; trained weights keep the block at
        // residual scale, so shrink the probe input to keep the comparison
        // well-conditioned — the reference sees the same scaled input.
        let x = dense(34, hidden, 7, &dev).affine(0.125, 0.0).unwrap();
        let reference = naive_forward(&w, &rope, &x, n_head, n_kv, hd, None);
        // Measured on this probe: rel_l2 ~8.2e-6, cos ~0.9999996 — weight
        // rounding only; the per-matmul activation-cast and output-rounding
        // noise of the old cast-based path (which measured ~1.8e-4 here) is
        // gone. Bounds hold ~5x headroom.
        let check = |got: &Tensor, want: &Tensor, what: &str| {
            assert_eq!(got.dtype(), DType::F32, "{what}: block output must return f32");
            let (cos, rel) = (cosine(got, want), rel_l2(got, want));
            assert!(cos > 0.999998 && rel < 5e-5, "{what}: cos {cos} rel_l2 {rel}");
        };

        let prefill = block.forward(&x.narrow(0, 0, 33).unwrap(), &mut cache, 0).unwrap();
        check(&prefill, &reference.narrow(0, 0, 33).unwrap(), "prefill");

        let decode = block.forward(&x.narrow(0, 33, 1).unwrap(), &mut cache, 33).unwrap();
        check(&decode, &reference.narrow(0, 33, 1).unwrap(), "decode");

        let win = 4;
        let cfg_swa = test_cfg(n_head, n_kv, hd, hidden, 4, win);
        let rope_swa = Arc::new(Rope::new(cfg_swa.rope(1), 128, &dev).unwrap());
        let block_swa = build_block(&w, &cfg_swa, 1, rope_swa.clone(), &dev, AttnWeights::F16);
        let mut cache_swa = LayerCache::new(&cfg_swa, 1, 128, &dev).unwrap();
        let xs = dense(12, hidden, 8, &dev).affine(0.125, 0.0).unwrap();
        let ref_swa = naive_forward(&w, &rope_swa, &xs, n_head, n_kv, hd, Some(win));
        let swa = block_swa.forward(&xs, &mut cache_swa, 0).unwrap();
        check(&swa, &ref_swa, "swa prefill");
    }

    /// Decode-attention perf benches (phase 0 of the attention-fusion work).
    /// Synthetic weights at production geometry — never loads a model file.
    /// All are `#[ignore]`d; run one at a time with e.g.
    /// `cargo test --release attn_decode_chain_bench -- --ignored --nocapture`.
    /// Iteration counts: LAGUNA_BENCH_WARMUP (default 10) / LAGUNA_BENCH_ITERS
    /// (default 50). Each iter ends in one small CPU readback so it measures
    /// end-to-end latency including the command-buffer flush, mirroring
    /// `plain_mv_lmhead_bench`.
    pub(crate) mod decode_bench {
        use super::*;
        use candle_nn::RmsNorm;
        use std::time::Instant;

        const HIDDEN: usize = 3072;
        const HEAD_DIM: usize = 128;
        const N_KV: usize = 8;
        const N_LAYER: usize = 48;
        const WINDOW: usize = 512;
        /// The timed token decodes at this absolute position (realistic sdpa
        /// cost: full layers see POS+1 keys, SWA layers a full 512-slot ring).
        const POS: usize = 512;
        const MAX_CTX: usize = 1024;

        fn n_head_of(il: usize) -> usize {
            if il % 4 == 0 { 48 } else { 72 }
        }

        fn metal() -> Device {
            Device::new_metal(0).expect("decode benches require the Metal device")
        }

        fn iter_counts() -> (usize, usize) {
            let get = |k: &str, d: usize| {
                std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
            };
            (get("LAGUNA_BENCH_WARMUP", 10), get("LAGUNA_BENCH_ITERS", 50))
        }

        /// Only the attention-relevant fields matter (kv-cache geometry + SWA
        /// period); the FFN/rope fields are placeholders — benches build their
        /// Rope tables directly.
        pub(crate) fn prod_cfg() -> LagunaConfig {
            LagunaConfig {
                n_layer: N_LAYER,
                hidden: HIDDEN,
                vocab: 32,
                n_head: (0..N_LAYER).map(n_head_of).collect(),
                n_kv_head: N_KV,
                head_dim: HEAD_DIM,
                dense_layers: 1,
                dense_ff: 32,
                n_expert: 1,
                n_expert_used: 1,
                expert_ff: 32,
                shared_expert_ff: 32,
                expert_weights_scale: 1.0,
                expert_weights_norm: true,
                rms_eps: 1e-6,
                sliding_window: WINDOW,
                swa_period: 4,
                n_ctx_train: MAX_CTX,
                rope_full: RopeKind::Plain { freq_base: 500_000.0, n_rot: 64 },
                rope_swa: RopeKind::Plain { freq_base: 10_000.0, n_rot: HEAD_DIM },
                bos_token: 1,
                eog_tokens: vec![2],
            }
        }

        /// Production rope tables: full layers YaRN partial-rotary 64/128
        /// (θ=500k), SWA plain over all 128 dims (θ=10k). The exact YaRN
        /// scaling values are perf-irrelevant (same table lookup); n_rot is the
        /// load-bearing part (partial rotary costs extra narrow/contiguous/cat
        /// dispatches).
        fn build_ropes(dev: &Device) -> (Arc<Rope>, Arc<Rope>) {
            let yarn = RopeKind::Yarn {
                freq_base: 500_000.0,
                factor: 32.0,
                original_ctx: 8192,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attn_factor: 1.3466,
                n_rot: 64,
            };
            let plain = RopeKind::Plain { freq_base: 10_000.0, n_rot: HEAD_DIM };
            (
                Arc::new(Rope::new(&yarn, MAX_CTX, dev).unwrap()),
                Arc::new(Rope::new(&plain, MAX_CTX, dev).unwrap()),
            )
        }

        /// One layer's weights as dense f32 Metal tensors (for the hand-written
        /// chain) plus, optionally, a real `AttnBlock` built from the SAME
        /// values through `QLinear::from_qtensor` — which dequantizes an F32
        /// QTensor to exactly the dense-f32 `QMatMul::Tensor` form that gguf
        /// loading produces for the model's F16 attention weights.
        pub(crate) struct BenchLayer {
            attn_norm: RmsNorm,
            wq: Tensor,
            wk: Tensor,
            wv: Tensor,
            wg: Tensor,
            wo: Tensor,
            q_norm: RmsNorm,
            k_norm: RmsNorm,
            rope: Arc<Rope>,
            n_head: usize,
            block: Option<AttnBlock>,
        }

        fn norm_w(dim: usize, seed: u64, dev: &Device) -> Tensor {
            let v: Vec<f32> = seeded(dim, seed).iter().map(|x| 1.0 + 0.1 * x).collect();
            Tensor::from_vec(v, dim, dev).unwrap()
        }

        pub(crate) fn build_layers(dev: &Device, with_blocks: bool) -> Vec<BenchLayer> {
            let (rope_full, rope_swa) = build_ropes(dev);
            (0..N_LAYER)
                .map(|il| {
                    let h = n_head_of(il);
                    let s = il as u64 * 100;
                    let cpu = |rows: usize, cols: usize, seed: u64| {
                        Tensor::from_vec(seeded(rows * cols, seed), (rows, cols), &Device::Cpu)
                            .unwrap()
                    };
                    let wq = cpu(h * HEAD_DIM, HIDDEN, s + 1);
                    let wk = cpu(N_KV * HEAD_DIM, HIDDEN, s + 2);
                    let wv = cpu(N_KV * HEAD_DIM, HIDDEN, s + 3);
                    let wg = cpu(h, HIDDEN, s + 4);
                    let wo = cpu(HIDDEN, h * HEAD_DIM, s + 5);
                    let qn = norm_w(HEAD_DIM, s + 6, dev);
                    let kn = norm_w(HEAD_DIM, s + 7, dev);
                    let an = norm_w(HIDDEN, s + 8, dev);
                    let rope = if il % 4 == 0 { rope_full.clone() } else { rope_swa.clone() };
                    let block = with_blocks.then(|| {
                        let ql = |t: &Tensor| {
                            let qt = QTensor::quantize_onto(t, GgmlDType::F32, dev).unwrap();
                            QLinear::from_qtensor(Arc::new(qt)).unwrap()
                        };
                        AttnBlock {
                            q_proj: Proj::Quant(ql(&wq)),
                            k_proj: Proj::Quant(ql(&wk)),
                            v_proj: Proj::Quant(ql(&wv)),
                            g_proj: Proj::Quant(ql(&wg)),
                            o_proj: Proj::Quant(ql(&wo)),
                            q_norm: RmsNorm::new(qn.clone(), 1e-6),
                            k_norm: RmsNorm::new(kn.clone(), 1e-6),
                            rope: rope.clone(),
                            n_head: h,
                            n_kv_head: N_KV,
                            head_dim: HEAD_DIM,
                        }
                    });
                    let up = |t: &Tensor| t.to_device(dev).unwrap();
                    BenchLayer {
                        attn_norm: RmsNorm::new(an, 1e-6),
                        wq: up(&wq),
                        wk: up(&wk),
                        wv: up(&wv),
                        wg: up(&wg),
                        wo: up(&wo),
                        q_norm: RmsNorm::new(qn, 1e-6),
                        k_norm: RmsNorm::new(kn, 1e-6),
                        rope,
                        n_head: h,
                        block,
                    }
                })
                .collect()
        }

        /// Per-layer caches pre-filled with POS tokens of random f16 k/v, so a
        /// decode iter at POS attends over a realistic context.
        pub(crate) fn build_caches(cfg: &LagunaConfig, dev: &Device) -> Vec<LayerCache> {
            (0..N_LAYER)
                .map(|il| {
                    let mut c = LayerCache::new(cfg, il, MAX_CTX, dev).unwrap();
                    let kv = |seed: u64| {
                        Tensor::from_vec(
                            seeded(N_KV * POS * HEAD_DIM, seed),
                            (N_KV, POS, HEAD_DIM),
                            dev,
                        )
                        .unwrap()
                        .to_dtype(DType::F16)
                        .unwrap()
                    };
                    c.append(&kv(9000 + il as u64), &kv(9500 + il as u64)).unwrap();
                    c
                })
                .collect()
        }

        /// Rewind every cache to POS stored tokens, so each timed iter decodes
        /// the same position with the same key count (no drift across iters).
        pub(crate) fn reset_caches(caches: &mut [LayerCache]) {
            for c in caches.iter_mut() {
                match c {
                    LayerCache::Full { len, .. } | LayerCache::Swa { len, .. } => *len = POS,
                }
            }
        }

        /// Which optional stages of the hand-written chain run. `proj_only`
        /// short-circuits everything except the five projection matmuls.
        #[derive(Clone, Copy)]
        struct Stages {
            proj_only: bool,
            rope: bool,
            gate_math: bool,
        }

        const FULL_CHAIN: Stages = Stages { proj_only: false, rope: true, gate_math: true };

        /// Hand-written mirror of one decode step: the same candle calls
        /// `AttnBlock::forward` makes at seq==1 (QMatMul::Tensor forwards are
        /// `x.matmul(&w.t())`), so its timing is interchangeable with the real
        /// block and stages can be ablated independently.
        fn hand_forward(
            l: &BenchLayer,
            x: &Tensor,
            cache: &mut LayerCache,
            pos: usize,
            st: Stages,
        ) -> Tensor {
            let mm = |x: &Tensor, w: &Tensor| x.matmul(&w.t().unwrap()).unwrap();
            if st.proj_only {
                let _g = mm(x, &l.wg);
                let q = mm(x, &l.wq);
                let _k = mm(x, &l.wk);
                let _v = mm(x, &l.wv);
                return mm(&q, &l.wo);
            }
            let h = l.n_head;
            let normed = l.attn_norm.forward(x).unwrap();
            let gate_logits = mm(&normed, &l.wg);
            let q = mm(&normed, &l.wq).reshape((1, h, HEAD_DIM)).unwrap();
            let k = mm(&normed, &l.wk).reshape((1, N_KV, HEAD_DIM)).unwrap();
            let v = mm(&normed, &l.wv).reshape((1, N_KV, HEAD_DIM)).unwrap();
            let q = l.q_norm.forward(&q).unwrap();
            let k = l.k_norm.forward(&k).unwrap();
            let q = q.reshape((h, 1, HEAD_DIM)).unwrap();
            let k = k.reshape((N_KV, 1, HEAD_DIM)).unwrap();
            let v = v.reshape((N_KV, 1, HEAD_DIM)).unwrap();
            let (q, k) = if st.rope { l.rope.apply(&q, &k, pos).unwrap() } else { (q, k) };
            let (k_all, v_all) = cache
                .append(&k.to_dtype(DType::F16).unwrap(), &v.to_dtype(DType::F16).unwrap())
                .unwrap();
            let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
            let qh = q.to_dtype(DType::F16).unwrap().unsqueeze(0).unwrap().contiguous().unwrap();
            let attn = candle_nn::ops::sdpa(
                &qh,
                &k_all.unsqueeze(0).unwrap(),
                &v_all.unsqueeze(0).unwrap(),
                None,
                false,
                scale,
                1.0,
            )
            .unwrap();
            let attn = attn.squeeze(0).unwrap().to_dtype(DType::F32).unwrap();
            let attn = if st.gate_math {
                let gate = softplus(&gate_logits)
                    .unwrap()
                    .transpose(0, 1)
                    .unwrap()
                    .reshape((h, 1, 1))
                    .unwrap();
                attn.broadcast_mul(&gate).unwrap()
            } else {
                attn
            };
            mm(&attn.reshape((1, h * HEAD_DIM)).unwrap(), &l.wo)
        }

        /// One token through all 48 layers, hand-written ops, mirroring the
        /// model.rs per-layer attention half: attn_norm → attn → residual add.
        fn hand_chain(layers: &[BenchLayer], caches: &mut [LayerCache], x0: &Tensor, st: Stages) -> Tensor {
            reset_caches(caches);
            let mut x = x0.clone();
            for (il, l) in layers.iter().enumerate() {
                let out = hand_forward(l, &x, &mut caches[il], POS, st);
                x = if st.proj_only { out } else { (&x + &out).unwrap() };
            }
            x
        }

        /// Same chain through the real `AttnBlock::forward`.
        fn real_chain(layers: &[BenchLayer], caches: &mut [LayerCache], x0: &Tensor) -> Tensor {
            reset_caches(caches);
            let mut x = x0.clone();
            for (il, l) in layers.iter().enumerate() {
                let normed = l.attn_norm.forward(&x).unwrap();
                let attn = l.block.as_ref().unwrap().forward(&normed, &mut caches[il], POS).unwrap();
                x = (&x + &attn).unwrap();
            }
            x
        }

        /// One production-shaped attention layer half at decode position POS:
        /// attn_norm → `AttnBlock::forward` → residual add, exactly as
        /// `real_chain` does per layer. Exposed for the full-stack decode bench
        /// (moe.rs tests::decode_bench), which interleaves it with MoE FFN
        /// halves. Requires layers built with `with_blocks = true`.
        pub(crate) fn attn_step(l: &BenchLayer, x: &Tensor, cache: &mut LayerCache) -> Tensor {
            let normed = l.attn_norm.forward(x).unwrap();
            let attn = l
                .block
                .as_ref()
                .expect("attn_step needs build_layers(dev, true)")
                .forward(&normed, cache, POS)
                .unwrap();
            (x + &attn).unwrap()
        }

        /// Small readback forcing command-buffer completion (the per-iter sync).
        fn read_scalar(t: &Tensor) -> f32 {
            let t = if t.dtype() == DType::F32 { t.clone() } else { t.to_dtype(DType::F32).unwrap() };
            t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]
        }

        /// Warm-up + timed loop; returns the mean ms/iter.
        fn bench(name: &str, mut f: impl FnMut() -> f32) -> f64 {
            let (warm, iters) = iter_counts();
            let mut sink = 0f32;
            for _ in 0..warm {
                sink += f();
            }
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                sink += f();
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
            eprintln!(
                "{name}: mean {mean:.3} ms/iter, min {min:.3} ms/iter ({iters} iters, sink {sink:.1})"
            );
            mean
        }

        /// Headline: attention-chain ms/token at production geometry, timed
        /// through the real `AttnBlock::forward`, with the hand-written mirror
        /// timed alongside (it must agree numerically and within ~5% on time
        /// for the ablation bench's stage attribution to be trustworthy).
        #[test]
        #[ignore = "perf bench"]
        fn attn_decode_chain_bench() {
            let dev = metal();
            let cfg = prod_cfg();
            eprintln!("building 48 synthetic attention layers (dense f32 + real AttnBlocks, ~22GB device memory)...");
            let layers = build_layers(&dev, true);
            let mut caches = build_caches(&cfg, &dev);
            let mut hand_caches = build_caches(&cfg, &dev);
            let x0 = dense(1, HIDDEN, 4242, &dev);

            // The two chains run the same candle ops on the same values, so
            // they must agree numerically before the hand timing means anything.
            let real = real_chain(&layers, &mut caches, &x0);
            let hand = hand_chain(&layers, &mut hand_caches, &x0, FULL_CHAIN);
            let cos = cosine(&real, &hand);
            eprintln!("hand-vs-real: cosine {cos}, max_abs_diff {}", max_abs_diff(&real, &hand));
            assert!(cos > 0.9999, "hand-written chain diverges from AttnBlock::forward: cos {cos}");

            let real_ms = bench("attn chain x48 (real AttnBlock::forward)", || {
                read_scalar(&real_chain(&layers, &mut caches, &x0))
            });
            let hand_ms = bench("attn chain x48 (hand-written mirror)", || {
                read_scalar(&hand_chain(&layers, &mut hand_caches, &x0, FULL_CHAIN))
            });
            eprintln!(
                "hand/real time ratio {:.3} (should be within ~5% of 1.0)",
                hand_ms / real_ms
            );
        }

        /// Stage ablations over the hand-written 48-layer chain. Prints each
        /// variant plus derived per-stage costs.
        #[test]
        #[ignore = "perf bench"]
        fn attn_decode_ablation_bench() {
            let dev = metal();
            let cfg = prod_cfg();
            eprintln!("building 48 synthetic attention layers (dense f32, ~11GB device memory)...");
            let layers = build_layers(&dev, false);
            let mut caches = build_caches(&cfg, &dev);
            let x0 = dense(1, HIDDEN, 4242, &dev);

            let full = bench("full chain", || {
                read_scalar(&hand_chain(&layers, &mut caches, &x0, FULL_CHAIN))
            });
            let no_gate = bench("minus softplus-gate math (g_proj kept)", || {
                read_scalar(&hand_chain(
                    &layers,
                    &mut caches,
                    &x0,
                    Stages { gate_math: false, ..FULL_CHAIN },
                ))
            });
            let no_rope = bench("minus rope", || {
                read_scalar(&hand_chain(&layers, &mut caches, &x0, Stages { rope: false, ..FULL_CHAIN }))
            });
            let proj = bench("projections-only (5 matmuls/layer)", || {
                read_scalar(&hand_chain(
                    &layers,
                    &mut caches,
                    &x0,
                    Stages { proj_only: true, ..FULL_CHAIN },
                ))
            });

            // sdpa-only: q and the cache k/v views are produced once, outside
            // the timed loop; an iter is just the 48 sdpa kernels + readback.
            reset_caches(&mut caches);
            let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
            let sdpa_in: Vec<(Tensor, Tensor, Tensor)> = (0..N_LAYER)
                .map(|il| {
                    let h = n_head_of(il);
                    let kv1 = |seed: u64| {
                        Tensor::from_vec(seeded(N_KV * HEAD_DIM, seed), (N_KV, 1, HEAD_DIM), &dev)
                            .unwrap()
                            .to_dtype(DType::F16)
                            .unwrap()
                    };
                    let (k_all, v_all) =
                        caches[il].append(&kv1(7000 + il as u64), &kv1(7500 + il as u64)).unwrap();
                    let q = Tensor::from_vec(
                        seeded(h * HEAD_DIM, 8000 + il as u64),
                        (1, h, 1, HEAD_DIM),
                        &dev,
                    )
                    .unwrap()
                    .to_dtype(DType::F16)
                    .unwrap();
                    (q, k_all.unsqueeze(0).unwrap(), v_all.unsqueeze(0).unwrap())
                })
                .collect();
            let sdpa = bench("sdpa-only (48 sdpa kernels)", || {
                let mut last = None;
                for (q, k, v) in &sdpa_in {
                    last = Some(candle_nn::ops::sdpa(q, k, v, None, false, scale, 1.0).unwrap());
                }
                read_scalar(&last.unwrap())
            });

            eprintln!("derived per-token costs:");
            eprintln!("  softplus-gate math: {:.3} ms", full - no_gate);
            eprintln!("  rope:               {:.3} ms", full - no_rope);
            eprintln!("  projections:        {:.3} ms", proj);
            eprintln!("  sdpa kernels:       {:.3} ms", sdpa);
            eprintln!("  everything else:    {:.3} ms", full - proj - sdpa);
        }

        /// One layer of the ALL-f16 activation variant: f16 projection weights
        /// and f16 activations end-to-end inside the block. Only `attn_norm`
        /// stays f32 (it reads the f32 residual stream); QK-norm weights and
        /// rope tables are f16 so the fused rmsnorm/rope kernels run natively
        /// in f16 (the pinned candle rev requires weight/cos/sin dtype == x
        /// dtype). Perf probe ONLY — this structure was measured and rejected
        /// for numerics (it drifts from the f32 oracle far more than the fork
        /// does); the shipped block keeps activations f32 and casts only the
        /// matmul operands.
        struct F16Layer {
            attn_norm: RmsNorm,
            wq: Tensor,
            wk: Tensor,
            wv: Tensor,
            wg: Tensor,
            wo: Tensor,
            q_norm: RmsNorm,
            k_norm: RmsNorm,
            cos: Tensor,
            sin: Tensor,
            n_rot: usize,
            n_head: usize,
        }

        /// Plain-rope cos/sin tables in f16. A perf probe only needs the right
        /// shapes and dtype, so the full layers' YaRN correction is omitted
        /// (same table lookup cost).
        fn rope_tables_f16(freq_base: f64, n_rot: usize, dev: &Device) -> (Tensor, Tensor) {
            let half = n_rot / 2;
            let mut cos = vec![0f32; MAX_CTX * half];
            let mut sin = vec![0f32; MAX_CTX * half];
            for p in 0..MAX_CTX {
                for j in 0..half {
                    let theta = p as f64 * freq_base.powf(-(2.0 * j as f64) / n_rot as f64);
                    cos[p * half + j] = theta.cos() as f32;
                    sin[p * half + j] = theta.sin() as f32;
                }
            }
            let up = |v: Vec<f32>| {
                Tensor::from_vec(v, (MAX_CTX, half), dev).unwrap().to_dtype(DType::F16).unwrap()
            };
            (up(cos), up(sin))
        }

        fn build_f16_layers(dev: &Device) -> Vec<F16Layer> {
            let full_tables = rope_tables_f16(500_000.0, 64, dev);
            let swa_tables = rope_tables_f16(10_000.0, HEAD_DIM, dev);
            (0..N_LAYER)
                .map(|il| {
                    let h = n_head_of(il);
                    let s = il as u64 * 100;
                    let w16 = |rows: usize, cols: usize, seed: u64| {
                        dense(rows, cols, seed, dev).to_dtype(DType::F16).unwrap()
                    };
                    let n16 = |seed: u64| norm_w(HEAD_DIM, seed, dev).to_dtype(DType::F16).unwrap();
                    let (n_rot, (cos, sin)) =
                        if il % 4 == 0 { (64, full_tables.clone()) } else { (HEAD_DIM, swa_tables.clone()) };
                    F16Layer {
                        attn_norm: RmsNorm::new(norm_w(HIDDEN, s + 8, dev), 1e-6),
                        wq: w16(h * HEAD_DIM, HIDDEN, s + 1),
                        wk: w16(N_KV * HEAD_DIM, HIDDEN, s + 2),
                        wv: w16(N_KV * HEAD_DIM, HIDDEN, s + 3),
                        wg: w16(h, HIDDEN, s + 4),
                        wo: w16(HIDDEN, h * HEAD_DIM, s + 5),
                        q_norm: RmsNorm::new(n16(s + 6), 1e-6),
                        k_norm: RmsNorm::new(n16(s + 7), 1e-6),
                        cos,
                        sin,
                        n_rot,
                        n_head: h,
                    }
                })
                .collect()
        }

        /// `Rope::rotate` in f16: narrow the rotated block (partial rotary on
        /// full layers), fused rope kernel, cat the pass-through dims back.
        fn rope_f16(x: &Tensor, cos: &Tensor, sin: &Tensor, n_rot: usize, pos: usize) -> Tensor {
            let (_, seq, head_dim) = x.dims3().unwrap();
            let cos = cos.narrow(0, pos, seq).unwrap();
            let sin = sin.narrow(0, pos, seq).unwrap();
            let x = x.unsqueeze(0).unwrap();
            let rotated = candle_nn::rotary_emb::rope(
                &x.narrow(3, 0, n_rot).unwrap().contiguous().unwrap(),
                &cos,
                &sin,
            )
            .unwrap();
            let out = if n_rot < head_dim {
                let pass = x.narrow(3, n_rot, head_dim - n_rot).unwrap();
                Tensor::cat(&[&rotated, &pass], 3).unwrap()
            } else {
                rotated
            };
            out.squeeze(0).unwrap().contiguous().unwrap()
        }

        /// One decode step of the all-f16 variant: activations stay f16 from
        /// the projection inputs through QK-norm, rope, cache append, sdpa,
        /// the softplus gate, and o_proj. Exactly two dtype casts per layer —
        /// f32→f16 once after the f32 attn_norm and f16→f32 once after o_proj
        /// for the residual.
        fn f16_forward(l: &F16Layer, x: &Tensor, cache: &mut LayerCache, pos: usize, proj_only: bool) -> Tensor {
            let mm = |x: &Tensor, w: &Tensor| x.matmul(&w.t().unwrap()).unwrap();
            if proj_only {
                // The end-state's cast structure with everything between the
                // projections removed: cast in, five matmuls, cast out.
                let xh = x.to_dtype(DType::F16).unwrap();
                let _g = mm(&xh, &l.wg);
                let q = mm(&xh, &l.wq);
                let _k = mm(&xh, &l.wk);
                let _v = mm(&xh, &l.wv);
                return mm(&q, &l.wo).to_dtype(DType::F32).unwrap();
            }
            let h = l.n_head;
            let normed = l.attn_norm.forward(x).unwrap();
            let xh = normed.to_dtype(DType::F16).unwrap();
            let gate_logits = mm(&xh, &l.wg);
            let q = mm(&xh, &l.wq).reshape((1, h, HEAD_DIM)).unwrap();
            let k = mm(&xh, &l.wk).reshape((1, N_KV, HEAD_DIM)).unwrap();
            let v = mm(&xh, &l.wv).reshape((1, N_KV, HEAD_DIM)).unwrap();
            let q = l.q_norm.forward(&q).unwrap();
            let k = l.k_norm.forward(&k).unwrap();
            let q = q.reshape((h, 1, HEAD_DIM)).unwrap();
            let k = k.reshape((N_KV, 1, HEAD_DIM)).unwrap();
            let v = v.reshape((N_KV, 1, HEAD_DIM)).unwrap();
            let q = rope_f16(&q, &l.cos, &l.sin, l.n_rot, pos);
            let k = rope_f16(&k, &l.cos, &l.sin, l.n_rot, pos);
            let (k_all, v_all) = cache.append(&k, &v).unwrap();
            let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
            let qh = q.unsqueeze(0).unwrap().contiguous().unwrap();
            let attn = candle_nn::ops::sdpa(
                &qh,
                &k_all.unsqueeze(0).unwrap(),
                &v_all.unsqueeze(0).unwrap(),
                None,
                false,
                scale,
                1.0,
            )
            .unwrap();
            let attn = attn.squeeze(0).unwrap();
            let gate = softplus(&gate_logits)
                .unwrap()
                .transpose(0, 1)
                .unwrap()
                .reshape((h, 1, 1))
                .unwrap();
            let attn = attn.broadcast_mul(&gate).unwrap();
            mm(&attn.reshape((1, h * HEAD_DIM)).unwrap(), &l.wo).to_dtype(DType::F32).unwrap()
        }

        fn f16_chain(layers: &[F16Layer], caches: &mut [LayerCache], x0: &Tensor, proj_only: bool) -> Tensor {
            reset_caches(caches);
            let mut x = x0.clone();
            for (il, l) in layers.iter().enumerate() {
                let out = f16_forward(l, &x, &mut caches[il], POS, proj_only);
                x = if proj_only { out } else { (&x + &out).unwrap() };
            }
            x
        }

        /// The all-f16 activation chain: f16 projection weights with
        /// activations kept f16 through the whole attention block. Prices that
        /// variant per token and shows the glue share once projections are
        /// halved. Perf shape probe only — the shipped block is the hybrid
        /// (f16 weights, f32 activations; see `attn_proj_f16_bench` for its
        /// projection structure) and makes no parity claim here.
        #[test]
        #[ignore = "perf bench"]
        fn attn_decode_f16_chain_bench() {
            let dev = metal();
            let cfg = prod_cfg();
            eprintln!("building 48 synthetic f16 attention layers (~5.6GB device memory)...");
            let layers = build_f16_layers(&dev);
            let mut caches = build_caches(&cfg, &dev);
            let x0 = dense(1, HIDDEN, 4242, &dev);

            let full = bench("all-f16 chain x48", || {
                read_scalar(&f16_chain(&layers, &mut caches, &x0, false))
            });
            let proj = bench("f16 projections-only (entry/exit casts kept)", || {
                read_scalar(&f16_chain(&layers, &mut caches, &x0, true))
            });
            eprintln!(
                "remaining glue in the f16 world: {:.3} ms/token ({:.0}% of the f16 chain)",
                full - proj,
                (full - proj) / full * 100.0
            );
        }

        /// The f16-weight lever: the five projection matmuls per layer x 48
        /// layers, three ways — f32 weights (candle), f16 weights through
        /// candle's same-dtype matmul (x cast to f16 per layer, output cast
        /// back), and f16 weights through the shipped vendored mixed-dtype
        /// kernels (no casts). Reports ms/iter and implied weight-traffic GB/s.
        #[test]
        #[ignore = "perf bench"]
        fn attn_proj_f16_bench() {
            let dev = metal();
            struct Proj {
                wq: Tensor,
                wk: Tensor,
                wv: Tensor,
                wg: Tensor,
                wo: Tensor,
            }
            eprintln!("building 48 layers of projection weights (f32 + f16 copies, ~17GB device memory)...");
            let f32_layers: Vec<Proj> = (0..N_LAYER)
                .map(|il| {
                    let h = n_head_of(il);
                    let s = il as u64 * 100;
                    Proj {
                        wq: dense(h * HEAD_DIM, HIDDEN, s + 1, &dev),
                        wk: dense(N_KV * HEAD_DIM, HIDDEN, s + 2, &dev),
                        wv: dense(N_KV * HEAD_DIM, HIDDEN, s + 3, &dev),
                        wg: dense(h, HIDDEN, s + 4, &dev),
                        wo: dense(HIDDEN, h * HEAD_DIM, s + 5, &dev),
                    }
                })
                .collect();
            let f16_layers: Vec<Proj> = f32_layers
                .iter()
                .map(|p| {
                    let c = |t: &Tensor| t.to_dtype(DType::F16).unwrap();
                    Proj { wq: c(&p.wq), wk: c(&p.wk), wv: c(&p.wv), wg: c(&p.wg), wo: c(&p.wo) }
                })
                .collect();
            let elems: usize = f32_layers
                .iter()
                .map(|p| {
                    [&p.wq, &p.wk, &p.wv, &p.wg, &p.wo].iter().map(|t| t.elem_count()).sum::<usize>()
                })
                .sum();
            let x0 = dense(1, HIDDEN, 777, &dev);
            let mm = |x: &Tensor, w: &Tensor| x.matmul(&w.t().unwrap()).unwrap();

            let f32_ms = bench("proj x48, f32 weights / f32 x", || {
                let mut x = x0.clone();
                for p in &f32_layers {
                    let _g = mm(&x, &p.wg);
                    let q = mm(&x, &p.wq);
                    let _k = mm(&x, &p.wk);
                    let _v = mm(&x, &p.wv);
                    x = mm(&q, &p.wo);
                }
                read_scalar(&x)
            });
            let f16_ms = bench("proj x48, f16 weights / f16 x (cast in+out per layer)", || {
                let mut x = x0.clone();
                for p in &f16_layers {
                    let xh = x.to_dtype(DType::F16).unwrap();
                    let _g = mm(&xh, &p.wg);
                    let q = mm(&xh, &p.wq);
                    let _k = mm(&xh, &p.wk);
                    let _v = mm(&xh, &p.wv);
                    x = mm(&q, &p.wo).to_dtype(DType::F32).unwrap();
                }
                read_scalar(&x)
            });
            let vendored_ms = bench("proj x48, f16 weights / vendored mixed-dtype (no casts)", || {
                let mut x = x0.clone();
                for p in &f16_layers {
                    let mv = |w: &Tensor, x: &Tensor| crate::ops::matmul_f16(w, x).unwrap();
                    let _g = mv(&p.wg, &x);
                    let q = mv(&p.wq, &x);
                    let _k = mv(&p.wk, &x);
                    let _v = mv(&p.wv, &x);
                    x = mv(&p.wo, &q);
                }
                read_scalar(&x)
            });
            let gbs = |bytes: usize, ms: f64| bytes as f64 / 1e9 / (ms / 1e3);
            eprintln!(
                "weight traffic: f32 {:.2} GB @ {:.1} GB/s | f16 cast {:.2} GB @ {:.1} GB/s | vendored @ {:.1} GB/s | f32/f16 {:.2}x, cast/vendored {:.2}x",
                elems as f64 * 4.0 / 1e9,
                gbs(elems * 4, f32_ms),
                elems as f64 * 2.0 / 1e9,
                gbs(elems * 2, f16_ms),
                gbs(elems * 2, vendored_ms),
                f32_ms / f16_ms,
                f16_ms / vendored_ms
            );
        }

        /// Per-dispatch overhead: 512 chained tiny elementwise ops vs 8 ops of
        /// equal total arithmetic. The time difference divided by the dispatch
        /// count difference approximates the cost of one dependent dispatch.
        #[test]
        #[ignore = "perf bench"]
        fn dispatch_overhead_bench() {
            let dev = metal();
            let small = dense(1, HIDDEN, 1, &dev);
            let big = dense(1, HIDDEN * 64, 2, &dev);
            // Both variants read back the same 3072 elements.
            let read_head = |t: &Tensor| read_scalar(&t.narrow(1, 0, HIDDEN).unwrap());

            let many = bench("512 chained affines on [1,3072]", || {
                let mut x = small.clone();
                for _ in 0..512 {
                    x = x.affine(1.0000001, 1e-9).unwrap();
                }
                read_head(&x)
            });
            let few = bench("8 chained affines on [1,196608] (equal arithmetic)", || {
                let mut x = big.clone();
                for _ in 0..8 {
                    x = x.affine(1.0000001, 1e-9).unwrap();
                }
                read_head(&x)
            });
            eprintln!(
                "approx per-dispatch overhead: {:.2} us ({} extra dispatches cost {:.3} ms)",
                (many - few) / 504.0 * 1e3,
                504,
                many - few
            );
        }

        // --- prefill-isolation attention bench (seq=512) -------------------

        /// Prefill chunk length (matches the MoE prefill benches in moe.rs so the
        /// two halves' numbers are directly comparable).
        const PREFILL_SEQ: usize = 512;

        /// One SHIPPED f16 attention block at `il`'s geometry: f16 projection
        /// weights (`Proj::DenseF16`, the production prefill default) consumed by
        /// the vendored mixed-dtype kernels, with the same QK-norm/rope/head
        /// counts the real layer uses. Built directly so the bench runs the
        /// production f16 path rather than the dequant-f32 `QMatMul` one the
        /// decode benches build.
        fn build_f16_block(dev: &Device, il: usize, rope: Arc<Rope>) -> AttnBlock {
            let h = n_head_of(il);
            let s = il as u64 * 100;
            let f16w = |rows: usize, cols: usize, seed: u64| {
                dense(rows, cols, seed, dev).to_dtype(DType::F16).unwrap()
            };
            AttnBlock {
                q_proj: Proj::DenseF16(f16w(h * HEAD_DIM, HIDDEN, s + 1)),
                k_proj: Proj::DenseF16(f16w(N_KV * HEAD_DIM, HIDDEN, s + 2)),
                v_proj: Proj::DenseF16(f16w(N_KV * HEAD_DIM, HIDDEN, s + 3)),
                g_proj: Proj::DenseF16(f16w(h, HIDDEN, s + 4)),
                o_proj: Proj::DenseF16(f16w(HIDDEN, h * HEAD_DIM, s + 5)),
                q_norm: RmsNorm::new(norm_w(HEAD_DIM, s + 6, dev), 1e-6),
                k_norm: RmsNorm::new(norm_w(HEAD_DIM, s + 7, dev), 1e-6),
                rope,
                n_head: h,
                n_kv_head: N_KV,
                head_dim: HEAD_DIM,
            }
        }

        /// Prices the full attention chain (projections + QK-norm + rope + sdpa
        /// + softplus gate + o_proj) at a 512-token prefill chunk, pos 0, timed
        /// separately for the two per-layer variants the model interleaves: a
        /// FULL-attention layer (48 Q heads, YaRN partial rotary 64/128 dims)
        /// and an SWA layer (72 Q heads, plain rope over all 128 dims, window
        /// 512). Each iter runs one block forward over a fresh pos-0 cache and
        /// ends in a small readback. `#[ignore]`d; run e.g.
        /// `cargo test --release prefill_attn_chain_bench -- --ignored --nocapture`.
        #[test]
        #[ignore = "perf bench"]
        fn prefill_attn_chain_bench() {
            let dev = metal();
            let cfg = prod_cfg();
            let (rope_full, rope_swa) = build_ropes(&dev);
            eprintln!(
                "building 2 synthetic f16 attention blocks (full-attn + SWA) at seq={PREFILL_SEQ}..."
            );
            // il 0 is full-attention (il % 4 == 0), il 1 is SWA.
            let full_block = build_f16_block(&dev, 0, rope_full);
            let swa_block = build_f16_block(&dev, 1, rope_swa);
            let mut full_cache = LayerCache::new(&cfg, 0, MAX_CTX, &dev).unwrap();
            let mut swa_cache = LayerCache::new(&cfg, 1, MAX_CTX, &dev).unwrap();
            let x = dense(PREFILL_SEQ, HIDDEN, 4242, &dev);

            bench(
                &format!("prefill attn FULL block (48 heads, YaRN partial rope), seq={PREFILL_SEQ}"),
                || {
                    full_cache.reset();
                    read_scalar(&full_block.forward(&x, &mut full_cache, 0).unwrap())
                },
            );
            bench(
                &format!("prefill attn SWA block (72 heads, plain rope, window 512), seq={PREFILL_SEQ}"),
                || {
                    swa_cache.reset();
                    read_scalar(&swa_block.forward(&x, &mut swa_cache, 0).unwrap())
                },
            );
        }
    }

    /// Test 5: both per-layer query-head widths produce correct end-to-end shapes.
    #[test]
    fn per_layer_head_widths() {
        let dev = Device::Cpu;
        let (n_kv, hd, hidden) = (8, 16, 64);
        for &n_head in &[48usize, 72usize] {
            let cfg = test_cfg(n_head, n_kv, hd, hidden, 4, 512);
            let rope = Arc::new(Rope::new(cfg.rope(0), 32, &dev).unwrap());
            let w = raw_weights(n_head, n_kv, hd, hidden, &dev);
            let block = build_block(&w, &cfg, 0, rope, &dev, AttnWeights::DequantF32);
            let mut cache = LayerCache::new(&cfg, 0, 32, &dev).unwrap();
            let x = dense(4, hidden, 11, &dev);
            let out = block.forward(&x, &mut cache, 0).unwrap();
            assert_eq!(out.dims(), &[4, hidden]);
            assert!(out.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|v| v.is_finite()));
        }
    }
}
