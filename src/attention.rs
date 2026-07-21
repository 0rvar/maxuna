use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};

use crate::config::LagunaConfig;
use crate::gguf::{QLinear, Weights};
use crate::kv_cache::LayerCache;
use crate::rope::Rope;

/// One Laguna attention block. Order (fork laguna.cpp:200-265):
/// gate logits come from the pre-attention normed input; q/k get per-head-dim
/// RMSNorm before rope; sdpa at scale 1/sqrt(head_dim) in f16; the softplus
/// gate multiplies attention output per-head (broadcast over head_dim) BEFORE
/// o_proj. Q-head count is per-layer (48 full / 72 SWA); KV heads always 8.
pub struct AttnBlock {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    g_proj: QLinear,
    o_proj: QLinear,
    q_norm: candle_nn::RmsNorm,
    k_norm: candle_nn::RmsNorm,
    rope: Arc<Rope>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
}

impl AttnBlock {
    /// `w` is positioned at the block prefix (e.g. `blk.7`).
    pub fn new(w: &Weights, cfg: &LagunaConfig, il: usize, rope: Arc<Rope>) -> Result<Self> {
        Ok(Self {
            q_proj: w.qlinear("attn_q")?,
            k_proj: w.qlinear("attn_k")?,
            v_proj: w.qlinear("attn_v")?,
            g_proj: w.qlinear("attn_gate")?,
            o_proj: w.qlinear("attn_output")?,
            q_norm: w.rms_norm("attn_q_norm", cfg.rms_eps)?,
            k_norm: w.rms_norm("attn_k_norm", cfg.rms_eps)?,
            rope,
            n_head: cfg.n_head(il),
            n_kv_head: cfg.n_kv_head,
            head_dim: cfg.head_dim,
        })
    }

    /// x_normed: [seq, hidden] f32 (already attn_norm'ed by the caller).
    pub fn forward(&self, x_normed: &Tensor, cache: &mut LayerCache, pos: usize) -> Result<Tensor> {
        let (seq, _hidden) = x_normed.dims2()?;

        // Gate logits from the *pre-attention* normed input (not the attn output).
        let gate_logits = self.g_proj.forward(x_normed)?; // [seq, n_head]

        let q = self.q_proj.forward(x_normed)?.reshape((seq, self.n_head, self.head_dim))?;
        let k = self.k_proj.forward(x_normed)?.reshape((seq, self.n_kv_head, self.head_dim))?;
        let v = self.v_proj.forward(x_normed)?.reshape((seq, self.n_kv_head, self.head_dim))?;

        // QK-norm: RMSNorm over head_dim (f32) before rope, in [seq, head, dim]
        // layout where head_dim is contiguous last.
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // To [head, seq, head_dim] for rope + attention.
        let q = q.transpose(0, 1)?.contiguous()?;
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;

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

        // Softplus output gate, per-head, broadcast over head_dim.
        let gate = softplus(&gate_logits)?.transpose(0, 1)?.reshape((self.n_head, seq, 1))?;
        let attn = attn.broadcast_mul(&gate)?;

        // Back to [seq, n_head*head_dim] then o_proj.
        let out = attn.transpose(0, 1)?.contiguous()?.reshape((seq, self.n_head * self.head_dim))?;
        Ok(self.o_proj.forward(&out)?)
    }

    /// Metal MLX fused attention. q [n_head, seq, hd] f32, k/v [n_kv_head, K, hd]
    /// f16. GQA (n_head multiple of n_kv_head) is handled by the kernel; k/v are
    /// not pre-tiled. Returns [n_head, seq, hd] f32.
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

    /// Explicit softmax(q·kᵀ·scale + mask)·v in f32, GQA via a broadcast over the
    /// query group dim. q [n_head, seq, hd] f32, k/v [n_kv_head, K, hd] f16.
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
        let k = k_all.to_dtype(DType::F32)?;
        let v = v_all.to_dtype(DType::F32)?;

        let q4 = q.reshape((self.n_kv_head, g, seq, self.head_dim))?;
        let k4 = k.reshape((self.n_kv_head, 1, k_seq, self.head_dim))?;
        let v4 = v.reshape((self.n_kv_head, 1, k_seq, self.head_dim))?;

        let scores = q4.broadcast_matmul(&k4.transpose(2, 3)?)?.affine(scale as f64, 0.0)?;
        let scores = match mask {
            Some(m) => scores.broadcast_add(&m.reshape((1, 1, seq, k_seq))?)?,
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
mod tests {
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
    fn build_block(w: &RawWeights, cfg: &LagunaConfig, il: usize, rope: Arc<Rope>, dev: &Device) -> AttnBlock {
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
        let weights = Weights::from_gguf(src).pp("blk.0");
        let block = AttnBlock::new(&weights, cfg, il, rope).unwrap();
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
        let block = build_block(&w, &cfg, 0, rope.clone(), &dev);
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
        let block = build_block(&w, &cfg, 1, rope.clone(), &dev); // il 1 => SWA
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
        let block = build_block(&w, &cfg, 0, rope.clone(), &dev);
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
        let block_swa = build_block(&w, &cfg_swa, 1, rope_swa.clone(), &dev);
        let mut cache_swa = LayerCache::new(&cfg_swa, 1, 128, &dev).unwrap();
        let xs = dense(6, hidden, 8, &dev);
        let ref_swa = naive_forward(&w, &rope_swa, &xs, n_head, n_kv, hd, Some(win));
        let swa = block_swa.forward(&xs, &mut cache_swa, 0).unwrap();
        assert!(cosine(&swa, &ref_swa) > 0.999, "swa cos {}", cosine(&swa, &ref_swa));
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
            let block = build_block(&w, &cfg, 0, rope, &dev);
            let mut cache = LayerCache::new(&cfg, 0, 32, &dev).unwrap();
            let x = dense(4, hidden, 11, &dev);
            let out = block.forward(&x, &mut cache, 0).unwrap();
            assert_eq!(out.dims(), &[4, hidden]);
            assert!(out.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|v| v.is_finite()));
        }
    }
}
