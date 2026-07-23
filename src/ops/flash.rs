//! Host side of the vendored flash-attention prefill kernel (flash.metal —
//! the modified copy of candle's MLX steel attention). The kernel computes the
//! causal / sliding-window mask in-kernel from (pos, k_off, window), so the
//! prefill path needs no materialized mask tensor (1.5-2.3 GB at 4k context).
//! Numerics: the block structure and accumulation order are candle's float32
//! steel kernel's exactly, and the tests below measure the output against the
//! composed f32 sdpa reference (widened k/v + additive mask — the
//! `LAGUNA_SDPA_F32`-equivalent chain).

use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

/// Fused flash-attention prefill: `softmax(q·kᵀ·scale + mask)·v` with the mask
/// computed in-kernel. `q` is `[n_head, seq, 128]` f32 contiguous (the rope
/// output); `k`/`v` are `[n_kv, K, 128]` f16 cache views (head-strided views
/// are consumed via their strides — never forced contiguous). Query row i has
/// absolute position `pos + i`; key column j has absolute position
/// `k_off + j`; row i sees column j iff `k_off + j <= pos + i` and
/// `(pos + i) - (k_off + j) < window` (`None` = full attention) — exactly
/// `kv_cache::attn_mask_for`'s rule, so `k_off` is 0 for full-attention
/// layers and `pos - min(pos, window)` for SWA prefill (the oldest→newest
/// cache view). Returns `[n_head, seq, 128]` f32 contiguous. Metal only; the
/// caller's kill-switch is the sdpa path (`LAGUNA_FLASH_CLASSIC`).
pub fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    pos: usize,
    k_off: usize,
    window: Option<usize>,
    scale: f32,
) -> Result<Tensor> {
    dispatch::run_flash_attn(q, k, v, pos, k_off, window, scale, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::kv_cache::{MaskKind, attn_mask_for};
    use candle_core::{DType, Device, Tensor};

    /// Deterministic pseudo-random f32s in [lo, hi] (xorshift, no deps).
    fn rand(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = (s >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
                lo + (hi - lo) * u as f32
            })
            .collect()
    }

    fn rand_t(seed: u64, dims: (usize, usize, usize), dev: &Device) -> Tensor {
        Tensor::from_vec(rand(seed, dims.0 * dims.1 * dims.2, -2.0, 2.0), dims, dev).unwrap()
    }

    /// f16 k/v inputs: values drawn f32 then rounded through f16 (the cache's
    /// dtype), so both the kernel and the widened reference see identical values.
    fn rand_kv(seed: u64, dims: (usize, usize, usize), dev: &Device) -> Tensor {
        rand_t(seed, dims, dev).to_dtype(DType::F16).unwrap()
    }

    /// The composed f32 reference: widen k/v to f32, materialize the additive
    /// mask exactly as `attn_mask_for` builds it (broadcast to
    /// `[1, n_head, seq, K]` f32, matching q's dtype as candle requires), and
    /// run candle's f32 sdpa — the `LAGUNA_SDPA_F32`-equivalent chain, i.e.
    /// candle's float32 steel kernel that flash.metal derives from.
    fn reference_sdpa(
        q: &Tensor,
        k16: &Tensor,
        v16: &Tensor,
        kind: MaskKind,
        pos: usize,
        scale: f32,
    ) -> Tensor {
        let (n_head, seq, _) = q.dims3().unwrap();
        let k_len = k16.dim(1).unwrap();
        let k = k16.to_dtype(DType::F32).unwrap();
        let v = v16.to_dtype(DType::F32).unwrap();
        let mask = attn_mask_for(kind, seq, pos, q.device())
            .unwrap()
            .map(|raw| {
                assert_eq!(raw.dims(), &[seq, k_len], "fixture mask/K mismatch");
                raw.reshape((1, 1, seq, k_len))
                    .unwrap()
                    .broadcast_as((1, n_head, seq, k_len))
                    .unwrap()
                    .contiguous()
                    .unwrap()
            });
        let out = candle_nn::ops::sdpa(
            &q.unsqueeze(0).unwrap(),
            &k.unsqueeze(0).unwrap(),
            &v.unsqueeze(0).unwrap(),
            mask.as_ref(),
            false,
            scale,
            1.0,
        )
        .unwrap();
        out.squeeze(0).unwrap()
    }

    /// Compare flash output against a reference: report bitwise equality when
    /// it holds, otherwise report the measured max abs/rel diff and assert
    /// rel <= 1e-6 (the accumulation order is designed to be identical, so
    /// bitwise is the expected outcome; the printout is the diagnostic).
    fn check_close(got: &Tensor, want: &Tensor, label: &str) {
        assert_eq!(got.dims(), want.dims(), "{label}: shape");
        assert_eq!(got.dtype(), DType::F32, "{label}: dtype");
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        let bitwise = g.iter().zip(w.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
        if bitwise {
            println!("{label}: BITWISE-IDENTICAL to the f32 sdpa reference");
            return;
        }
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        for (a, b) in g.iter().zip(w.iter()) {
            assert!(a.is_finite(), "{label}: non-finite flash output {a}");
            let abs = (a - b).abs();
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(abs / b.abs().max(1e-20));
        }
        println!("{label}: not bitwise; max abs diff {max_abs:.3e}, max rel diff {max_rel:.3e}");
        assert!(max_rel <= 1e-6, "{label}: max rel diff {max_rel:.3e} exceeds 1e-6");
    }

    fn assert_bits_eq(got: &Tensor, want: &Tensor, label: &str) {
        assert_eq!(got.dims(), want.dims(), "{label}: shape");
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (a, b)) in g.iter().zip(w.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}: element {i} differs ({a:?} vs {b:?})"
            );
        }
    }

    /// UNIT 1: flash_attn matches the composed f32 sdpa reference across the
    /// production mask geometries — full/SWA, aligned/unaligned seq and K,
    /// pos 0 and mid-stream, ring-style k_off, both real head-count classes
    /// (48/8 and 72/8) plus scaled-down heads at the same gqa ratios (6, 9).
    #[test]
    fn flash_attn_matches_f32_sdpa() {
        let dev = metal_device().unwrap();
        let hd = 128usize;
        let scale = 1.0f32 / (hd as f32).sqrt();

        // (label, n_head, n_kv, seq, pos, window). K and k_off derive from the
        // mask kind exactly as the production caller derives them.
        type Case = (&'static str, usize, usize, usize, usize, Option<usize>);
        let cases: &[Case] = &[
            // Full attention, pos 0, aligned seq (64 % 32 == 0) and K, real 48/8.
            ("full pos=0 aligned 48h", 48, 8, 64, 0, None),
            // Full attention mid-stream: seq 33 (unaligned Q), K = 65 (unaligned K).
            ("full pos>0 unaligned", 12, 2, 33, 32, None),
            // SWA shorter than the window (pure causal in effect), unaligned.
            ("swa short seq<window", 18, 2, 40, 0, Some(64)),
            // SWA crossing the window inside one prefill, real 72/8: whole KV
            // blocks expire for later query blocks (block-skip in anger).
            ("swa crossing window 72h", 72, 8, 40, 0, Some(16)),
            // SWA mid-stream with a ring-style k_off = pos - min(pos, window).
            ("swa pos>0 ring k_off", 18, 2, 20, 24, Some(16)),
            // Minimal prefill.
            ("full seq=2", 6, 1, 2, 5, None),
        ];

        for &(label, n_head, n_kv, seq, pos, window) in cases {
            let (kind, k_off, k_len) = match window {
                None => (MaskKind::Full, 0usize, pos + seq),
                Some(w) => {
                    let m = pos.min(w);
                    (MaskKind::Swa { window: w }, pos - m, m + seq)
                }
            };
            let seed = (n_head * 1000 + seq * 10 + pos) as u64;
            let q = rand_t(seed, (n_head, seq, hd), &dev);
            let k16 = rand_kv(seed + 1, (n_kv, k_len, hd), &dev);
            let v16 = rand_kv(seed + 2, (n_kv, k_len, hd), &dev);

            let got = flash_attn(&q, &k16, &v16, pos, k_off, window, scale).unwrap();
            let want = reference_sdpa(&q, &k16, &v16, kind, pos, scale);
            check_close(&got, &want, label);
        }
    }

    /// UNIT 2: head-strided cache views — k/v as narrowed views of a larger
    /// `[n_kv, max_slots, hd]` buffer (nonzero start offset + a head-axis gap)
    /// must match the packed-copy result bitwise (the strides, not a forced
    /// contiguous copy, feed the kernel).
    #[test]
    fn flash_attn_head_strided_views() {
        let dev = metal_device().unwrap();
        let (n_head, n_kv, hd) = (12usize, 2usize, 128usize);
        let (seq, pos) = (24usize, 8usize);
        let k_len = pos + seq;
        let (max_slots, off) = (100usize, 5usize);
        let scale = 1.0f32 / (hd as f32).sqrt();

        let q = rand_t(90, (n_head, seq, hd), &dev);
        let k_big = rand_kv(91, (n_kv, max_slots, hd), &dev);
        let v_big = rand_kv(92, (n_kv, max_slots, hd), &dev);
        let k_view = k_big.narrow(1, off, k_len).unwrap();
        let v_view = v_big.narrow(1, off, k_len).unwrap();
        assert!(!k_view.is_contiguous(), "the view must exercise the strided path");

        let strided = flash_attn(&q, &k_view, &v_view, pos, 0, None, scale).unwrap();
        let packed = flash_attn(
            &q,
            &k_view.contiguous().unwrap(),
            &v_view.contiguous().unwrap(),
            pos,
            0,
            None,
            scale,
        )
        .unwrap();
        assert_bits_eq(&strided, &packed, "head-strided vs packed");
    }

    /// UNIT 3: the block-level skip is exact — a windowed case whose expired /
    /// future KV blocks are skipped must be BIT-IDENTICAL to the same kernel
    /// with skipping disabled (which processes those blocks as all -inf:
    /// exp terms of exactly 0, rescale factor exactly 1).
    #[test]
    fn flash_attn_block_skip_is_exact() {
        let dev = metal_device().unwrap();
        let (n_head, n_kv, hd) = (18usize, 2usize, 128usize);
        let scale = 1.0f32 / (hd as f32).sqrt();

        // seq 40 (two query blocks), window 16, pos 0: KV block 0 is fully
        // expired for query block 1 (rows 32-39 see only columns >= 17), and
        // trailing KV blocks are fully future for query block 0.
        let (seq, pos, window) = (40usize, 0usize, 16usize);
        let q = rand_t(70, (n_head, seq, hd), &dev);
        let k16 = rand_kv(71, (n_kv, seq, hd), &dev);
        let v16 = rand_kv(72, (n_kv, seq, hd), &dev);

        let skipping =
            dispatch::run_flash_attn(&q, &k16, &v16, pos, 0, Some(window), scale, false).unwrap();
        let full =
            dispatch::run_flash_attn(&q, &k16, &v16, pos, 0, Some(window), scale, true).unwrap();
        assert_bits_eq(&skipping, &full, "block-skip vs no-skip");
    }

    /// Rejection paths: shape/dtype/stride/position preconditions fail cleanly.
    #[test]
    fn flash_attn_rejects_bad_inputs() {
        let dev = metal_device().unwrap();
        let q = rand_t(50, (6, 4, 128), &dev);
        let k = rand_kv(51, (1, 9, 128), &dev);
        let v = rand_kv(52, (1, 9, 128), &dev);
        let scale = 0.1f32;

        // Valid baseline (pos 5, K = 9 covers rows 5..9).
        flash_attn(&q, &k, &v, 5, 0, None, scale).unwrap();

        // f32 k.
        let k32 = k.to_dtype(DType::F32).unwrap();
        assert!(flash_attn(&q, &k32, &v, 5, 0, None, scale).is_err());
        // Wrong head_dim.
        let q64 = rand_t(53, (6, 4, 64), &dev);
        assert!(flash_attn(&q64, &k, &v, 5, 0, None, scale).is_err());
        // Own key out of range: rows reach position 9 but keys end at 8.
        let k_short = k.narrow(1, 0, 8).unwrap();
        let v_short = v.narrow(1, 0, 8).unwrap();
        assert!(flash_attn(&q, &k_short, &v_short, 5, 0, None, scale).is_err());
        // k_off beyond pos.
        assert!(flash_attn(&q, &k, &v, 5, 6, None, scale).is_err());
        // Zero window.
        assert!(flash_attn(&q, &k, &v, 5, 0, Some(0), scale).is_err());
        // Head-dim-strided (transposed) k is rejected, not silently misread.
        let k_t = rand_kv(54, (1, 128, 128), &dev).transpose(1, 2).unwrap();
        assert!(flash_attn(&q, &k_t, &k_t, 5, 0, None, scale).is_err());
        // n_head not a multiple of n_kv.
        let k2 = rand_kv(55, (4, 9, 128), &dev);
        assert!(flash_attn(&q, &k2, &k2, 5, 0, None, scale).is_err());
    }
}
