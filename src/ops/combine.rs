use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

/// Fused MoE weighted combine against the vendored `combine.metal` kernels — the
/// routed-expert combine tail of `FusedExperts::forward`. Reads `down`
/// (`[seq, top_k, n_out]` f32 contiguous) once and returns `[seq, n_out]` f32:
///   - `col_l2 == None`  (rescale-free): `dst[s,c] = Σ_k down[s,k,c] * w[s,k]`
///   - `col_l2 == Some`  (`[seq, top_k, 1]` f32): also undoes the per-column L2
///     rescale — `dst[s,c] = Σ_k down[s,k,c] * col_l2[s,k] * 2^-15 * w[s,k]`.
/// `weights` is `[seq, top_k]` f32. Bit-identical to the candle broadcast/affine/
/// sum chain it replaces (combine.rs `fused_matches_candle_bitwise` proves it),
/// so the fused path is safe under every parity tier. Metal only; the caller's
/// kill-switch is the candle chain (`LAGUNA_COMBINE_CLASSIC`).
pub fn combine(down: &Tensor, col_l2: Option<&Tensor>, weights: &Tensor) -> Result<Tensor> {
    dispatch::run_combine(down, col_l2, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use candle_core::{DType, Device};

    /// F16's safe headroom — the exact `f16_safe` constant `FusedExperts::forward`
    /// scales the down projection by; `1.0 / f16_safe` is `2^-15`, exact.
    const F16_SAFE: f64 = 32768.0;

    /// A deterministic f32 with a wide magnitude span (`10^-6 .. 10^4`) and a
    /// random sign — the stress the bitwise test needs: catastrophic cancellation
    /// in the reduction would surface any accumulation-order difference between the
    /// fused kernel and the candle chain.
    fn wide(seed: u64, n: usize, signed: bool) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 // [0, 1)
        };
        (0..n)
            .map(|_| {
                let exp = -6.0 + next() * 10.0; // 10^-6 .. 10^4
                let mag = 10f64.powf(exp) as f32;
                if signed && next() < 0.5 { -mag } else { mag }
            })
            .collect()
    }

    /// The exact candle chain `FusedExperts::forward` runs today, for the given
    /// branch — the ground truth the fused kernel must reproduce bit-for-bit.
    fn candle_chain(
        down: &Tensor,
        col_l2: Option<&Tensor>,
        weights: &Tensor,
        seq: usize,
        top_k: usize,
    ) -> Tensor {
        let w = weights.reshape((seq, top_k, 1)).unwrap();
        match col_l2 {
            Some(l2) => {
                let d = (down.broadcast_mul(l2).unwrap() * (1.0 / F16_SAFE)).unwrap();
                d.broadcast_mul(&w).unwrap().sum(1).unwrap()
            }
            None => down.broadcast_mul(&w).unwrap().sum(1).unwrap(),
        }
    }

    /// The fused combine kernel must reproduce the live candle broadcast/affine/sum
    /// chain BIT-FOR-BIT (compared on `f32::to_bits`, not a tolerance), for both
    /// branches, across the production seq / n_out / top_k grid with wide-magnitude
    /// signed inputs. Bit-identity is the whole justification for shipping the
    /// fused kernel on the strict parity tier, so any mismatch is a hard failure —
    /// never loosen this to a tolerance.
    #[test]
    fn fused_matches_candle_bitwise() {
        let device = metal_device().unwrap();
        let cpu = Device::Cpu;
        let top_k = 10usize;

        for &seq in &[1usize, 8, 512] {
            for &n_out in &[1024usize, 3072] {
                let down_v = wide(0x100 + seq as u64 * 31 + n_out as u64, seq * top_k * n_out, true);
                // col_l2 is a positive per-column norm in production.
                let l2_v: Vec<f32> = wide(0x200 + seq as u64, seq * top_k, false);
                let w_v = wide(0x300 + n_out as u64, seq * top_k, true);

                let down = Tensor::from_vec(down_v, (seq, top_k, n_out), &cpu)
                    .unwrap()
                    .to_device(&device)
                    .unwrap();
                let col_l2 = Tensor::from_vec(l2_v, (seq, top_k, 1), &cpu)
                    .unwrap()
                    .to_device(&device)
                    .unwrap();
                let weights = Tensor::from_vec(w_v, (seq, top_k), &cpu)
                    .unwrap()
                    .to_device(&device)
                    .unwrap();

                for (label, l2) in [("rescale", Some(&col_l2)), ("plain", None)] {
                    let fused = combine(&down, l2, &weights).unwrap();
                    assert_eq!(fused.dims(), &[seq, n_out]);
                    assert_eq!(fused.dtype(), DType::F32);
                    let want = candle_chain(&down, l2, &weights, seq, top_k);

                    let fb: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();
                    let wb: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
                    assert_eq!(fb.len(), wb.len());
                    for (i, (f, w)) in fb.iter().zip(wb.iter()).enumerate() {
                        assert_eq!(
                            f.to_bits(),
                            w.to_bits(),
                            "{label} combine seq={seq} n_out={n_out}: element {i} differs \
                             (fused {f:?} bits {:#010x}, candle {w:?} bits {:#010x})",
                            f.to_bits(),
                            w.to_bits(),
                        );
                    }
                }
            }
        }
    }

    /// The combine kernels reduce with a single 32-lane `simd_sum`, so a top_k
    /// whose candle threadgroup width `next_pow2(top_k/2)` exceeds 32 would
    /// silently drop lanes 32.. — `run_combine` must bail (an error, not a
    /// fallback). top_k=66 is the first such width (66/2=33 → next_pow2=64 > 32);
    /// production top_k is 10 (width 8), so this is out of contract.
    #[test]
    fn combine_bails_on_wide_top_k() {
        let device = metal_device().unwrap();
        let top_k = 66usize;
        let down = Tensor::zeros((1, top_k, 4), DType::F32, &device).unwrap();
        let w = Tensor::zeros((1, top_k), DType::F32, &device).unwrap();
        let err = combine(&down, None, &w).unwrap_err().to_string();
        assert!(err.contains("> 32"), "unexpected error: {err}");
    }

    #[test]
    fn shape_and_dtype_errors() {
        let device = metal_device().unwrap();
        // weights shape mismatch.
        let down = Tensor::zeros((4, 10, 8), DType::F32, &device).unwrap();
        let bad_w = Tensor::zeros((4, 9), DType::F32, &device).unwrap();
        assert!(combine(&down, None, &bad_w).is_err());
        // col_l2 wrong last dim.
        let w = Tensor::zeros((4, 10), DType::F32, &device).unwrap();
        let bad_l2 = Tensor::zeros((4, 10, 2), DType::F32, &device).unwrap();
        assert!(combine(&down, Some(&bad_l2), &w).is_err());
        // down not f32.
        let down_f16 = Tensor::zeros((4, 10, 8), DType::F16, &device).unwrap();
        assert!(combine(&down_f16, None, &w).is_err());
    }
}
