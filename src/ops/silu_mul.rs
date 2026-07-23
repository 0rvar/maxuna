use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

/// Fused MoE SwiGLU activation against the vendored `silu_mul.metal` kernel — the
/// silu*mul glue between the up/gate expert matvecs and the down matvec in
/// `FusedExperts::forward`. Reads `gate` and `up` (same-shape f32 contiguous, the
/// `[seq, top_k, expert_ff]` expert-matvec outputs) once and returns their
/// `silu(gate) * up`, same shape and dtype. Bit-identical to the candle
/// `silu(gate) * up` chain it replaces (silu_mul.rs `fused_matches_candle_bitwise`
/// proves it), so the fused path is safe under every parity tier. Metal only; the
/// caller's kill-switch is the candle chain (`LAGUNA_ACT_CLASSIC`).
pub fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    dispatch::run_silu_mul(gate, up)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use candle_core::{DType, Device};
    use candle_nn::ops::silu;

    /// A deterministic f32 with a wide magnitude span (`10^-6 .. 10^4`) and a
    /// random sign — silu's `exp(-x)` saturates at both ends, so this exercises
    /// the small-, mid- and large-magnitude regimes where the fused kernel and
    /// candle's silu could round differently if the arithmetic diverged.
    fn wide(seed: u64, n: usize) -> Vec<f32> {
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
                if next() < 0.5 { -mag } else { mag }
            })
            .collect()
    }

    /// The exact candle chain `FusedExperts::forward` runs when `LAGUNA_ACT_CLASSIC`
    /// is set — the ground truth the fused kernel must reproduce bit-for-bit.
    fn candle_chain(gate: &Tensor, up: &Tensor) -> Tensor {
        (silu(gate).unwrap() * up).unwrap()
    }

    /// The fused activation kernel must reproduce the live candle `silu(gate) * up`
    /// chain BIT-FOR-BIT (compared on `f32::to_bits`, not a tolerance), across the
    /// production seq / top_k / expert_ff grid with wide-magnitude signed inputs.
    /// Bit-identity is the whole justification for shipping the fused kernel on the
    /// strict parity tier, so any mismatch is a hard failure — never loosen this to
    /// a tolerance.
    #[test]
    fn fused_matches_candle_bitwise() {
        let device = metal_device().unwrap();
        let cpu = Device::Cpu;
        let top_k = 10usize;
        // Decode (seq 1) and prefill (seq 8, 512) shapes at the production
        // expert_ff (1024) plus a non-multiple width to catch tail-thread bugs.
        for &seq in &[1usize, 8, 512] {
            for &expert_ff in &[1024usize, 1000] {
                let n = seq * top_k * expert_ff;
                let gate_v = wide(0x100 + seq as u64 * 31 + expert_ff as u64, n);
                let up_v = wide(0x900 + seq as u64 * 17 + expert_ff as u64, n);

                let gate = Tensor::from_vec(gate_v, (seq, top_k, expert_ff), &cpu)
                    .unwrap()
                    .to_device(&device)
                    .unwrap();
                let up = Tensor::from_vec(up_v, (seq, top_k, expert_ff), &cpu)
                    .unwrap()
                    .to_device(&device)
                    .unwrap();

                let fused = silu_mul(&gate, &up).unwrap();
                assert_eq!(fused.dims(), &[seq, top_k, expert_ff]);
                assert_eq!(fused.dtype(), DType::F32);
                let want = candle_chain(&gate, &up);

                let fb: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();
                let wb: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
                assert_eq!(fb.len(), wb.len());
                for (i, (f, w)) in fb.iter().zip(wb.iter()).enumerate() {
                    assert_eq!(
                        f.to_bits(),
                        w.to_bits(),
                        "silu_mul seq={seq} expert_ff={expert_ff}: element {i} differs \
                         (fused {f:?} bits {:#010x}, candle {w:?} bits {:#010x})",
                        f.to_bits(),
                        w.to_bits(),
                    );
                }
            }
        }
    }

    #[test]
    fn shape_and_dtype_errors() {
        let device = metal_device().unwrap();
        // Shape mismatch between gate and up.
        let gate = Tensor::zeros((4, 10, 8), DType::F32, &device).unwrap();
        let bad_up = Tensor::zeros((4, 10, 9), DType::F32, &device).unwrap();
        assert!(silu_mul(&gate, &bad_up).is_err());
        // Non-f32 operands.
        let up = Tensor::zeros((4, 10, 8), DType::F32, &device).unwrap();
        let gate_f16 = Tensor::zeros((4, 10, 8), DType::F16, &device).unwrap();
        assert!(silu_mul(&gate_f16, &up).is_err());
        let up_f16 = Tensor::zeros((4, 10, 8), DType::F16, &device).unwrap();
        assert!(silu_mul(&gate, &up_f16).is_err());
    }
}
