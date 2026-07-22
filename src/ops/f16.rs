use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

/// Dense f16-weight x f32-activation matmul against the vendored ggml-geometry
/// kernels (src/ops/f16.metal) — the attention projections. `weight` is a
/// rank-2 `[n_out, k]` dense f16 tensor, `x` is `[t, k]` f32; returns
/// `[t, n_out]` f32. Semantically the fork's mixed-dtype mul_mat: the stored
/// f16 weights are the ONLY f16 in the chain — no activation cast, f32
/// products/accumulation, f32 output (candle's own f16 matmul rounds both the
/// activation and the output). Dispatches the gemv for t <= 8 tokens and the
/// tiled gemm above, mirroring ggml's host split. Metal only; the caller's
/// fallback is the dequant-f32 `QMatMul` path (`LAGUNA_ATTN_F32`).
pub fn matmul_f16(weight: &Tensor, x: &Tensor) -> Result<Tensor> {
    dispatch::run_matmul_f16(weight, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::ops::dispatch::testutil::{pseudo_random, rel_l2};
    use candle_core::{DType, Device};

    /// Kernel output vs a CPU f32 reference matmul over the SAME f16-rounded
    /// weights: the kernel's only rounding is the stored weights, which the
    /// reference shares, so the residual is pure f32 accumulation-order noise.
    fn run_shape(n_out: usize, k: usize, t: usize, seed: u64) -> f32 {
        let device = metal_device().unwrap();
        let cpu = Device::Cpu;

        let w = Tensor::from_vec(pseudo_random(n_out * k, seed, -0.5, 0.5), (n_out, k), &cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let x = Tensor::from_vec(pseudo_random(t * k, seed ^ 0xF00D, -1.0, 1.0), (t, k), &cpu).unwrap();

        let got = matmul_f16(&w.to_device(&device).unwrap(), &x.to_device(&device).unwrap())
            .unwrap();
        assert_eq!(got.dims(), &[t, n_out]);
        assert_eq!(got.dtype(), DType::F32);
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let want = x
            .matmul(&w.to_dtype(DType::F32).unwrap().t().unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        rel_l2(&got, &want)
    }

    // Both kernels accumulate in f32 with f16 rounding only in the stored
    // weights (which the reference shares), so the measured error is f32
    // accumulation-order noise: rel_l2 3.3e-7..7.1e-7 on the gemv and
    // 9.3e-7..1.8e-6 on the gemm (worst at t=512, K=9216) across the shapes
    // below. Bound at 1e-5 (~5x headroom over the worst).
    const TOL: f32 = 1e-5;

    /// Decode gemv (t <= 8) at every production projection shape, including
    /// the tiny 48/72-row gate projections and the o_proj K=9216 shape.
    #[test]
    fn f16_mv_production_shapes() {
        for (n_out, k) in [(6144, 3072), (9216, 3072), (1024, 3072), (48, 3072), (72, 3072), (3072, 9216)] {
            let rel = run_shape(n_out, k, 1, 0x51 + n_out as u64);
            assert!(rel < TOL, "mv [{n_out}x{k}] t=1 rel_l2 {rel}");
        }
        // The mv/mm boundary: t = 8 is the last gemv seq.
        let rel = run_shape(1024, 3072, 8, 0x61);
        assert!(rel < TOL, "mv t=8 rel_l2 {rel}");
    }

    /// Prefill gemm (t > 8): the first mm seq (9), a real fixture seq (58,
    /// matching the code-short parity prompt), and a full 512-token chunk, over
    /// production out-dims including the sub-tile 48/72 gate projections
    /// (nr0 < 64: guarded store-back) and the o_proj K=6144/9216 shapes.
    #[test]
    fn f16_mm_production_shapes() {
        for (n_out, k, t) in [
            (1024, 3072, 9),
            (9216, 3072, 58),
            (48, 3072, 58),
            (72, 3072, 58),
            (3072, 6144, 58),
            (6144, 3072, 512),
            (3072, 9216, 512),
        ] {
            let rel = run_shape(n_out, k, t, 0x71 + n_out as u64 + t as u64);
            assert!(rel < TOL, "mm [{n_out}x{k}] t={t} rel_l2 {rel}");
        }
    }

    /// The two kernels are one op behind a seq threshold: at adjacent seqs the
    /// gemv (t=8) and gemm (t=9) must agree with the shared reference to the
    /// same bound (implicitly covered above) AND with each other row-for-row on
    /// the overlapping tokens.
    #[test]
    fn f16_mv_mm_boundary_agrees() {
        let device = metal_device().unwrap();
        let cpu = Device::Cpu;
        let (n_out, k) = (1024, 3072);
        let w = Tensor::from_vec(pseudo_random(n_out * k, 0x81, -0.5, 0.5), (n_out, k), &cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&device)
            .unwrap();
        let x9 = Tensor::from_vec(pseudo_random(9 * k, 0x82, -1.0, 1.0), (9, k), &device).unwrap();
        let mm = matmul_f16(&w, &x9).unwrap(); // t=9: gemm
        let mv = matmul_f16(&w, &x9.narrow(0, 0, 8).unwrap()).unwrap(); // t=8: gemv
        let a = mm.narrow(0, 0, 8).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = mv.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rel = rel_l2(&a, &b);
        assert!(rel < TOL, "mv/mm boundary rel_l2 {rel}");
    }

    #[test]
    fn f16_shape_and_dtype_errors() {
        let device = metal_device().unwrap();
        let w = Tensor::zeros((64, 32), DType::F16, &device).unwrap();
        // k mismatch.
        let x = Tensor::zeros((1, 64), DType::F32, &device).unwrap();
        assert!(matmul_f16(&w, &x).is_err());
        // f32 weight (must be pre-cast f16 at load, not here).
        let wf = Tensor::zeros((64, 32), DType::F32, &device).unwrap();
        let x = Tensor::zeros((1, 32), DType::F32, &device).unwrap();
        assert!(matmul_f16(&wf, &x).is_err());
        // k not a multiple of 32 (the kernels have no K tail at our shapes).
        let w20 = Tensor::zeros((64, 20), DType::F16, &device).unwrap();
        let x20 = Tensor::zeros((1, 20), DType::F32, &device).unwrap();
        assert!(matmul_f16(&w20, &x20).is_err());
    }
}
