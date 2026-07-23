use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

/// Dense f16-weight x f32-activation matmul against the vendored ggml-geometry
/// kernels — the attention projections. `weight` is a rank-2 `[n_out, k]` dense
/// f16 tensor, `x` is `[t, k]` f32; returns `[t, n_out]` f32. Semantically the
/// fork's mixed-dtype mul_mat: f32 products/accumulation, f32 output.
///
/// Dispatches the classic mat-vec (`f16.metal`) for t <= 8 tokens; above that,
/// the prefill gemm — by default the classic simdgroup kernel (`f16.metal`), or
/// the opt-in Metal-4 cooperative-tensor kernel (`f16_t.metal`) under
/// `LAGUNA_ATTN_MM_TENSOR`. The tensor prefill gemm stages the activation as f16
/// (its only extra rounding over the classic float-tile kernel), which put our
/// decode drift outside the fork envelope, so it is opt-in, not shipped (see
/// docs/parity.md §3b). Metal only; the caller's fallback is the dequant-f32
/// `QMatMul` path (`LAGUNA_ATTN_F32`), which bypasses this module entirely.
pub fn matmul_f16(weight: &Tensor, x: &Tensor) -> Result<Tensor> {
    dispatch::run_matmul_f16(weight, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::ops::dispatch::testutil::{max_abs, pseudo_random, rel_l2};
    use candle_core::{DType, Device, Tensor};

    /// Kernel output vs a CPU f32 reference matmul over the SAME f16-rounded
    /// weights, on the CLASSIC path (float tiles): the kernel's only rounding is
    /// the stored weights, which the reference shares, so the residual is pure
    /// f32 accumulation-order noise. The tensor prefill kernel additionally
    /// rounds the activation to f16, so it is graded against the classic kernel
    /// (not the f32 CPU reference) in `f16_tensor_matches_classic`.
    fn run_shape(n_out: usize, k: usize, t: usize, seed: u64) -> f32 {
        let device = metal_device().unwrap();
        let cpu = Device::Cpu;

        let w = Tensor::from_vec(pseudo_random(n_out * k, seed, -0.5, 0.5), (n_out, k), &cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let x = Tensor::from_vec(pseudo_random(t * k, seed ^ 0xF00D, -1.0, 1.0), (t, k), &cpu).unwrap();

        // Classic mm variant (float tiles) for the tight f32-reference bound; the
        // mv branch (t <= 8) is identical for either variant.
        let got = dispatch::run_matmul_f16_variant(
            &w.to_device(&device).unwrap(),
            &x.to_device(&device).unwrap(),
            true,
        )
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

    // The classic kernels accumulate in f32 with f16 rounding only in the stored
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

    /// Classic prefill gemm (t > 8): the first mm seq (9), a real fixture seq
    /// (58, matching the code-short parity prompt), and a full 512-token chunk,
    /// over production out-dims including the sub-tile 48/72 gate projections
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
    /// gemv (t=8) and the CLASSIC gemm (t=9) must agree with the shared reference
    /// to the same bound (implicitly covered above) AND with each other row-for-row
    /// on the overlapping tokens.
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
        let mm = dispatch::run_matmul_f16_variant(&w, &x9, true).unwrap(); // t=9: classic gemm
        let mv = matmul_f16(&w, &x9.narrow(0, 0, 8).unwrap()).unwrap(); // t=8: gemv
        let a = mm.narrow(0, 0, 8).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = mv.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rel = rel_l2(&a, &b);
        assert!(rel < TOL, "mv/mm boundary rel_l2 {rel}");
    }

    /// The production attention projection shapes at a full prefill chunk, as
    /// (n_out, k): SWA q (72h -> 9216), FULL q (48h -> 6144), k/v (8kv -> 1024),
    /// o_proj (9216 -> 3072), and the sub-tile gate projections (48/72 rows, the
    /// guarded store-back). Every shape the tensor prefill gemm runs in production.
    const PREFILL_SHAPES: [(usize, usize); 6] = [
        (9216, 3072),
        (6144, 3072),
        (1024, 3072),
        (3072, 9216),
        (48, 3072),
        (72, 3072),
    ];

    /// The shipped tensor prefill gemm vs the classic simdgroup gemm on every
    /// production projection shape at a 512-token chunk. The tensor kernel stages
    /// the activation as f16, so it carries the fork's ~2e-4 prefill precision
    /// relative to the classic float-tile kernel (both share the f16 weights);
    /// bound at 5e-4 (the graduation of the prototype's numerics probe). This is
    /// the transitive correctness link: classic is pinned to the f32 CPU reference
    /// above, tensor is pinned to classic here.
    #[test]
    fn f16_tensor_matches_classic() {
        let device = metal_device().unwrap();
        const T: usize = 512;
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for (n_out, k) in PREFILL_SHAPES {
            let w = Tensor::from_vec(pseudo_random(n_out * k, 0x300 + n_out as u64, -0.5, 0.5), (n_out, k), &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            let x = Tensor::from_vec(pseudo_random(T * k, 0x400 + n_out as u64, -1.0, 1.0), (T, k), &device).unwrap();

            let tensor = flat(&dispatch::run_matmul_f16_variant(&w, &x, false).unwrap());
            let classic = flat(&dispatch::run_matmul_f16_variant(&w, &x, true).unwrap());

            // rel_l2 is the relative (scale-invariant) error, ~1.8e-4 on the worst
            // shape — the fork's own prefill precision class. max_abs is a raw
            // absolute diff (diagnostic only): it scales with the output magnitude
            // (K=3072..9216 dot products), so ~9e-3 there is ~3e-4 relative, not a
            // precision regression.
            let rel = rel_l2(&tensor, &classic);
            let mabs = max_abs(&tensor, &classic);
            assert!(
                rel < 5e-4,
                "tensor vs classic [{n_out}x{k}] t={T}: rel_l2 {rel} (max_abs {mabs})"
            );
        }
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

    /// Isolation timing: classic simdgroup gemm vs Metal-4 cooperative-tensor gemm
    /// on each production projection shape at a 512-token chunk. `#[ignore]`d —
    /// run on a `pgrep`-verified free GPU with:
    ///   cargo test --release -p laguna f16_tensor_vs_classic_timing -- --ignored --nocapture
    /// `LAGUNA_BENCH_WARMUP` / `LAGUNA_BENCH_ITERS` override the loop counts.
    #[test]
    #[ignore = "perf bench"]
    fn f16_tensor_vs_classic_timing() {
        use std::time::Instant;

        let device = metal_device().unwrap();
        const T: usize = 512;
        let read_scalar = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        let get = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        let (warm, iters) = (get("LAGUNA_BENCH_WARMUP", 10), get("LAGUNA_BENCH_ITERS", 100));

        // Warm-up then a timed loop, each iter ending in a small readback (the
        // per-iter command-buffer flush). Returns (mean, plateau = mean of the last
        // half) ms/iter — the LPM burst→clamp makes the plateau the honest figure.
        let bench = |name: &str, mut f: Box<dyn FnMut() -> f32>| {
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
            let plateau: f64 = times[iters / 2..].iter().sum::<f64>() / (iters - iters / 2) as f64;
            eprintln!("{name}: mean {mean:.3} ms | plateau {plateau:.3} ms (sink {sink:.1})");
        };

        for (n_out, k) in PREFILL_SHAPES {
            let w = Tensor::from_vec(pseudo_random(n_out * k, 0x100 + n_out as u64, -0.5, 0.5), (n_out, k), &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            let x = Tensor::from_vec(pseudo_random(T * k, 0x200 + n_out as u64, -1.0, 1.0), (T, k), &device).unwrap();

            eprintln!("--- [{n_out}x{k}] t={T} ---");
            let (w0, x0) = (w.clone(), x.clone());
            bench("  classic", Box::new(move || read_scalar(&dispatch::run_matmul_f16_variant(&w0, &x0, true).unwrap())));
            let (w1, x1) = (w.clone(), x.clone());
            bench("  tensor ", Box::new(move || read_scalar(&dispatch::run_matmul_f16_variant(&w1, &x1, false).unwrap())));
        }
    }
}
