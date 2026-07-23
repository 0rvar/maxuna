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

/// PROTOTYPE isolation bench (not production-wired): times the shipped classic
/// simdgroup attention prefill gemm (`matmul_f16`) against two Metal-4
/// cooperative-tensor ports (src/ops/f16_t_proto.metal) on the real attention
/// projection shapes, and probes their numeric drift from the classic kernel.
/// All `#[ignore]`d; see the module doc comment for how to run.
#[cfg(test)]
mod proto_bench {
    use super::matmul_f16;
    use crate::gguf::metal_device;
    use crate::ops::dispatch::{output_tensor, testutil};
    use candle_core::{DType, Device, MetalDevice, Storage, Tensor};
    use candle_metal_kernels::metal::{ComputePipeline, ComputeCommandEncoder};
    use candle_metal_kernels::utils::EncoderProvider;
    use std::time::Instant;

    const PROTO_SRC: &str = include_str!("f16_t_proto.metal");

    /// `ggml_metal_kargs_mul_mm` (== dispatch.rs's private `MmArgs` /
    /// f16_t_proto.metal's `mm_args`). Redeclared here so the prototype bench does
    /// not reach into dispatch's private struct.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct MmArgs {
        ne00: i32,
        ne02: i32,
        nb01: u64,
        nb02: u64,
        nb03: u64,
        ne12: i32,
        nb10: u64,
        nb11: u64,
        nb12: u64,
        nb13: u64,
        ne0: i32,
        ne1: i32,
        r2: i16,
        r3: i16,
    }

    /// `MTLSize` is `objc2_metal`'s (deliberately not a laguna dep) so it cannot be
    /// named in a signature; build it through candle's factory as dispatch.rs does.
    macro_rules! mtl_size {
        ($w:expr, $h:expr, $d:expr) => {{
            let mut sz = candle_metal_kernels::utils::get_block_dims(1, 1, 1);
            sz.width = $w;
            sz.height = $h;
            sz.depth = $d;
            sz
        }};
    }

    fn iter_counts() -> (usize, usize) {
        let get = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        (get("LAGUNA_BENCH_WARMUP", 10), get("LAGUNA_BENCH_ITERS", 100))
    }

    fn dense(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
        testutil::pseudo_random(rows * cols, seed, -0.5, 0.5)
    }

    /// Compile both prototype tensor kernels once against the live device.
    fn compile(mdev: &MetalDevice) -> (ComputePipeline, ComputePipeline) {
        let lib = mdev
            .device()
            .new_library_with_source(PROTO_SRC, None)
            .expect("f16_t_proto.metal compiles (Metal-4 cooperative tensors, see mm_id probe)");
        let build = |name: &str| {
            let func = lib.get_function(name, None).unwrap_or_else(|e| panic!("fn {name}: {e}"));
            mdev.device()
                .new_compute_pipeline_state_with_function(&func)
                .unwrap_or_else(|e| panic!("pipeline {name}: {e}"))
        };
        (build("kernel_mul_mm_f16_f32_t"), build("kernel_mul_mm_f16_f32_t_hp"))
    }

    /// Dispatch one prototype gemm: f16 `weight` [n_out, k] x f32 `x` [t, k] ->
    /// f32 [t, n_out]. `smem` is the tile threadgroup memory (8192 for the half
    /// variant, 12288 for the float variant). Same geometry as the shipped gemm.
    fn proto_gemm(
        mdev: &MetalDevice,
        pipeline: &ComputePipeline,
        weight: &Tensor,
        x: &Tensor,
        smem: usize,
    ) -> Tensor {
        let (n_out, k) = weight.dims2().unwrap();
        let (t, _) = x.dims2().unwrap();
        let nb01 = (k * DType::F16.size_in_bytes()) as u64;
        let nb11 = (k * DType::F32.size_in_bytes()) as u64;
        let args = MmArgs {
            ne00: k as i32,
            ne02: 1,
            nb01,
            nb02: n_out as u64 * nb01,
            nb03: n_out as u64 * nb01,
            ne12: 1,
            nb10: DType::F32.size_in_bytes() as u64,
            nb11,
            nb12: t as u64 * nb11,
            nb13: t as u64 * nb11,
            ne0: n_out as i32,
            ne1: t as i32,
            r2: 1,
            r3: 1,
        };

        let out_count = t * n_out;
        let dst = mdev.new_buffer(out_count, DType::F32, "f16_t_proto").unwrap();

        let (w_g, w_l) = weight.storage_and_layout();
        let Storage::Metal(w_s) = &*w_g else { unreachable!() };
        let w_buf = w_s.buffer();
        let w_off = w_l.start_offset() * DType::F16.size_in_bytes();
        let (x_g, x_l) = x.storage_and_layout();
        let Storage::Metal(x_s) = &*x_g else { unreachable!() };
        let x_buf = x_s.buffer();
        let x_off = x_l.start_offset() * DType::F32.size_in_bytes();

        {
            let cmd = mdev.command_encoder().unwrap();
            let ep = &cmd;
            let enc = ep.encoder();
            let enc: &ComputeCommandEncoder = enc.as_ref();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_bytes(0, &args);
            enc.set_input_buffer(1, Some(w_buf), w_off);
            enc.set_input_buffer(2, Some(x_buf), x_off);
            enc.set_output_buffer(3, Some(&dst), 0);
            enc.set_threadgroup_memory_length(0, smem);
            enc.dispatch_thread_groups(mtl_size!(t.div_ceil(32), n_out.div_ceil(64), 1), mtl_size!(128, 1, 1));
        }
        drop(w_g);
        drop(x_g);
        output_tensor(dst, mdev, out_count, (t, n_out))
    }

    fn read_scalar(t: &Tensor) -> f32 {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]
    }

    /// Warm-up then a timed loop, ending each iter in a small readback (the
    /// per-iter command-buffer flush). Prints a coarse per-decile time series so
    /// the LPM burst→plateau is visible, and returns (overall mean, plateau mean =
    /// mean of the last half of the timed iters), both ms/iter.
    fn bench(name: &str, mut f: impl FnMut() -> f32) -> (f64, f64) {
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
        let plateau_from = iters / 2;
        let plateau: f64 = times[plateau_from..].iter().sum::<f64>() / (iters - plateau_from) as f64;

        let chunks = 10.min(iters);
        let cs = iters / chunks;
        let series: Vec<String> = (0..chunks)
            .map(|c| {
                let seg = &times[c * cs..((c + 1) * cs).min(iters)];
                format!("{:.2}", seg.iter().sum::<f64>() / seg.len() as f64)
            })
            .collect();
        eprintln!(
            "{name}: mean {mean:.3} ms | plateau(last half) {plateau:.3} ms | min {min:.3} ms | series [{}] (sink {sink:.1})",
            series.join(" ")
        );
        (mean, plateau)
    }

    /// The four dense attention projection shapes at the 512-token prefill chunk,
    /// as (label, n_out, k): SWA q (72h), FULL q (48h), k/v (8kv), o_proj (SWA
    /// 72h input). t is fixed at 512.
    const PREFILL_T: usize = 512;
    const SHAPES: [(&str, usize, usize); 4] = [
        ("SWA q     [512,3072]->9216", 9216, 3072),
        ("FULL q    [512,3072]->6144", 6144, 3072),
        ("k/v       [512,3072]->1024", 1024, 3072),
        ("o_proj    [512,9216]->3072", 3072, 9216),
    ];

    /// Timing: classic (shipped `matmul_f16`) vs variant A (`_t_hp`, float tiles)
    /// vs variant B (`_t`, half tiles) on each projection shape. One cargo
    /// invocation, one model process — run with `pgrep`-verified free GPU.
    #[test]
    #[ignore = "perf prototype bench"]
    fn f16_t_proto_timing_bench() {
        let device = metal_device().unwrap();
        let Device::Metal(mdev) = &device else { unreachable!() };
        let (pipe_t, pipe_t_hp) = compile(mdev);

        for (label, n_out, k) in SHAPES {
            let w = Tensor::from_vec(dense(n_out, k, 0x100 + n_out as u64), (n_out, k), &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            let x = Tensor::from_vec(
                testutil::pseudo_random(PREFILL_T * k, 0x200 + n_out as u64, -1.0, 1.0),
                (PREFILL_T, k),
                &device,
            )
            .unwrap();

            eprintln!("--- {label} (t={PREFILL_T}) ---");
            bench("  classic ", || read_scalar(&matmul_f16(&w, &x).unwrap()));
            bench("  A _t_hp ", || read_scalar(&proto_gemm(mdev, &pipe_t_hp, &w, &x, 12288)));
            bench("  B _t    ", || read_scalar(&proto_gemm(mdev, &pipe_t, &w, &x, 8192)));
        }
    }

    /// Numerics: max relative error (rel-L2 and max-abs, elementwise) of variant A
    /// and B vs the shipped classic kernel — the reference point the task asks
    /// for. Classic and A both keep activations unrounded, so A should match
    /// classic to f32 accumulation-reorder noise; B rounds activations to f16, so
    /// it carries the fork's ~2e-4 prefill precision.
    #[test]
    #[ignore = "perf prototype bench"]
    fn f16_t_proto_numerics() {
        let device = metal_device().unwrap();
        let Device::Metal(mdev) = &device else { unreachable!() };
        let (pipe_t, pipe_t_hp) = compile(mdev);

        for (label, n_out, k) in SHAPES {
            let w = Tensor::from_vec(dense(n_out, k, 0x300 + n_out as u64), (n_out, k), &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            let x = Tensor::from_vec(
                testutil::pseudo_random(PREFILL_T * k, 0x400 + n_out as u64, -1.0, 1.0),
                (PREFILL_T, k),
                &device,
            )
            .unwrap();

            let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let classic = flat(&matmul_f16(&w, &x).unwrap());
            let a = flat(&proto_gemm(mdev, &pipe_t_hp, &w, &x, 12288));
            let b = flat(&proto_gemm(mdev, &pipe_t, &w, &x, 8192));

            eprintln!(
                "{label}: A(_t_hp) rel_l2 {:.2e} max_abs {:.2e} | B(_t) rel_l2 {:.2e} max_abs {:.2e}",
                testutil::rel_l2(&a, &classic),
                testutil::max_abs(&a, &classic),
                testutil::rel_l2(&b, &classic),
                testutil::max_abs(&b, &classic),
            );
        }
    }
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
