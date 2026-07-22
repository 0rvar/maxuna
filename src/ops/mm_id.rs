use anyhow::Result;
use candle_core::Tensor;
use candle_core::quantized::GgmlDType;

use crate::gguf::ExpertStack;
use crate::ops::dispatch::{self, Mode};

/// Quantized gather-matmul over a stacked expert tensor (prefill path):
/// tokens are grouped per expert so each expert's rows are read once per chunk.
/// Dispatches the vendored `kernel_mul_mm_id_<dtype>_f32[_hp]`.
///
/// Shapes as in `mv_id::mul_mv_id`; t is the prefill chunk length.
pub fn mul_mm_id(stack: &ExpertStack, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    dispatch::run(stack, x, ids, Mode::Mm, crate::ops::mm_id_variant())
}

/// Whether the vendored two-pass mm_id kernels are instantiated for this dtype
/// and top_k. `moe` gates the seq>=32 prefill branch on this and falls back to
/// mv_id when a checkpoint uses an uninstantiated dtype/top_k.
pub fn supported(dt: GgmlDType, top_k: usize) -> bool {
    dispatch::mm_kernel_name(dt).is_ok() && dispatch::map0_kernel_name(top_k).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::ops::dispatch::testutil::*;
    use candle_core::Tensor;
    use candle_core::quantized::GgmlDType;

    const DTYPES: &[(GgmlDType, &str)] = &[
        (GgmlDType::Q8_0, "Q8_0"),
        (GgmlDType::Q4K, "Q4K"),
        (GgmlDType::Q5K, "Q5K"),
        (GgmlDType::Q6K, "Q6K"),
    ];

    /// Full matmul case: build stack, x, ids on device, dispatch, compare to oracle.
    fn run_case(
        dt: GgmlDType,
        n_expert: usize,
        n_out: usize,
        k: usize,
        t: usize,
        top_k: usize,
        x_per_row: usize,
        ids: Vec<u32>,
        seed: u64,
    ) -> (f32, f32) {
        let device = metal_device().unwrap();
        let (stack, deq) = build_stack(&device, dt, n_expert, n_out, k, seed).unwrap();

        let x_vec = pseudo_random(t * x_per_row * k, seed ^ 0xABCD, -1.0, 1.0);
        let x = Tensor::from_vec(x_vec.clone(), (t, x_per_row, k), &device).unwrap();
        let ids_t = Tensor::from_vec(ids.clone(), (t, top_k), &device).unwrap();

        let out = mul_mm_id(&stack, &x, &ids_t).unwrap();
        assert_eq!(out.dims(), &[t, top_k, n_out]);
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let want = oracle(&deq, &x_vec, &ids, n_out, k, t, top_k, x_per_row);
        (rel_l2(&got, &want), max_abs(&got, &want))
    }

    #[test]
    fn prefill_t16() {
        for (dt, name) in DTYPES {
            let (top_k, t) = (2, 16);
            let ids = distinct_ids(t, top_k, 4, 0x41);
            let (rel, max) = run_case(*dt, 4, 8, 256, t, top_k, 1, ids, 0x1600);
            assert!(rel < 1e-3, "{name} t=16 rel_l2 {rel} too high (max_abs {max})");
        }
    }

    #[test]
    fn prefill_t64() {
        for (dt, name) in DTYPES {
            let (top_k, t) = (2, 64);
            let ids = distinct_ids(t, top_k, 4, 0x42);
            let (rel, max) = run_case(*dt, 4, 8, 256, t, top_k, 1, ids, 0x6400);
            assert!(rel < 1e-3, "{name} t=64 rel_l2 {rel} too high (max_abs {max})");
        }
    }

    #[test]
    fn prefill_per_slot_shared_expert() {
        // Down-projection layout (x_per_row == top_k) where several tokens route
        // to the same expert, giving that expert's compacted row list multiple
        // columns spanning more than one 32-wide token block. Experts stay
        // distinct WITHIN a token (the routing invariant the two-pass kernel
        // relies on); here a small expert pool forces heavy cross-token reuse.
        for (dt, name) in DTYPES {
            let (top_k, t) = (4, 40);
            let ids = distinct_ids(t, top_k, 6, 0x43);
            let (rel, max) = run_case(*dt, 6, 8, 256, t, top_k, top_k, ids, 0x1601);
            assert!(rel < 1e-3, "{name} per-slot shared rel_l2 {rel} too high (max_abs {max})");
        }
    }

    /// The two-pass mm_id path must reproduce the dequantize-then-dot oracle at
    /// the production geometry (256 experts, top_k 10, a 512-token prefill
    /// chunk), for both the gate/up layout (one shared activation per token) and
    /// the down layout (a distinct activation per slot). mm_id multiplies in f16
    /// simdgroup tiles (accumulating in f32), so it lands ~2-3e-4 rel from the
    /// f32 oracle — an order looser than mv_id's ~1e-7, but far under any
    /// wiring-bug scale. This also exercises the readback-between-calls path that
    /// exposed the test-harness residency bug in `build_stack`.
    #[test]
    fn mm_matches_oracle_production_scale() {
        let device = metal_device().unwrap();
        let (n_expert, n_out, k) = (256usize, 64usize, 256usize);
        let (t, top_k) = (512usize, 10usize);

        for (dt, name) in &[(GgmlDType::Q4K, "Q4K"), (GgmlDType::Q6K, "Q6K")] {
            let (stack, deq) = build_stack(&device, *dt, n_expert, n_out, k, 0x51).unwrap();
            let ids = distinct_ids(t, top_k, n_expert, 0x52);
            let ids_t = Tensor::from_vec(ids.clone(), (t, top_k), &device).unwrap();

            // gate/up geometry: one shared activation row per token.
            let x_vec = pseudo_random(t * 1 * k, 0x53, -1.0, 1.0);
            let x = Tensor::from_vec(x_vec.clone(), (t, 1, k), &device).unwrap();
            let mm = mul_mm_id(&stack, &x, &ids_t).unwrap();
            let mm_v = mm.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let want = oracle(&deq, &x_vec, &ids, n_out, k, t, top_k, 1);
            let rel = rel_l2(&mm_v, &want);
            assert!(rel < 1e-3, "{name} gate/up rel_l2 {rel} too high (max_abs {})", max_abs(&mm_v, &want));

            // down-projection geometry: a distinct activation row per slot.
            let xd_vec = pseudo_random(t * top_k * k, 0x54, -1.0, 1.0);
            let xd = Tensor::from_vec(xd_vec.clone(), (t, top_k, k), &device).unwrap();
            let mm_d = mul_mm_id(&stack, &xd, &ids_t).unwrap();
            let mm_dv = mm_d.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let want_d = oracle(&deq, &xd_vec, &ids, n_out, k, t, top_k, top_k);
            let rel_d = rel_l2(&mm_dv, &want_d);
            assert!(rel_d < 1e-3, "{name} down rel_l2 {rel_d} too high (max_abs {})", max_abs(&mm_dv, &want_d));
        }
    }

    /// A/B the f32-tile `_hp` default against the f16-tile variant at production
    /// scale: the f32 tiles must land far tighter to the oracle than f16's
    /// ~2.6e-4. Also prints an amortized throughput ratio (informational). The
    /// variant is passed explicitly to `dispatch::run` (no env toggling), so this
    /// is safe under default parallel `cargo test`; add `--nocapture` to see the
    /// numbers.
    /// All three mm_id variants vs the dequantize-then-dot oracle at production
    /// scale, plus tensor-vs-classic-f16 agreement (both f16-operand, so they
    /// should track each other) and an amortized throughput print. Variants are
    /// passed explicitly to dispatch::run (no env), so this is parallel-safe.
    #[test]
    fn mm_variants_precision_and_throughput() {
        use crate::ops::MmVariant;
        use std::time::Instant;

        let device = metal_device().unwrap();
        let (n_expert, n_out, k) = (256usize, 256usize, 512usize);
        let (t, top_k) = (512usize, 10usize);

        for (dt, name) in &[(GgmlDType::Q4K, "Q4K"), (GgmlDType::Q6K, "Q6K")] {
            let (stack, deq) = build_stack(&device, *dt, n_expert, n_out, k, 0x71).unwrap();
            let ids = distinct_ids(t, top_k, n_expert, 0x72);
            let ids_t = Tensor::from_vec(ids.clone(), (t, top_k), &device).unwrap();
            let x_vec = pseudo_random(t * k, 0x73, -1.0, 1.0);
            let x = Tensor::from_vec(x_vec.clone(), (t, 1, k), &device).unwrap();
            let want = oracle(&deq, &x_vec, &ids, n_out, k, t, top_k, 1);

            let run_mm = |v: MmVariant| dispatch::run(&stack, &x, &ids_t, Mode::Mm, v).unwrap();
            let measure = |v: MmVariant| -> (Vec<f32>, f32, f64) {
                let out = run_mm(v);
                let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let rel = rel_l2(&got, &want);
                for _ in 0..8 {
                    let _ = run_mm(v);
                }
                device.synchronize().unwrap();
                let iters = 40usize;
                let start = Instant::now();
                for _ in 0..iters {
                    let _ = run_mm(v);
                }
                device.synchronize().unwrap();
                let tps = (t * iters) as f64 / start.elapsed().as_secs_f64();
                (got, rel, tps)
            };

            let (tensor_out, rel_t, tps_t) = measure(MmVariant::Tensor);
            let (_, rel_thp, tps_thp) = measure(MmVariant::TensorHp);
            let (_, rel_hp, tps_hp) = measure(MmVariant::ClassicHp);
            let (f16_out, rel_f16, tps_f16) = measure(MmVariant::ClassicF16);
            let tensor_vs_f16 = rel_l2(&tensor_out, &f16_out);

            eprintln!(
                "{name}: tensor rel_l2={rel_t:.2e} ({tps_t:.0} tok/s)  \
                 tensor-hp rel_l2={rel_thp:.2e} ({tps_thp:.0} tok/s)  \
                 classic-hp rel_l2={rel_hp:.2e} ({tps_hp:.0} tok/s)  \
                 classic-f16 rel_l2={rel_f16:.2e} ({tps_f16:.0} tok/s)  \
                 tensor-vs-f16 rel_l2={tensor_vs_f16:.2e}"
            );

            // f32-operand variants (classic-hp, tensor-hp) are f32-tight; the
            // f16-operand ones (tensor, classic-f16) are looser but well under a
            // wiring-bug scale, and tensor tracks classic-f16.
            assert!(rel_hp < 5e-5, "{name} classic-hp rel_l2 {rel_hp} not near f32 floor");
            assert!(rel_thp < 5e-5, "{name} tensor-hp rel_l2 {rel_thp} not near f32 floor");
            assert!(rel_t < 1e-3, "{name} tensor rel_l2 {rel_t} too high");
            assert!(rel_f16 < 1e-3, "{name} classic-f16 rel_l2 {rel_f16} too high");
            assert!(tensor_vs_f16 < 1e-3, "{name} tensor vs classic-f16 rel_l2 {tensor_vs_f16} too high");
        }
    }

    #[test]
    fn shape_mismatch_errors() {
        // ids t disagreeing with x t must error rather than fault the GPU.
        let device = metal_device().unwrap();
        let (stack, _) = build_stack(&device, GgmlDType::Q4K, 4, 8, 256, 1).unwrap();
        let x = Tensor::from_vec(vec![0f32; 16 * 1 * 256], (16, 1, 256), &device).unwrap();
        let ids = Tensor::from_vec(vec![0u32; 8 * 2], (8, 2), &device).unwrap();
        assert!(mul_mm_id(&stack, &x, &ids).is_err());
    }

    /// PHASE 3 PROBE: confirm ggml's Metal-4 cooperative tensor ops
    /// (`<metal_tensor>` + `mpp::tensor_ops::matmul2d`) compile under candle's
    /// default `new_library_with_source(None)` options and run on this device.
    /// Gates the tensor-path mm_id port: if this fails to compile, the port is a
    /// non-starter and the compiler error is the actionable output. Mirrors the
    /// fork's tile setup (NR0=64/NR1=32/NK=32, sc aliases sa, mm.run(sB,sA,cT)).
    /// A=B=ones -> each C entry should be the K-sum = 32.
    const TENSOR_PROBE_SRC: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void probe_matmul2d(
        device const half  * A [[buffer(0)]],
        device const half  * B [[buffer(1)]],
        device       float * C [[buffer(2)]],
        threadgroup  char  * shmem [[threadgroup(0)]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;

    threadgroup half  * sa = (threadgroup half  *)(shmem);
    threadgroup half  * sb = (threadgroup half  *)(shmem + 4096);
    threadgroup float * sc = (threadgroup float *)(shmem);

    for (uint i = tiitg; i < NK*NR0;  i += 128) { sa[i] = A[i]; }
    for (uint i = tiitg; i < NR1*NK;  i += 128) { sb[i] = B[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK,  NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK ));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NR1, NR0, NK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    auto sA = tA.slice(0, 0);
    auto sB = tB.slice(0, 0);
    mm.run(sB, sA, cT);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tiitg; i < NR0*NR1; i += 128) { C[i] = sc[i]; }
}
"#;

    #[test]
    fn tensor_matmul2d_probe() {
        use candle_core::{DType, Device, Storage};
        use candle_metal_kernels::metal::ComputeCommandEncoder;
        use candle_metal_kernels::utils::EncoderProvider;

        let device = metal_device().unwrap();
        let Device::Metal(mdev) = &device else { unreachable!("metal_device is Metal") };

        let lib = match mdev.device().new_library_with_source(TENSOR_PROBE_SRC, None) {
            Ok(l) => l,
            Err(e) => panic!(
                "PHASE 3 PROBE FAILED: cooperative tensor ops did not compile under candle's \
                 default compile options:\n{e}"
            ),
        };
        let func = lib.get_function("probe_matmul2d", None).expect("probe fn present");
        let pipeline = mdev
            .device()
            .new_compute_pipeline_state_with_function(&func)
            .expect("probe pipeline builds");

        // A [NK=32, NR0=64] and B [NR1=32, NK=32], both f16 ones.
        let a = Tensor::ones((32, 64), DType::F16, &device).unwrap();
        let b = Tensor::ones((32, 32), DType::F16, &device).unwrap();
        let c = mdev.new_buffer(64 * 32, DType::F32, "probe_c").unwrap();

        let (a_g, a_l) = a.storage_and_layout();
        let Storage::Metal(a_s) = &*a_g else { unreachable!() };
        let a_buf = a_s.buffer();
        let a_off = a_l.start_offset() * DType::F16.size_in_bytes();
        let (b_g, b_l) = b.storage_and_layout();
        let Storage::Metal(b_s) = &*b_g else { unreachable!() };
        let b_buf = b_s.buffer();
        let b_off = b_l.start_offset() * DType::F16.size_in_bytes();

        {
            let cmd = mdev.command_encoder().unwrap();
            let ep = &cmd;
            let enc = ep.encoder();
            let enc: &ComputeCommandEncoder = enc.as_ref();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_input_buffer(0, Some(a_buf), a_off);
            enc.set_input_buffer(1, Some(b_buf), b_off);
            enc.set_output_buffer(2, Some(&c), 0);
            enc.set_threadgroup_memory_length(0, 8192);
            let mut grid = candle_metal_kernels::utils::get_block_dims(1, 1, 1);
            grid.width = 1;
            let mut threads = candle_metal_kernels::utils::get_block_dims(1, 1, 1);
            threads.width = 128;
            enc.dispatch_thread_groups(grid, threads);
        }
        drop(a_g);
        drop(b_g);

        let out = dispatch::output_tensor(c, mdev, 64 * 32, (64, 32));
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "probe output has non-finite values");
        eprintln!(
            "tensor-ops probe OK: compiled + ran. C[0..4]={:?} (K-sum of ones is 32)",
            &v[0..4]
        );
    }
}
