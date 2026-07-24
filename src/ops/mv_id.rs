use std::sync::OnceLock;

use anyhow::Result;
use candle_core::Tensor;
use candle_core::quantized::GgmlDType;
use candle_metal_kernels::metal::Buffer;

use crate::gguf::ExpertStack;
use crate::ops::dispatch::{self, Mode};

/// `LAGUNA_MV_CLASSIC` reverts every vendored ggml-geometry mat-vec (both the
/// routed expert gather and the lm_head bypass) to candle's baked
/// `kernel_mul_mv_id_<dtype>_f32` / QMatMul path. Read once and cached — it is
/// consulted per MoE layer and per lm_head on the hot path.
///
/// PRESENCE-BASED, like the `LAGUNA_MM_ID_*` / `LAGUNA_NO_MM_ID` toggles: any
/// value (even `LAGUNA_MV_CLASSIC=0`) enables the classic path — only leaving it
/// unset keeps the vendored kernels.
pub fn mv_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("LAGUNA_MV_CLASSIC").is_some())
}

/// Quantized gather-matvec over a stacked expert tensor (decode path).
///
/// By default dispatches the vendored ggml-geometry kernel
/// (`kernel_mul_mv_id_<dtype>_f32_v`, src/ops/mv.metal) for the supported
/// dtypes — q4_K/q6_K (official experts), q5_K and q8_0 (unsloth UD experts) —
/// in current ggml geometry (the K-quant per-simdgroup row fan-out, or q8_0's
/// shmem K-split). `LAGUNA_MV_CLASSIC` (or an unsupported dtype) falls back to
/// candle's baked `kernel_mul_mv_id_<dtype>_f32`.
///
/// x: [t, x_per_row, k] f32 — x_per_row is 1 when every selected expert of a
/// token consumes the same activation (gate/up), top_k for the down projection.
/// ids: [t, top_k] u32, on-device.
/// Returns [t, top_k, n_out] f32.
pub fn mul_mv_id(stack: &ExpertStack, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let mode = if mv_classic() || !dispatch::mv_vendored_supported(stack.dtype) {
        Mode::Mv
    } else {
        Mode::MvVendored
    };
    // The mm variant is a no-op for the matvec path (both Mv modes ignore it).
    dispatch::run(stack, x, ids, mode, crate::ops::MmVariant::ClassicHp)
}

/// Plain quantized mat-vec against the vendored ggml-geometry kernel — the
/// lm_head bypass at seq==1. `weight` is the rank-2 `[n_out, k]` quantized
/// tensor's raw device buffer (the caller retains it at load, same zero-copy
/// trick as `ExpertStack.buffer`). `x` is `[t, k]` f32; returns `[t, n_out]` f32.
/// Supports the vendored dtypes (`mv_vendored_supported` — q4_K/q5_K/q6_K/q8_0;
/// the official lm_head is q6_K, the UD lm_head q8_0); callers gate on
/// `mv_vendored_supported` and `mv_classic` and fall back to QMatMul otherwise.
pub fn mul_mv(weight: &Buffer, dtype: GgmlDType, n_out: usize, k: usize, x: &Tensor) -> Result<Tensor> {
    dispatch::run_plain_mv(weight, dtype, n_out, k, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::ops::dispatch::testutil::*;
    use candle_core::quantized::GgmlDType;
    use candle_core::{Module, Tensor};

    const DTYPES: &[(GgmlDType, &str)] = &[
        (GgmlDType::Q8_0, "Q8_0"),
        (GgmlDType::Q4K, "Q4K"),
        (GgmlDType::Q5K, "Q5K"),
        (GgmlDType::Q6K, "Q6K"),
    ];

    /// Full matvec case: build stack, x, ids on device, dispatch, compare to oracle.
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

        let out = mul_mv_id(&stack, &x, &ids_t).unwrap();
        assert_eq!(out.dims(), &[t, top_k, n_out]);
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let want = oracle(&deq, &x_vec, &ids, n_out, k, t, top_k, x_per_row);
        (rel_l2(&got, &want), max_abs(&got, &want))
    }

    #[test]
    fn shared_row_gate_up() {
        // x_per_row == 1: every slot of a token reads the same activation row.
        for (dt, name) in DTYPES {
            let top_k = 2;
            let t = 3;
            let ids = random_ids(t, top_k, 4, 0x11);
            let (rel, max) = run_case(*dt, 4, 8, 256, t, top_k, 1, ids, 0x100);
            assert!(rel < 1e-3, "{name} shared-row rel_l2 {rel} too high (max_abs {max})");
        }
    }

    #[test]
    fn per_slot_row_down() {
        // x_per_row == top_k: each slot reads its own activation row (down proj).
        for (dt, name) in DTYPES {
            let top_k = 4;
            let t = 5;
            let ids = random_ids(t, top_k, 4, 0x22);
            let (rel, max) = run_case(*dt, 4, 8, 256, t, top_k, top_k, ids, 0x200);
            assert!(rel < 1e-3, "{name} per-slot rel_l2 {rel} too high (max_abs {max})");
        }
    }

    #[test]
    fn repeated_expert_ids() {
        // Same expert selected in every slot of every token: exercises the id
        // decode when many (token, slot) pairs collapse onto one expert row set.
        for (dt, name) in DTYPES {
            let top_k = 3;
            let t = 4;
            let ids = vec![1u32; t * top_k];
            let (rel, max) = run_case(*dt, 4, 8, 256, t, top_k, 1, ids, 0x300);
            assert!(rel < 1e-3, "{name} repeated-id rel_l2 {rel} too high (max_abs {max})");
        }
    }

    #[test]
    fn shape_mismatch_errors() {
        // k mismatch must return an error, not fault the GPU.
        let device = metal_device().unwrap();
        let (stack, _) = build_stack(&device, GgmlDType::Q4K, 4, 8, 256, 1).unwrap();
        let x = Tensor::from_vec(vec![0f32; 1 * 1 * 128], (1, 1, 128), &device).unwrap();
        let ids = Tensor::from_vec(vec![0u32, 1u32], (1, 2), &device).unwrap();
        assert!(mul_mv_id(&stack, &x, &ids).is_err());
    }

    // ---- Vendored ggml-geometry kernels (mv.metal) -------------------------

    /// Same as `run_case` but forces the requested dispatch mode, so a test can
    /// exercise the vendored `Mode::MvVendored` path regardless of env toggles.
    #[allow(clippy::too_many_arguments)]
    fn run_case_mode(
        dt: GgmlDType,
        n_expert: usize,
        n_out: usize,
        k: usize,
        t: usize,
        top_k: usize,
        x_per_row: usize,
        ids: Vec<u32>,
        seed: u64,
        mode: Mode,
    ) -> (f32, f32) {
        let device = metal_device().unwrap();
        let (stack, deq) = build_stack(&device, dt, n_expert, n_out, k, seed).unwrap();

        let x_vec = pseudo_random(t * x_per_row * k, seed ^ 0xABCD, -1.0, 1.0);
        let x = Tensor::from_vec(x_vec.clone(), (t, x_per_row, k), &device).unwrap();
        let ids_t = Tensor::from_vec(ids.clone(), (t, top_k), &device).unwrap();

        let out = dispatch::run(&stack, &x, &ids_t, mode, crate::ops::MmVariant::ClassicHp).unwrap();
        assert_eq!(out.dims(), &[t, top_k, n_out]);
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let want = oracle(&deq, &x_vec, &ids, n_out, k, t, top_k, x_per_row);
        (rel_l2(&got, &want), max_abs(&got, &want))
    }

    /// The vendored id kernel exists for every dtype the two checkpoints decode
    /// through the gather: q4_K/q6_K (official experts + lm_head) and q5_K/q8_0
    /// (unsloth UD experts + lm_head). q5_K and q8_0 exercise the two other ggml
    /// geometries — q5_K's N_R0=1 K-quant fan-out and q8_0's shmem K-split.
    const VENDORED_DTYPES: &[(GgmlDType, &str)] =
        &[(GgmlDType::Q4K, "Q4K"), (GgmlDType::Q5K, "Q5K"), (GgmlDType::Q6K, "Q6K"), (GgmlDType::Q8_0, "Q8_0")];

    #[test]
    fn vendored_id_shared_and_per_slot() {
        // Both activation-sharing modes, and a larger n_out (100) exercising the
        // ragged final row-block (100 is not a multiple of NR0*NSG = 4) and a k
        // spanning multiple super-blocks (512 = 2*QK_K).
        for (dt, name) in VENDORED_DTYPES {
            let top_k = 4;
            let t = 5;

            let ids = random_ids(t, top_k, 8, 0x11);
            let (rel, max) = run_case_mode(*dt, 8, 100, 512, t, top_k, 1, ids, 0x100, Mode::MvVendored);
            assert!(rel < 1e-3, "{name} vendored shared rel_l2 {rel} too high (max_abs {max})");

            let ids = random_ids(t, top_k, 8, 0x22);
            let (rel, max) = run_case_mode(*dt, 8, 100, 512, t, top_k, top_k, ids, 0x200, Mode::MvVendored);
            assert!(rel < 1e-3, "{name} vendored per-slot rel_l2 {rel} too high (max_abs {max})");
        }
    }

    /// Every weights-binding encode site must honor `ExpertStack.base_off`
    /// (the mmap alias load binds a page-floored view, so the stack's first
    /// block sits at a nonzero buffer offset). The vendored-kernel and mm_id
    /// sites are covered bitwise by `gguf::tests::expert_stack_mmap_matches_classic`;
    /// this covers the remaining site, the candle-baked classic kernel
    /// (`Mode::Mv`, the `LAGUNA_MV_CLASSIC` fallback): the same stack bytes
    /// uploaded at offset 0 and behind a 32-byte-aligned pad must produce
    /// bitwise-identical results.
    #[test]
    fn base_off_offsets_candle_baked_kernel() {
        use crate::gguf::ExpertStack;
        use candle_core::Device;

        let device = metal_device().unwrap();
        let (n_expert, n_out, k, t, top_k) = (4usize, 8usize, 256usize, 3usize, 2usize);
        let dt = GgmlDType::Q4K;
        let (stack, _) = build_stack(&device, dt, n_expert, n_out, k, 0x0FF5E7).unwrap();

        // The same quantized bytes again, 96 bytes (32-aligned, like a GGUF
        // tensor offset inside its page) into a fresh buffer.
        let base_off = 96usize;
        let bytes = stack.qtensor.as_ref().unwrap().data().unwrap();
        let mut padded = vec![0u8; base_off];
        padded.extend_from_slice(&bytes);
        let Device::Metal(mdev) = &device else { unreachable!() };
        let buf = mdev.new_buffer_with_data(&padded).unwrap();
        let offset_stack = ExpertStack {
            qtensor: None,
            buffer: Some(buf),
            base_off,
            mmap: None,
            dtype: dt,
            n_expert,
            n_out,
            k,
        };

        let x_vec = pseudo_random(t * k, 0x0FF5E7 ^ 0xABCD, -1.0, 1.0);
        let x = Tensor::from_vec(x_vec, (t, 1, k), &device).unwrap();
        let ids = Tensor::from_vec(random_ids(t, top_k, n_expert, 0x44), (t, top_k), &device).unwrap();

        let run = |s: &ExpertStack| -> Vec<f32> {
            dispatch::run(s, &x, &ids, Mode::Mv, crate::ops::MmVariant::ClassicHp)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        let want = run(&stack);
        let got = run(&offset_stack);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "baked mv_id with base_off {base_off} differs at element {i}: {g} vs {w}"
            );
        }
    }

    #[test]
    fn vendored_id_matches_classic() {
        // The vendored kernel and candle's baked kernel are two geometries over
        // the same quantized weights; both must land within f32 noise of the
        // per-row oracle, so they agree with each other to the same bound.
        for (dt, name) in VENDORED_DTYPES {
            let top_k = 3;
            let t = 4;
            let ids = random_ids(t, top_k, 6, 0x33);
            let (rel_v, _) = run_case_mode(*dt, 6, 64, 256, t, top_k, 1, ids.clone(), 0x300, Mode::MvVendored);
            let (rel_c, _) = run_case_mode(*dt, 6, 64, 256, t, top_k, 1, ids, 0x300, Mode::Mv);
            assert!(rel_v < 1e-3, "{name} vendored rel_l2 {rel_v}");
            assert!(rel_c < 1e-3, "{name} classic rel_l2 {rel_c}");
        }
    }

    /// Build a rank-2 `[n_out, k]` quantized weight sharing one allocation between
    /// a retained Metal `Buffer` and a `QMatMul`, exactly as `qlinear_with_buffer`
    /// does at load. Returns (buffer, QMatMul oracle, CPU-dequantized weights).
    fn build_plain_weight(
        device: &candle_core::Device,
        dt: GgmlDType,
        n_out: usize,
        k: usize,
        seed: u64,
    ) -> (std::sync::Arc<Buffer>, candle_core::quantized::QMatMul, Vec<f32>) {
        use candle_core::quantized::{QMatMul, QStorage, QTensor};
        use std::sync::Arc;

        let w = pseudo_random(n_out * k, seed, -1.0, 1.0);
        let w_t = Tensor::from_vec(w, (n_out, k), device).unwrap();
        let qt = QTensor::quantize(&w_t, dt).unwrap();
        let deq = qt.dequantize(&candle_core::Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let storage = QStorage::from_data(qt.data().unwrap(), device, dt).unwrap();
        let buffer = match &storage {
            QStorage::Metal(qms) => Arc::new(qms.buffer().clone()),
            _ => panic!("expected Metal storage"),
        };
        let qtensor = Arc::new(QTensor::new(storage, (n_out, k)).unwrap());
        let qmm = QMatMul::from_arc(qtensor).unwrap();
        (buffer, qmm, deq)
    }

    fn plain_mv_case(dt: GgmlDType, n_out: usize, k: usize, t: usize, seed: u64) -> (f32, f32, f32) {
        let device = metal_device().unwrap();
        let (buffer, qmm, deq) = build_plain_weight(&device, dt, n_out, k, seed);

        let x_vec = pseudo_random(t * k, seed ^ 0x5A5A, -1.0, 1.0);
        let x = Tensor::from_vec(x_vec.clone(), (t, k), &device).unwrap();

        let out = mul_mv(&buffer, dt, n_out, k, &x).unwrap();
        assert_eq!(out.dims(), &[t, n_out]);
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Ground truth: dequantized weights times x, computed per-row on CPU.
        let mut want = vec![0f32; t * n_out];
        for ti in 0..t {
            for o in 0..n_out {
                let mut acc = 0f32;
                for i in 0..k {
                    acc += deq[o * k + i] * x_vec[ti * k + i];
                }
                want[ti * n_out + o] = acc;
            }
        }

        // Candle's baked QMatMul over the same weights is the fallback path; it
        // must also track the oracle (sanity that the two share the allocation).
        let qmm_out = qmm.forward(&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

        (rel_l2(&got, &want), max_abs(&got, &want), rel_l2(&qmm_out, &want))
    }

    #[test]
    fn plain_mv_q6k_basic() {
        // n_out a multiple of NR0*NSG=4, single row (the lm_head decode shape).
        let (rel, max, rel_qmm) = plain_mv_case(GgmlDType::Q6K, 256, 512, 1, 0x700);
        assert!(rel < 1e-3, "vendored plain mv rel_l2 {rel} too high (max_abs {max})");
        assert!(rel_qmm < 1e-3, "QMatMul oracle sanity rel_l2 {rel_qmm}");
    }

    #[test]
    fn plain_mv_q6k_ragged_rows() {
        // n_out NOT a multiple of NR0*NSG=4: the final row-block is partial, so
        // the kernel's `first_row + row < ne0` bound guard is exercised. 100352
        // is the production vocab; 3072 the hidden dim (both q6_K in the model).
        let (rel, max, rel_qmm) = plain_mv_case(GgmlDType::Q6K, 100352, 3072, 1, 0x800);
        assert!(rel < 1e-3, "vendored vocab-sized mv rel_l2 {rel} too high (max_abs {max})");
        assert!(rel_qmm < 1e-3, "QMatMul oracle sanity rel_l2 {rel_qmm}");

        // A deliberately awkward non-multiple-of-4 row count with a small k.
        let (rel, max, _) = plain_mv_case(GgmlDType::Q6K, 30, 256, 2, 0x801);
        assert!(rel < 1e-3, "vendored ragged mv rel_l2 {rel} too high (max_abs {max})");
    }

    #[test]
    fn plain_mv_q8_0_lmhead_and_ragged() {
        // The UD checkpoint's lm_head is q8_0: vocab-sized rows at seq==1, the
        // shmem K-split geometry (N_R0=2, N_SG=4). 100352 is the production vocab,
        // 3072 the hidden dim.
        let (rel, max, rel_qmm) = plain_mv_case(GgmlDType::Q8_0, 100352, 3072, 1, 0x820);
        assert!(rel < 1e-3, "vendored q8_0 vocab mv rel_l2 {rel} too high (max_abs {max})");
        assert!(rel_qmm < 1e-3, "QMatMul oracle sanity rel_l2 {rel_qmm}");

        // Odd row count (not a multiple of N_R0=2) exercises the helper's
        // `r0 + row < ne01` ragged store guard.
        let (rel, max, _) = plain_mv_case(GgmlDType::Q8_0, 31, 256, 2, 0x821);
        assert!(rel < 1e-3, "vendored q8_0 ragged mv rel_l2 {rel} too high (max_abs {max})");
    }

    #[test]
    fn plain_mv_q5k_basic_and_ragged() {
        // q5_K's N_R0=1 fan-out: a K-quant row count that is not a multiple of
        // N_R0*N_SG=2, plus a k spanning multiple super-blocks (512 = 2*QK_K).
        let (rel, max, rel_qmm) = plain_mv_case(GgmlDType::Q5K, 256, 512, 1, 0x830);
        assert!(rel < 1e-3, "vendored q5_K mv rel_l2 {rel} too high (max_abs {max})");
        assert!(rel_qmm < 1e-3, "QMatMul oracle sanity rel_l2 {rel_qmm}");

        let (rel, max, _) = plain_mv_case(GgmlDType::Q5K, 33, 256, 2, 0x831);
        assert!(rel < 1e-3, "vendored q5_K ragged mv rel_l2 {rel} too high (max_abs {max})");
    }

    /// Isolated lm_head-scale timing: vendored plain mv vs candle QMatMul on a
    /// q6_K `[100352, 3072]` weight at seq==1. Ignored (perf, not correctness);
    /// run with `cargo test --release --lib plain_mv_lmhead_bench -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf microbench"]
    fn plain_mv_lmhead_bench() {
        use candle_core::Module;
        use std::time::Instant;
        let device = metal_device().unwrap();
        let (n_out, k) = (100352usize, 3072usize);
        let (buffer, qmm, _) = build_plain_weight(&device, GgmlDType::Q6K, n_out, k, 0xBEEF);
        let x = Tensor::from_vec(pseudo_random(k, 0x1234, -1.0, 1.0), (1, k), &device).unwrap();

        let iters = 200;
        // Warm up both paths (first dispatch folds in pipeline compile / upload).
        for _ in 0..10 {
            mul_mv(&buffer, GgmlDType::Q6K, n_out, k, &x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            qmm.forward(&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = mul_mv(&buffer, GgmlDType::Q6K, n_out, k, &x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        }
        let vendored_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let t1 = Instant::now();
        for _ in 0..iters {
            let _ = qmm.forward(&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        }
        let qmm_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!(
            "lm_head q6_K [{n_out}x{k}] mv @seq1: vendored {vendored_ms:.3} ms/call, QMatMul {qmm_ms:.3} ms/call (incl. readback)"
        );
    }

    #[test]
    fn plain_mv_unsupported_dtype_errors() {
        // q4_K/q5_K/q6_K/q8_0 are vendored; an unsupported K-quant (q2_K) must
        // error rather than fault.
        let device = metal_device().unwrap();
        let (buffer, _, _) = build_plain_weight(&device, GgmlDType::Q2K, 8, 256, 0x900);
        let x = Tensor::from_vec(vec![0f32; 256], (1, 256), &device).unwrap();
        assert!(mul_mv(&buffer, GgmlDType::Q2K, 8, 256, &x).is_err());
    }
}
