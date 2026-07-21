use anyhow::Result;
use candle_core::Tensor;

use crate::gguf::ExpertStack;
use crate::ops::dispatch::{self, Mode};

/// Quantized gather-matvec over a stacked expert tensor (decode path).
/// Dispatches candle's compiled-but-unwired `kernel_mul_mv_id_<dtype>_f32`
/// (candle-metal-kernels quantized.metal); geometry mirrors the fork's
/// ggml-metal-ops.cpp mul_mm_id/mv_id setup.
///
/// x: [t, x_per_row, k] f32 — x_per_row is 1 when every selected expert of a
/// token consumes the same activation (gate/up), top_k for the down projection.
/// ids: [t, top_k] u32, on-device.
/// Returns [t, top_k, n_out] f32.
pub fn mul_mv_id(stack: &ExpertStack, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    dispatch::run(stack, x, ids, Mode::Mv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::ops::dispatch::testutil::*;
    use candle_core::quantized::GgmlDType;
    use candle_core::Tensor;

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
}
