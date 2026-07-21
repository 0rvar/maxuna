use anyhow::Result;
use candle_core::Tensor;

use crate::gguf::ExpertStack;
use crate::ops::dispatch::{self, Mode};

/// Quantized gather-matmul over a stacked expert tensor (prefill path):
/// tokens are grouped per expert so each expert's rows are read once per chunk.
/// Dispatches candle's `kernel_mul_mm_id_<dtype>_f32`.
///
/// Shapes as in `mv_id::mul_mv_id`; t is the prefill chunk length.
pub fn mul_mm_id(stack: &ExpertStack, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    dispatch::run(stack, x, ids, Mode::Mm)
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
            let ids = random_ids(t, top_k, 4, 0x41);
            let (rel, max) = run_case(*dt, 4, 8, 256, t, top_k, 1, ids, 0x1600);
            assert!(rel < 1e-3, "{name} t=16 rel_l2 {rel} too high (max_abs {max})");
        }
    }

    #[test]
    fn prefill_t64() {
        for (dt, name) in DTYPES {
            let (top_k, t) = (2, 64);
            let ids = random_ids(t, top_k, 4, 0x42);
            let (rel, max) = run_case(*dt, 4, 8, 256, t, top_k, 1, ids, 0x6400);
            assert!(rel < 1e-3, "{name} t=64 rel_l2 {rel} too high (max_abs {max})");
        }
    }

    #[test]
    fn prefill_per_slot_and_repeats() {
        // Down-projection layout (x_per_row == top_k) plus a token whose slots
        // repeat one expert, so that expert's row-id map gets multiple columns.
        for (dt, name) in DTYPES {
            let (top_k, t) = (4, 16);
            let mut ids = random_ids(t, top_k, 4, 0x43);
            for slot in 0..top_k {
                ids[slot] = 2; // token 0: every slot -> expert 2
            }
            let (rel, max) = run_case(*dt, 4, 8, 256, t, top_k, top_k, ids, 0x1601);
            assert!(rel < 1e-3, "{name} per-slot/repeat rel_l2 {rel} too high (max_abs {max})");
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
}
