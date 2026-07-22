use anyhow::Result;
use candle_core::Tensor;
use candle_core::quantized::GgmlDType;

use crate::gguf::ExpertStack;
use crate::ops::MmVariant;
use crate::ops::dispatch::{self, Mode};

/// Quantized gather-matmul over a stacked expert tensor (prefill path):
/// tokens are grouped per expert so each expert's rows are read once per chunk.
/// Dispatches the vendored `kernel_mul_mm_id_<dtype>_f32[_hp]`.
///
/// Shapes as in `mv_id::mul_mv_id`; t is the prefill chunk length.
pub fn mul_mm_id(stack: &ExpertStack, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    dispatch::run(stack, x, ids, Mode::Mm, crate::ops::mm_id_variant())
}

/// Whether the vendored two-pass mm_id kernels are instantiated for this dtype,
/// top_k, and the ACTIVE variant. `moe` gates the seq>=MM_ID_MIN_SEQ prefill
/// branch on this and falls back to mv_id when a checkpoint uses an uninstantiated
/// dtype/top_k, or the selected variant lacks a kernel for that dtype (e.g.
/// `_t_hp` on q8_0/q5_K). Variant-aware because `encode_mul_mm_id` appends the
/// variant suffix to the kernel name, and an uninstantiated combo faults the
/// pipeline lookup instead of gracefully falling back.
pub(crate) fn supported(dt: GgmlDType, top_k: usize, variant: MmVariant) -> bool {
    dispatch::mm_kernel_instantiated(dt, variant) && dispatch::map0_kernel_name(top_k).is_ok()
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

    /// The variant-aware support matrix (`dispatch::mm_kernel_instantiated`,
    /// consumed by `supported`) must exactly track the kernels actually
    /// instantiated in mm_id.metal: for every (variant, dtype) it claims
    /// supported there must be a matching `[[host_name("...")]]` line, or `moe`
    /// would route to a kernel that faults at pipeline lookup instead of falling
    /// back to mv_id. Cross-check the matrix against the source of truth.
    /// Strip `/* ... */` block comments and `//` line comments from Metal source
    /// so a commented-out `[[host_name(...)]]` instantiation cannot satisfy a
    /// `contains` check. Byte-wise over the ASCII delimiters; retained bytes stay
    /// valid UTF-8 (comment markers never fall inside a multi-byte sequence).
    fn strip_metal_comments(src: &str) -> String {
        let b = src.as_bytes();
        let mut out: Vec<u8> = Vec::with_capacity(b.len());
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
                i += 2;
                while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                    i += 1;
                }
                i = (i + 2).min(b.len());
            } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
                i += 2;
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            } else {
                out.push(b[i]);
                i += 1;
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    /// An instantiation is present iff a `[[host_name("<name>"...` line exists in
    /// the (comment-stripped) source. Match only up to the closing quote, not the
    /// closing `)]]`, because some instantiations pad the token (`"...ne20_1" )`);
    /// the trailing quote already disambiguates prefixes (`..._1"` never matches
    /// `..._10"`).
    fn host_name_present(src: &str, name: &str) -> bool {
        src.contains(&format!("host_name(\"{name}\""))
    }

    #[test]
    fn instantiation_matrix_matches_metal() {
        use crate::ops::MmVariant;
        use crate::ops::dispatch::{map0_kernel_name, mm_kernel_instantiated, mm_kernel_name};

        // The `_t_hp` (TensorHp) float-cooperative-tensor instantiations live in a
        // SEPARATE source (mm_id_t_hp.metal), concatenated onto mm_id.metal only for
        // the lazily-compiled TensorHp library. Every other variant lives in
        // mm_id.metal. Check each variant against the source that actually hosts it.
        const MM_ID_SRC: &str = include_str!("mm_id.metal");
        const T_HP_SRC: &str = include_str!("mm_id_t_hp.metal");
        // Strip comments FIRST: a commented-out instantiation must not count as
        // present, or the matrix could claim a kernel the compiler never emits.
        let src = strip_metal_comments(MM_ID_SRC);
        let t_hp_src = strip_metal_comments(T_HP_SRC);

        // Partition proof: the default library (mm_id.metal) must contain NO
        // float-operand cooperative-tensor (`_t_hp`) instantiation, so a toolchain
        // rejecting float `matmul2d` operands cannot break the default prefill path.
        assert!(
            !src.contains("_t_hp"),
            "mm_id.metal must not host any `_t_hp` instantiation — it belongs in \
             mm_id_t_hp.metal so the default library stays free of float-cooperative-tensor code"
        );
        // And the split-out source must host ONLY `_t_hp` kernels (no default ones).
        for line in t_hp_src.lines() {
            if let Some(rest) = line.split_once("host_name(\"").map(|(_, r)| r) {
                let host = &rest[..rest.find('"').unwrap_or(rest.len())];
                assert!(
                    host.ends_with("_t_hp"),
                    "mm_id_t_hp.metal hosts non-`_t_hp` kernel {host:?}; only the split-out \
                     float-cooperative-tensor instantiations belong there"
                );
            }
        }

        // Per-variant: the source hosting a variant's instantiations.
        let host_src = |variant: MmVariant| -> &str {
            match variant {
                MmVariant::TensorHp => &t_hp_src,
                MmVariant::Tensor | MmVariant::ClassicHp | MmVariant::ClassicF16 => &src,
            }
        };

        const VARIANTS: &[MmVariant] = &[
            MmVariant::Tensor,
            MmVariant::TensorHp,
            MmVariant::ClassicHp,
            MmVariant::ClassicF16,
        ];
        // The four base dtypes the mm path knows, plus dtypes with no mm kernel at
        // all (to assert the matrix denies them under every variant).
        const DTYPES: &[GgmlDType] = &[
            GgmlDType::Q8_0,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q4_0,
            GgmlDType::Q2K,
        ];

        for &variant in VARIANTS {
            for &dt in DTYPES {
                let claimed = mm_kernel_instantiated(dt, variant);
                // The exact host name the encoder would dispatch: base + variant suffix.
                let name = mm_kernel_name(dt).ok().map(|base| format!("{base}{}", variant.suffix()));
                let present = name.as_ref().is_some_and(|n| host_name_present(host_src(variant), n));
                assert_eq!(
                    claimed, present,
                    "support matrix disagrees with mm_id.metal for {dt:?}/{variant:?}: \
                     mm_kernel_instantiated={claimed}, host_name present={present} (name={name:?})"
                );
            }
        }

        // map0 (the top_k dimension `supported()` also gates on): every top_k
        // `map0_kernel_name` accepts must have a matching `map0_ne20_{top_k}`
        // host_name, and every one it rejects must have none — else `moe` would
        // route a top_k with no map0 pass into a faulting pipeline lookup.
        const TOP_KS: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16];
        for &top_k in TOP_KS {
            match map0_kernel_name(top_k) {
                Ok(name) => assert!(
                    host_name_present(&src, &name),
                    "map0_kernel_name claims {name:?} for top_k={top_k}, but no host_name in mm_id.metal"
                ),
                Err(_) => {
                    let ghost = format!("kernel_mul_mm_id_map0_ne20_{top_k}");
                    assert!(
                        !host_name_present(&src, &ghost),
                        "map0_kernel_name denies top_k={top_k}, but mm_id.metal instantiates {ghost:?}"
                    );
                }
            }
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
    /// The store lands `C[r0][r1]` at `sc[r0*NR1 + r1]` (extents (NR0, NR1)); the
    /// test drives it with per-index-constant operands so the output separates as
    /// `C[r0][r1] = NK * a(r0) * b(r1)` — non-uniform in BOTH axes, so a transposed
    /// store (`C[r1][r0]`) no longer matches (all-ones would hide that).
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

        // Non-uniform, layout-sensitive operands. The probe runs ONE 32-wide
        // matmul2d tile with a plain LINEAR tile load (not the production swizzle),
        // so it is not a clean C=W·Xᵀ; empirically (and deterministically) the tile
        // yields, at the stored linear position sc[r1*NR0 + r0]:
        //     value(r0, r1) = NK * A1[r0 / 2] * B1[r1]
        // where A1[k] (k in 0..NK) is A's value along its FIRST extent (held
        // constant down the second) and B1[m] (m in 0..NR1) is B's value along its
        // first extent (held constant down NK). (The `/2` and the `NK` factor are
        // artifacts of one cooperative-tensor tile fed linearly loaded data; the
        // PRODUCTION expert layout is validated against the oracle in
        // `mm_matches_oracle_production_scale`.) Choosing A1 and B1 as distinct,
        // non-constant small-integer patterns makes the output vary in BOTH r0 and
        // r1, so a transposed store (extents (NR1, NR0) → sc[r0*NR1 + r1]) reorders
        // the linear array and fails the exact comparison — which all-ones (every
        // entry == NK) could not detect. Values are exact in f16, products exact in f32.
        const NK: usize = 32;
        const NR0: usize = 64;
        const NR1: usize = 32;
        let a1 = |k: usize| (1 + (k % 4)) as f32; // A first-extent pattern {1,2,3,4}
        let b1 = |m: usize| (1 + (m % 5)) as f32; // B first-extent pattern {1,2,3,4,5}
        // A [NK, NR0]: value depends only on the first extent k = i / NR0.
        let a_vec: Vec<f32> = (0..NK * NR0).map(|i| a1(i / NR0)).collect();
        // B [NR1, NK]: value depends only on the first extent m = i / NK.
        let b_vec: Vec<f32> = (0..NR1 * NK).map(|i| b1(i / NK)).collect();
        let a = Tensor::from_vec(a_vec, (NK, NR0), &device).unwrap().to_dtype(DType::F16).unwrap();
        let b = Tensor::from_vec(b_vec, (NR1, NK), &device).unwrap().to_dtype(DType::F16).unwrap();
        let c = mdev.new_buffer(NR0 * NR1, DType::F32, "probe_c").unwrap();

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

        let out = dispatch::output_tensor(c, mdev, NR0 * NR1, (NR0, NR1));
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "probe output has non-finite values");

        // Exact expected C[r0][r1] = NK * a(r0) * b(r1), laid out row-major over
        // (NR0, NR1). Products/sums are small integers exact in f32, so compare
        // exactly. A transposed store would place a(r1)*b(r0) here and fail.
        // Exact expected, laid out exactly as the kernel stores it: linear index
        // i = r1*NR0 + r0, value = NK * A1[r0/2] * B1[r1]. A transposed store would
        // land a(r1)/b(r0)-shaped values here and fail. Products are small integers
        // exact in f32, so compare exactly (tiny epsilon for the f16→f32 store).
        let mut mismatches = 0usize;
        for r1 in 0..NR1 {
            for r0 in 0..NR0 {
                let want = NK as f32 * a1(r0 / 2) * b1(r1);
                let got = v[r1 * NR0 + r0];
                if (got - want).abs() > 1e-3 {
                    mismatches += 1;
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "tensor-ops probe: {mismatches} of {} entries mismatch the CPU value \
             (NK*A1[r0/2]*B1[r1]); a transposed store would reorder them. C[0..4]={:?}",
            NR0 * NR1,
            &v[0..4]
        );
        eprintln!("tensor-ops probe OK: compiled, ran, and matched NK*A1[r0/2]*B1[r1] exactly");
    }
}
