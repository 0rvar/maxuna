// Vendored fused MoE weighted-combine kernels — the routed-expert combine tail
// of FusedExperts::forward (src/moe.rs). They replace the 3-4 strided candle
// broadcast/affine/sum passes over down=[seq, top_k, n_out] with ONE pass that
// reads `down` a single time, and are BIT-IDENTICAL to the candle chain by
// construction (compared bitwise in the combine.rs ops test), so the fused path
// is safe under every parity tier including strict.
//
// The identity rests on reproducing candle's `sum(1)` launch geometry and
// instruction structure EXACTLY. candle dispatches `fast_sum_f32_strided` with
// out_length = seq*n_out threadgroups, each `block_dim` threads wide where
// block_dim = min(pipeline_max, next_power_of_two(top_k/2)); the reduced axis is
// top_k. Its per-thread loader accumulates `value = value + src[strided_idx(i)]`
// for i = tid, tid+block_dim, ... < top_k (candle reduce.metal loader), then the
// BLOCKSIZE<=32 path reduces the lane partials with the hardware `simd_sum`, and
// lane 0 stores. We mirror that: the same launch geometry, the same ascending
// per-lane loader partition, the same hardware `simd_sum`, lane-0 store — but we
// compute each loaded element in-register with candle's per-op rounding
// boundaries instead of reading a pre-materialized product:
//
// SINGLE-SIMDGROUP CONSTRAINT: these kernels implement ONLY the block_dim <= 32
// path — a lone `simd_sum` folds exactly one 32-lane simdgroup, so a wider
// threadgroup would leave lanes 32.. in a second simdgroup whose partials are
// never added (silent lane drop). candle's >32 threadgroup-reduction tree is
// deliberately NOT reproduced (production top_k is 10, so candle's width
// next_pow2(top_k/2) is 8). `run_combine` (combine.rs dispatch) refuses
// host-side any top_k whose candle width would exceed 32, so a reduction wider
// than one simdgroup can never reach these kernels.
//   rescale-free : r3 = down * w                                (one f32 mul)
//   rescale      : r1 = down * col_l2 (f32 mul), r2 = r1 * 2^-15 (exact,
//                  power of two), r3 = r2 * w (f32 mul)
// then `value = value + r3` per element. The three scalars are NOT folded into
// one multiplier (that would reorder rounding), and no fma is used anywhere.
//
// FP contraction is disabled at file scope: candle's own combine kernels are
// single-op (a lone mul or a lone add — nothing to contract), so their per-op
// results are correctly-rounded f32; our fused loop has an adjacent mul and add
// that fast-math would otherwise contract into one fma (one rounding instead of
// two), breaking the identity. `#pragma clang fp contract(off)` pins the two
// roundings regardless of the library's math mode, and `#pragma clang fp
// reassociate(off)` pins the r1→r2→r3 multiply chain's written order (fast math
// would otherwise be licensed to regroup it, e.g. hoisting col_l2·2^-15·w into
// one scalar — different rounding). clang REJECTS unknown `fp` pragma options,
// so these compiling proves they are honored; together they close every
// fast-math transform that could touch a bit (the per-lane partition is at most
// two elements wide, so the add chain has no reassociation freedom either).
//
// A SEPARATE library from mm_id.metal / mv.metal / f16.metal (own runtime
// compile via src/ops/pipelines.rs, no Metal-4 dependency).

#include <metal_stdlib>

using namespace metal;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

// Matches dispatch.rs CombineArgs (#[repr(C)]).
typedef struct {
    int32_t top_k;
    int32_t n_out;
} combine_args;

// Rescale-free combine: dst[s, c] = Σ_k down[s, k, c] * w[s, k].
kernel void kernel_moe_combine(
        constant combine_args & args [[buffer(0)]],
        device const float * down    [[buffer(1)]],
        device const float * w       [[buffer(2)]],
        device       float * dst     [[buffer(3)]],
        uint tid       [[thread_index_in_threadgroup]],
        uint dst_id    [[threadgroup_position_in_grid]],
        uint block_dim [[threads_per_threadgroup]]) {
    const int top_k = args.top_k;
    const int n_out = args.n_out;
    const int did = (int) dst_id;
    const int s = did / n_out;
    const int c = did % n_out;
    const int down_base = s * top_k * n_out + c;
    const int sk_base   = s * top_k;

    float value = 0.0f;
    for (int k = (int) tid; k < top_k; k += (int) block_dim) {
        float d  = down[down_base + k * n_out];
        float ww = w[sk_base + k];
        float r3 = d * ww;
        value = value + r3;
    }
    value = simd_sum(value);
    if (tid == 0) {
        dst[did] = value;
    }
}

// Rescale combine: dst[s, c] = Σ_k (down[s, k, c] * col_l2[s, k]) * 2^-15 * w[s, k],
// with each multiply rounded separately (2^-15 is exact) — the per-column L2
// rescale the f16-tile down projection needs, undone here per candle's chain.
kernel void kernel_moe_combine_rescale(
        constant combine_args & args [[buffer(0)]],
        device const float * down    [[buffer(1)]],
        device const float * col_l2  [[buffer(2)]],
        device const float * w       [[buffer(3)]],
        device       float * dst     [[buffer(4)]],
        uint tid       [[thread_index_in_threadgroup]],
        uint dst_id    [[threadgroup_position_in_grid]],
        uint block_dim [[threads_per_threadgroup]]) {
    const int top_k = args.top_k;
    const int n_out = args.n_out;
    const int did = (int) dst_id;
    const int s = did / n_out;
    const int c = did % n_out;
    const int down_base = s * top_k * n_out + c;
    const int sk_base   = s * top_k;

    float value = 0.0f;
    for (int k = (int) tid; k < top_k; k += (int) block_dim) {
        float d  = down[down_base + k * n_out];
        float l  = col_l2[sk_base + k];
        float ww = w[sk_base + k];
        float r1 = d * l;
        float r2 = r1 * 0x1p-15f;
        float r3 = r2 * ww;
        value = value + r3;
    }
    value = simd_sum(value);
    if (tid == 0) {
        dst[did] = value;
    }
}
