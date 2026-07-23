// Vendored fused MoE SwiGLU-activation kernel — the silu*mul glue between the
// up/gate expert matvecs and the down matvec in FusedExperts::forward
// (src/moe.rs). It replaces candle's two elementwise dispatches (a `silu` unary
// pass then a `mul` binary pass over the [seq, top_k, expert_ff] activation)
// with ONE pass that reads `gate`/`up` once and writes `act`, and is
// BIT-IDENTICAL to that candle chain by construction (compared bitwise in the
// silu_mul.rs ops test), so the fused path is safe under every parity tier
// including strict.
//
// The identity rests on reproducing candle's two ops with the SAME per-op
// rounding boundaries:
//   silu : candle's usilu (metal_src/unary.metal) is `x / (1 + exp(-x))` for the
//          f32 unary kernel — copied here VERBATIM (`g / (1 + exp(-g))`). The
//          integer `1` widens to exactly 1.0f, matching candle.
//   mul  : candle's bmul (metal_src/binary.metal) is `x * y` — the silu result
//          times `up`, a separate f32 rounding.
// candle materializes the silu output to an f32 buffer before the mul reads it,
// so silu and mul each round to f32 independently. This kernel keeps `s` in an
// f32 register (already f32-rounded by the division) and then multiplies, which
// is the same two roundings PROVIDED no fast-math transform fuses or reorders
// across the boundary — which the fp pragmas below pin.
//
// FP contraction / reassociation are disabled at file scope: without them,
// fast math could reassociate `(g / D) * up` into e.g. `g * (up / D)` (a
// different rounding), fusing the silu result and the multiply into one
// expression. `#pragma clang fp reassociate(off)` fixes the written expression
// tree — `s = g / D` rounded, then `s * up` rounded — so the boundary matches
// candle's two separate kernels; `#pragma clang fp contract(off)` pins the (here
// vacuous — there is no multiply-add) contraction axis for parity with the
// sibling vendored glue files. clang REJECTS unknown `fp` pragma options, so
// these compiling proves they are honored. `#pragma METAL fp math_mode(fast)`
// pins the library math-mode axis to what nil compile options resolve to today
// (candle's own kernels are compiled with explicit MTLMathMode::Fast), so a
// future OS default change cannot move this library's mode WITHIN which the
// silu expression's own arithmetic (the `exp`, and any fast-math reciprocal
// lowering of the division) stays identical to candle's fast-compiled usilu.
//
// A SEPARATE library from mm_id.metal / mv.metal / f16.metal / combine.metal
// (own runtime compile via src/ops/pipelines.rs, no Metal-4 dependency).

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp math_mode(fast)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

// Matches dispatch.rs SiluMulArgs (#[repr(C)]).
typedef struct {
    int32_t n;
} silu_mul_args;

// act[i] = silu(gate[i]) * up[i], with silu = x / (1 + exp(-x)) (candle's usilu)
// and the multiply each rounded separately (see file header). `gate`, `up` and
// `dst` are contiguous f32 of the same length; one thread per element.
kernel void kernel_moe_silu_mul(
        constant silu_mul_args & args [[buffer(0)]],
        device const float * gate     [[buffer(1)]],
        device const float * up       [[buffer(2)]],
        device       float * dst      [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    // Unsigned compare: at n == i32::MAX (the host-side launch cap) the
    // rounded-up grid emits a stray thread whose tid wraps negative under an
    // (int) cast, slipping past a signed guard into a one-element OOB write.
    if (tid >= (uint) args.n) {
        return;
    }
    const float g = gate[tid];
    const float s = g / (1 + exp(-g));
    dst[tid] = s * up[tid];
}
