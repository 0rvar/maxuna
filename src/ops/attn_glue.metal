// Vendored fused attention-glue kernels: the softplus output gate and the
// permute(+cast) copy family of AttnBlock::forward (src/attention.rs). Each
// kernel replaces a chain of candle elementwise/copy dispatches with ONE pass
// and is BIT-IDENTICAL to the candle chain it replaces (compared bitwise in the
// attn_glue.rs ops tests), so the fused path is safe under every parity tier.
// The kill-switch back to the candle chains is LAGUNA_ATTN_GLUE_CLASSIC.
//
// kernel_attn_gate replaces the 10-dispatch chain
//   softplus(gate_logits) = relu + ln(1 + exp(-|x|))   (abs, add, affine, neg,
//   exp, affine, log, add) → transpose/reshape copy → broadcast_mul
// with one pass over attn. The identity rests on reproducing candle's per-op
// ROUNDING BOUNDARIES: every candle op in that chain stores its f32 result to
// memory before the next reads it, so each step here is one correctly-rounded
// f32 operation on the previous step's rounded value:
//   ax   = abs(g)                     (exact)
//   sum  = g + ax                     (one f32 add — candle's badd `x + y`)
//   relu = fma(sum, 0.5, 0.0)         (candle's affine kernel is an EXPLICIT
//                                      fma(float(x), mul, add))
//   e    = exp(-ax)                   (negation exact; exp is the same plain
//                                      metal::exp candle's uexp compiles, both
//                                      libraries built with default-fast math)
//   t    = fma(e, 1.0, 1.0)           (candle's affine again)
//   tail = log(t)                     (candle's ulog, plain metal::log)
//   sp   = relu + tail                (one f32 add)
//   out  = attn * sp                  (one f32 mul — candle's bmul; the
//                                      transpose/reshape between sp and the mul
//                                      is a pure copy, folded into indexing)
// sp is recomputed per output element instead of materialized per (s, h); the
// recomputation is deterministic, so every element of a head-row sees the same
// bits candle's materialized gate tensor held.
//
// The permute family (kernel_permute_cast_*) performs transpose(0,1) of a
// rank-3 tensor plus an optional dtype conversion in one pass. The f32->f32
// variant is a pure copy (bit-identity is structural); the converting variants
// use static_cast, exactly candle's cast kernel scalar (float->half is
// round-to-nearest-even; half->float is exact). A no-permute cast is the same
// kernel with d0 == 1.
//
// FP contraction/reassociation are disabled at file scope: candle's own chain
// ops are single-rounding (a lone add/mul/abs/exp/log, or an explicit fma), so
// there is nothing fast math could legally fuse on candle's side; our fused
// gate body has ADJACENT ops that fast math would otherwise contract or regroup
// across the rounding boundaries listed above (e.g. attn * (relu + tail) must
// stay two roundings). clang REJECTS unknown `fp` pragma options, so these
// compiling proves they are honored. The explicit fma() calls are intrinsics
// and unaffected by contract(off) — they reproduce candle's affine exactly.
// exp/log intrinsic SELECTION is a library math-mode property, not governed by
// the clang fp pragmas. candle compiles its kernels with EXPLICIT
// MTLMathMode::Fast + MTLMathFloatingPointFunctions::Fast (its kernel.rs
// get_compile_options — private, unreachable at the pinned rev); our library
// compiles with nil options, whose documented default resolves to the same
// Fast/Fast. `#pragma METAL fp math_mode(fast)` below pins OUR side of the
// math-mode axis at the source level, so a future OS changing the nil-options
// default cannot silently move this library off candle's mode (clang
// hard-errors on bad options in the `METAL fp` namespace, so a successful
// compile proves the pragma is honored). The FP32-functions axis has no
// pragma; it rests on the documented default (per-call `metal::fast::`
// namespacing exists if it ever needs pinning), guarded by the bitwise tests
// against the live candle chain.
//
// A SEPARATE library from the other vendored sources (own runtime compile via
// src/ops/pipelines.rs, no Metal-4 dependency). The rope kernel is NOT here:
// it must compile WITHOUT these pragmas (see rope.metal).

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp math_mode(fast)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

// Matches dispatch.rs AttnGateArgs (#[repr(C)]).
typedef struct {
    int32_t n_head;
    int32_t seq;
    int32_t head_dim;
} attn_gate_args;

// dst[h, s, d] = attn[h, s, d] * softplus_chain(gate[s, h]).
// attn/dst are [n_head, seq, head_dim] f32 contiguous; gate is [seq, n_head]
// f32 contiguous (the g_proj output layout). One thread per output element.
kernel void kernel_attn_gate(
        constant attn_gate_args & args [[buffer(0)]],
        device const float * attn      [[buffer(1)]],
        device const float * gate      [[buffer(2)]],
        device       float * dst       [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    const int n = args.n_head * args.seq * args.head_dim;
    if ((int) tid >= n) {
        return;
    }
    const int hs = (int) tid / args.head_dim; // h * seq + s
    const int s = hs % args.seq;
    const int h = hs / args.seq;
    const float g = gate[s * args.n_head + h];

    // The candle chain, one rounding boundary per step (see file header).
    const float ax = abs(g);
    const float sum = g + ax;
    const float relu = fma(sum, 0.5f, 0.0f);
    const float e = exp(-ax);
    const float t = fma(e, 1.0f, 1.0f);
    const float tail = log(t);
    const float sp = relu + tail;
    dst[tid] = attn[tid] * sp;
}

// Matches dispatch.rs PermuteArgs (#[repr(C)]). Source is [d0, d1, d2];
// destination is [d1, d0, d2] (transpose of the first two dims). d0 == 1
// degenerates to a plain (optionally casting) copy.
typedef struct {
    int32_t d0;
    int32_t d1;
    int32_t d2;
} permute_args;

#define PERMUTE_CAST(NAME, SRC_T, DST_T)                              \
kernel void NAME(                                                     \
        constant permute_args & args [[buffer(0)]],                   \
        device const SRC_T * src     [[buffer(1)]],                   \
        device       DST_T * dst     [[buffer(2)]],                   \
        uint tid [[thread_position_in_grid]]) {                       \
    const int n = args.d0 * args.d1 * args.d2;                        \
    if ((int) tid >= n) {                                             \
        return;                                                       \
    }                                                                 \
    /* Decompose tid over the dst layout [d1, d0, d2]. */             \
    const int k = (int) tid % args.d2;                                \
    const int r = (int) tid / args.d2;                                \
    const int i = r % args.d0;                                        \
    const int j = r / args.d0;                                        \
    dst[tid] = static_cast<DST_T>(src[(i * args.d1 + j) * args.d2 + k]); \
}

PERMUTE_CAST(kernel_permute_cast_f32_f32, float, float)
PERMUTE_CAST(kernel_permute_cast_f32_f16, float, half)
PERMUTE_CAST(kernel_permute_cast_f16_f32, half, float)
