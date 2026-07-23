// Vendored NEOX rope kernel with INTERNAL partial rotary — the fused
// replacement for Rope::rotate's candle chain (src/rope.rs): narrow the first
// n_rot dims + contiguous copy, candle's `rope_f32` by-halves kernel, then cat
// the pass-through dims back. This kernel reads the full [heads, seq, head_dim]
// f32 tensor once and writes it once: dims < n_rot are rotated with candle's
// exact math, dims >= n_rot are copied verbatim. BIT-IDENTICAL to that chain
// (compared bitwise in the attn_glue.rs ops tests); the kill-switch back to it
// is LAGUNA_ATTN_GLUE_CLASSIC.
//
// The rotation body is candle's `rope` template (reduce.metal) VERBATIM for a
// pair (i1, i2 = i1 + n_rot/2):
//   dst[i1] = src[i1] * c - src[i2] * s;
//   dst[i2] = src[i1] * s + src[i2] * c;
// One thread handles one pair, writing both halves in the same two adjacent
// statements candle's thread does. NO `#pragma clang fp` contraction/
// reassociation overrides in this file, deliberately: candle compiles its
// kernels with EXPLICIT MTLMathMode::Fast + MTLMathFloatingPointFunctions::Fast
// (its kernel.rs get_compile_options — private, unreachable at the pinned
// rev), under which each mul-sub/mul-add here may contract into an fma; this
// library compiles with nil options, whose documented default resolves to the
// same Fast/Fast, with the SAME expressions, so the compiler makes the same
// choice. `#pragma METAL fp math_mode(fast)` below pins OUR math-mode axis at
// the source level so a future OS changing the nil-options default cannot
// silently move this library off candle's mode (clang hard-errors on bad
// options in the `METAL fp` namespace, so a successful compile proves it is
// honored); the FP32-functions axis has no pragma and rests on the documented
// default (irrelevant here — this kernel uses no math-library functions).
// Pinning contraction off instead (as combine.metal/attn_glue.metal do for
// their chains) would FORCE two-rounding sequences and break bit-identity if
// candle's compile contracted. The attn_glue.rs bitwise test compares against
// the live candle kernel and guards this equivalence on the running toolchain.
//
// cos/sin are the full precomputed [max_ctx, n_rot/2] f32 tables Rope holds
// (YaRN-scaled for full-attention layers, plain for SWA — table CONTENT never
// touches this kernel); `pos` selects the starting row, replacing the host-side
// narrow. n_rot == head_dim (SWA layers) makes the pass-through region empty.
//
// A SEPARATE library from attn_glue.metal because pragmas are file-scoped and
// this file must stay pragma-free (see above). No Metal-4 dependency.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp math_mode(fast)

// Matches dispatch.rs RopeArgs (#[repr(C)]).
typedef struct {
    int32_t heads;
    int32_t seq;
    int32_t head_dim;
    int32_t n_rot;
    int32_t pos;
} rope_args;

// src: [heads, seq, head_dim] f32 contiguous. dst: same shape, OT (float or
// half — the f16 variant computes the rotation in f32 exactly as the f32 one
// and only ROUNDS THE FINAL STORE, one RTNE rounding, so it is bit-identical
// to f32-rope + candle cast_f16; pass-through dims round identically). cos/sin:
// [max_ctx, n_rot/2] f32 contiguous. One thread per (row, dim) element;
// threads with n_rot/2 <= d < n_rot idle (their element is written by the
// paired thread at d - n_rot/2, mirroring candle's one-thread-per-pair
// structure).
template <typename OT>
kernel void kernel_rope_neox(
        constant rope_args & args   [[buffer(0)]],
        device const float * src    [[buffer(1)]],
        device const float * cos_t  [[buffer(2)]],
        device const float * sin_t  [[buffer(3)]],
        device       OT    * dst    [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    const int n = args.heads * args.seq * args.head_dim;
    if ((int) tid >= n) {
        return;
    }
    const int d = (int) tid % args.head_dim;
    const int row = (int) tid / args.head_dim; // h * seq + s
    if (d >= args.n_rot) {
        // Pass-through dims: the candle chain routes them through narrow + cat,
        // pure copies (the f16 variant rounds them like candle's cast scalar).
        dst[tid] = static_cast<OT>(src[tid]);
        return;
    }
    const int hlf = args.n_rot / 2; // (`half` is a Metal type keyword)
    if (d >= hlf) {
        // Written by the paired thread at d - hlf.
        return;
    }
    const int s = row % args.seq;
    const int base = row * args.head_dim;
    const int i1 = base + d;
    const int i2 = i1 + hlf;
    const int i_cs = (args.pos + s) * hlf + d;
    const float c = cos_t[i_cs];
    const float ss = sin_t[i_cs];
    // candle reduce.metal `rope` body, verbatim (see file header); the f32
    // arithmetic is shared by both instantiations, only the store narrows.
    dst[i1] = static_cast<OT>(src[i1] * c - src[i2] * ss);
    dst[i2] = static_cast<OT>(src[i1] * ss + src[i2] * c);
}

// Per-instantiation typedefs: the output type is part of the kernel signature,
// so (unlike ggml's char*-typed operands) one shared typedef cannot cover both.
typedef decltype(kernel_rope_neox<float>) kernel_rope_neox_f32_t;
typedef decltype(kernel_rope_neox<half>) kernel_rope_neox_f16_t;

template [[host_name("kernel_rope_neox_f32")]] kernel kernel_rope_neox_f32_t kernel_rope_neox<float>;
template [[host_name("kernel_rope_neox_f16")]] kernel kernel_rope_neox_f16_t kernel_rope_neox<half>;
