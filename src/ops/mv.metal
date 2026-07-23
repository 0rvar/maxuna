// Vendored quantized mat-vec (MoE decode + lm_head) kernels, ported from the
// llama.cpp laguna fork (ggml/src/ggml-metal/ggml-metal.metal). candle's baked
// kernel_mul_mv_{id_,}q4_K/q6_K kernels are an OLDER geometry (one row per
// simdgroup, no sgitg row fan-out) that runs ~15x under memory bandwidth on this
// device; ggml's current impls assign `(r0*NSG + sgitg)*nr0` rows per simdgroup
// and accumulate `nr0` register rows at once. We vendor the current bodies for
// the only two dtypes the decode path touches: the q4_K/q6_K routed experts and
// the q6_K lm_head.
//
// Deliberately a SEPARATE library from mm_id.metal so this decode-critical file
// carries no Metal-4 `<metal_tensor>` dependency (mm_id.metal requires it) and no
// experimental tile variant can break it. Only q4_K and q6_K are instantiated.
//
// ggml's function-constant layer (FC_mul_mv_nsg / _ne12 / _r2 / _r3) is dropped:
// nr0 and NSG are hardcoded constexpr per the fork's ggml-metal-impl.h constants
// (both dtypes: N_R0=2, N_SG=2), and the broadcast dims are 1 (ne12/r2/r3 == 1)
// for our single-matrix / one-expert-per-slot usage.
//
// Compiled at runtime by src/ops/pipelines.rs via candle's
// new_library_with_source.

#include <metal_stdlib>

using namespace metal;

// Pin the library math-mode axis to the value nil compile options resolve to
// today (and that candle's kernels are explicitly compiled with), so a future
// OS default change cannot silently alter this library's codegen. clang
// hard-errors on bad `METAL fp` options, so compiling proves it is honored.
#pragma METAL fp math_mode(fast)

#define QK_K 256

// N_R0_Q4_K / N_R0_Q6_K = 2, N_SG_Q4_K / N_SG_Q6_K = 2 (ggml-metal-impl.h). The
// fork carries these as function constants; we hardcode them (M5-only, single
// dtype set) so the kernels need no specialization.
#define NR0 2
#define NSG 2

// ---- Quantized block layouts (from ggml-common.h; unions there don't change
// the byte layout) -----------------------------------------------------------

#define K_SCALE_SIZE 12

typedef struct {
    half    d;                    // super-block scale for quantized scales
    half    dmin;                 // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4-bit quants
} block_q4_K;

typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    half    d;               // super-block scale
} block_q6_K;

// ---- Argument structs -------------------------------------------------------
// Byte-for-byte the fork's ggml_metal_kargs_mul_mv / _mul_mv_id (ggml-metal-impl.h);
// the host writes the identical layout (dispatch.rs MvArgs / MvIdArgs).

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
} mv_args;

typedef struct {
    int32_t  nei0;   // n_expert_used (top_k)
    int32_t  nei1;   // n_tokens
    uint64_t nbi1;   // ids row stride (bytes)
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    int32_t  nr0;
} mv_id_args;

// ---- q4_K impl (verbatim from ggml-metal.metal kernel_mul_mv_q4_K_f32_impl,
// with the function-constant broadcast dims resolved to 1) --------------------

template<typename args_t>
static inline void mul_mv_q4_K_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg/8;  // 0...3
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    // ne12 == 1, r2 == 1, r3 == 1: i12 == i13 == 0, so nb02/nb03/nb12/nb13 drop.
    const uint64_t offset0 = first_row*args.nb01;
    const uint64_t offset1 =        r1*args.nb11;

    device const block_q4_K * x = (device const block_q4_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[16];
    float yh[16];

    float sumf[NR0]={0.f};

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < NR0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01/2;
            sc += args.nb01/2;
            dh += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

    // Ragged tail (ne0 % (NR0*NSG) != 0): only the STORE is row-guarded, matching
    // ggml. The accumulation loop above reads up to NR0*NSG-1 weight rows past
    // ne0; those reads land in the next stacked expert's rows or the buffer's
    // page-rounded padding and their sums are discarded here. Benign by design —
    // do not "fix" by guarding the compute loop (it would diverge from the fork).
    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

// ---- q6_K impl (verbatim from ggml-metal.metal kernel_mul_mv_q6_K_f32_impl) --

template<typename args_t>
static inline void mul_mv_q6_K_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint64_t offset0 = first_row*args.nb01;
    const uint64_t offset1 =        r1*args.nb11;

    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[NR0] = { 0.f };

    float yl[16];

    const short tid = tiisg/2;
    const short ix  = tiisg%2;
    const short ip  = tid/8;         // 0 or 1
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;
        device const half    * dh = &x[i].d;

        device const float * y = yy + i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        for (short row = 0; row < NR0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};

            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

            q1 += args.nb01;
            q2 += args.nb01;
            qh += args.nb01;
            sc += args.nb01;
            dh += args.nb01/2;
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    // Same ragged-tail store-only guard as the q4_K impl above (see comment there).
    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

// ---- Plain entry points (lm_head, seq==1) -----------------------------------

kernel void kernel_mul_mv_q4_K_f32_v(
        constant mv_args   & args [[buffer(0)]],
        device const char * src0  [[buffer(1)]],
        device const char * src1  [[buffer(2)]],
        device       char * dst   [[buffer(3)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_mv_q4_K_impl(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q6_K_f32_v(
        constant mv_args   & args [[buffer(0)]],
        device const char * src0  [[buffer(1)]],
        device const char * src1  [[buffer(2)]],
        device       char * dst   [[buffer(3)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_mv_q6_K_impl(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

// ---- Indexed entry points (routed expert gather) ----------------------------
// Ports ggml's kernel_mul_mv_id wrapper (ggml-metal.metal:10820): tgpig.z
// enumerates (token, slot); decode it to the expert id via the ids buffer,
// offset src0/src1/dst, then call the same per-quant impl with a synthetic
// single-matrix arg (ne02/ne11/ne12 == 1). The wrapper is a macro rather than a
// function-pointer template (Metal's fragile support for those) — it expands the
// id-decode + synthetic-arg setup, then calls the named impl inline.

#define MUL_MV_ID_BODY(IMPL_FN)                                                    \
    const int iid1 = tgpig.z/args.nei0;                                            \
    const int idx  = tgpig.z%args.nei0;                                            \
    tgpig.z = 0;                                                                    \
    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];     \
    const int64_t i11 = idx % args.ne11;                                           \
    const int64_t i12 = iid1;                                                       \
    const int64_t i1 = idx;                                                         \
    const int64_t i2 = i12;                                                         \
    device const char * src0_cur = src0s + i02*args.nb02;                           \
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;           \
    device char * dst_cur = dst + (i1*args.ne0 + i2*args.ne1*args.ne0)*sizeof(float);\
    mv_args args0 = {                                                               \
        /*.ne00 =*/ args.ne00,                                                      \
        /*.ne01 =*/ args.ne01,                                                      \
        /*.ne02 =*/ 1,                                                              \
        /*.nb00 =*/ args.nb00,                                                      \
        /*.nb01 =*/ args.nb01,                                                      \
        /*.nb02 =*/ args.nb02,                                                      \
        /*.nb03 =*/ args.nb02,                                                      \
        /*.ne10 =*/ args.ne10,                                                      \
        /*.ne11 =*/ 1,                                                              \
        /*.ne12 =*/ 1,                                                              \
        /*.nb10 =*/ args.nb10,                                                      \
        /*.nb11 =*/ args.nb11,                                                      \
        /*.nb12 =*/ args.nb12,                                                      \
        /*.nb13 =*/ args.nb12,                                                      \
        /*.ne0  =*/ args.ne0,                                                       \
        /*.ne1  =*/ 1,                                                              \
        /*.nr0  =*/ args.nr0,                                                       \
        /*.r2   =*/ 1,                                                              \
        /*.r3   =*/ 1,                                                              \
    };                                                                             \
    IMPL_FN(args0, src0_cur, src1_cur, dst_cur, tgpig, tiisg, sgitg);

kernel void kernel_mul_mv_id_q4_K_f32_v(
        constant mv_id_args & args  [[buffer(0)]],
        device const char *  src0s  [[buffer(1)]],
        device const char *  src1   [[buffer(2)]],
        device       char *  dst    [[buffer(3)]],
        device const char *  ids    [[buffer(4)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    MUL_MV_ID_BODY(mul_mv_q4_K_impl)
}

kernel void kernel_mul_mv_id_q6_K_f32_v(
        constant mv_id_args & args  [[buffer(0)]],
        device const char *  src0s  [[buffer(1)]],
        device const char *  src1   [[buffer(2)]],
        device       char *  dst    [[buffer(3)]],
        device const char *  ids    [[buffer(4)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    MUL_MV_ID_BODY(mul_mv_q6_K_impl)
}
