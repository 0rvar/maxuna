// Vendored q8_0-weight x f32-activation mat-vec kernel for the attention
// projections of a q8_0-quantized checkpoint (the unsloth UD-Q4_K_XL file stores
// ALL attention weights q8_0). Ported from the llama.cpp laguna fork's
// kernel_mul_mv_q8_0_f32 (ggml/src/ggml-metal/ggml-metal.metal): q8_0 weights x
// f32 activations, f32 products, f32 accumulate, f32 out — the fork's exact
// mul_mat precision (the stored q8_0 weights are the only quantized values in
// the chain, no activation or output rounding), at ~half the decode bandwidth of
// streaming the dequantized f16 dense plane (the f16.metal fallback the prefill/
// mm path keeps using).
//
// This is the ATTENTION decode gemv ONLY (seq < 8); prefill (seq >= 8) runs the
// f16 dense plane through f16.metal's tiled gemm, so no gemm lives here. It is a
// mirror of f16.metal's kernel_mul_mv_f16_f32_v — the same mixed-dtype
// convention, only the weight dtype (q8_0 vs f16) and its dequant differ.
//
// Deliberately a SEPARATE library from mv.metal — which carries the
// byte-identical MoE-decode q8_0 GATHER kernel (kernel_mul_mv_id_q8_0_f32_v) —
// exactly as f16.metal is kept apart from mv.metal: attention-critical vs
// MoE-decode-critical, so neither library can break the other. No Metal-4
// dependency. Compiled at runtime by src/ops/pipelines.rs.

#include <metal_stdlib>

using namespace metal;

// Pin the library math-mode axis to the value nil compile options resolve to
// today (and that candle's kernels are explicitly compiled with), so a future
// OS default change cannot silently alter this library's codegen. clang
// hard-errors on bad `METAL fp` options, so compiling proves it is honored.
#pragma METAL fp math_mode(fast)

#define QK8_0 32
#define N_SIMDWIDTH 32

// N_R0_Q8_0 = 2, N_SG_Q8_0 = 4 (ggml-metal-impl.h). The fork carries these as
// function constants; we hardcode them (M5-only) so the kernel needs no
// specialization. The N_SG simdgroups split the K reduction over the SAME N_R0
// rows and combine through threadgroup memory (helper below); the host reserves
// N_R0*N_SIMDWIDTH floats (dispatch.rs run_matmul_q8).
#define MV_NR0 2
#define MV_NSG 4

// block_q8_0 layout (ggml-common.h): one f16 delta then 32 int8 quants.
typedef struct {
    half   d;          // delta
    int8_t qs[QK8_0];  // quants
} block_q8_0;

// ---- Argument struct --------------------------------------------------------
// Byte-for-byte the fork's ggml_metal_kargs_mul_mv (ggml-metal-impl.h); the host
// writes the identical layout (dispatch.rs MvArgs).

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

// Verbatim from ggml-metal.metal helper_mv_reduce_and_write<NR0>: per-row
// simd_sum, cross-simdgroup combine via shmem (NW floats per row), single
// writer, ragged-tail row guard on the store.
template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

// kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0> with the broadcast function constants
// resolved (ne12 == 1, r2 == r3 == 1 -> i12 == i13 == 0). Grid: (ceil(ne01/NR0),
// ne11, 1); threads (32, MV_NSG, 1); each simdgroup covers a disjoint slice of
// the K reduction for the same NR0 rows. Weights are int8 quants scaled by the
// block's f16 delta and widened to float in the dot — f32 products, f32 accum.
kernel void kernel_mul_mv_q8_0_f32_attn(
        constant mv_args   & args  [[buffer(0)]],
        device const char * src0   [[buffer(1)]],
        device const char * src1   [[buffer(2)]],
        device       char * dst    [[buffer(3)]],
        threadgroup  char * shmem  [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = MV_NR0;
    constexpr short NSG = MV_NSG;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00/QK8_0;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint64_t offset1 = r1*args.nb11;

    device const float * y = (device const float *) (src1 + offset1);

    // pointers to src0 rows
    device const block_q8_0 * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01;

        ax[row] = (device const block_q8_0 *) ((device const char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);

    const int ib0 = sgitg*NQ + ix;

    float yl[NQ];

    device const float * yb = y + ib0*QK8_0 + il*NQ;

    // each thread in a SIMD group deals with NQ quants at a time
    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;

            float sumq = 0.f;
            for (short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += NSG*NQ*QK8_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}
