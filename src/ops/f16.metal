// Vendored f16-weight x f32-activation matmul kernels for the attention
// projections, ported from the llama.cpp laguna fork
// (ggml/src/ggml-metal/ggml-metal.metal). candle's Metal f16 matmul requires
// same-dtype operands, so its path rounds the f32 activation to f16 AND stores
// the output f16 before the f32 upcast (~2.4e-4 noise each, per matmul). ggml's
// mixed-dtype convention avoids both: the stored f16 weights are the only f16
// in the chain, products/accumulation/output are f32.
//
// Two kernels, mirroring ggml's host-side split on ne11 (token count):
//   kernel_mul_mv_f16_f32_v — the decode gemv, ported from
//     kernel_mul_mv_t_t_4_impl<half, half4, float, float4> (the `_4` vectorized
//     body the fork's host selects whenever ne00 % 4 == 0, which all our K
//     dims satisfy). NR0/NSG are the fork's host choices for our shapes,
//     hardcoded constexpr (M5-only): nr0 = 2 (the only case in ggml's disp
//     switch) and nsg = min(4, ceil(ne00/128)) = 4 (every attention K is
//     >= 3072). Simdgroups split the K reduction; helper_mv_reduce_and_write
//     combines them through threadgroup memory (NR0*32 floats).
//   kernel_mul_mm_f16_f32_v — the prefill gemm, ported from the classic
//     (non-tensor) kernel_mul_mm body with dequantize_f16 / nl = 1. ONE
//     deliberate deviation from the fork's f16_f32 instantiation: the fork
//     stages BOTH tiles as half (its sb store rounds the f32 activations to
//     f16), we stage float tiles — same body, only the tile element type
//     differs, exactly the mm_id.metal `_hp` precedent (no measured throughput
//     cost there; smem 8192 -> 12288 B). This keeps the "weights are the only
//     f16 rounding" contract at prefill too, strictly tighter than the fork.
//     Accumulators are simdgroup_float8x8 in both.
//
// Function-constant specialization is resolved for our usage: single matrix
// (ne02 == ne12 == 1, r2 == r3 == 1, so i12 == i13 == 0), bc_inp false (host
// requires ne00 % 32 == 0; every attention K is a multiple of 1024), bc_out
// true (the guarded threadgroup store-back handles every out-dim/seq shape,
// including the 48/72-row gate projections; the fork itself takes this branch
// whenever seq % 32 != 0, i.e. on virtually every real prefill).
//
// Deliberately a SEPARATE library from mm_id.metal (no Metal-4 <metal_tensor>
// dependency) and from mv.metal (this file is attention-critical; that one is
// MoE-decode-critical — neither can break the other). Compiled at runtime by
// src/ops/pipelines.rs via candle's new_library_with_source.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// Pin the library math-mode axis to the value nil compile options resolve to
// today (and that candle's kernels are explicitly compiled with), so a future
// OS default change cannot silently alter this library's codegen. clang
// hard-errors on bad `METAL fp` options, so compiling proves it is honored.
#pragma METAL fp math_mode(fast)

#define N_SIMDWIDTH 32

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// ---- Argument structs -------------------------------------------------------
// Byte-for-byte the fork's ggml_metal_kargs_mul_mv / _mul_mm (ggml-metal-impl.h);
// the host writes the identical layout (dispatch.rs MvArgs / MmArgs).

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
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} mm_args;

// ---- kernel_mul_mv_f16_f32_v (decode gemv) ----------------------------------
// The fork's host constants for our shapes (see file header).
#define MV_NR0 2
#define MV_NSG 4

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

// kernel_mul_mv_t_t_4_impl<half, half4, float, float4, 2> with the broadcast
// function constants resolved (ne12 == 1, r2 == r3 == 1 -> i12 == i13 == 0).
// Grid: (ceil(ne01/NR0), ne11, 1); threads (32, MV_NSG, 1); each simdgroup
// covers a disjoint slice of the K reduction for the same NR0 rows. Weights are
// read as half4 and widened to float in the dot — f32 products, f32 accum.
kernel void kernel_mul_mv_f16_f32_v(
        constant mv_args   & args [[buffer(0)]],
        device const char * src0  [[buffer(1)]],
        device const char * src1  [[buffer(2)]],
        device       char * dst   [[buffer(3)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = MV_NR0;
    constexpr short NSG = MV_NSG;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;

    const uint64_t offset1 = r1*args.nb11;

    device const float  * y  = (device const float  *) (src1 + offset1);
    device const float4 * y4 = (device const float4 *) (src1 + offset1);

    // pointers to src0 rows
    device const half  * ax [NR0];
    device const half4 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01;

        ax [row] = (device const half  *) ((device const char *) src0 + offset0);
        ax4[row] = (device const half4 *) ((device const char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    float4 yl4[NF4];

    device const float4 * yb4 = y4 + (ib0*NB + il*NF)/4;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const half4 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), yl4[i]);
            }

            sumf[row] += sumq;
        }

        yb4 += NSG*NF*NW/4;
    }

    // K tail (ne00 % 32 != 0) — never taken at our shapes (host requires
    // ne00 % 32 == 0), kept verbatim from the fork.
    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += (float) ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

// ---- kernel_mul_mm_f16_f32_v (prefill gemm) ---------------------------------
// Classic simdgroup kernel_mul_mm with dequantize_f16 (a 16-element copy) and
// nl = 1, float tiles (see file header for the half->float tile deviation).
// Grid: (ceil(ne1/NR1), ceil(ne0/NR0), 1); 128 threads (4 simdgroups). Each
// threadgroup computes a 64(out-row) x 32(token) tile; the output goes through
// the guarded threadgroup store-back (bc_out resolved true).
kernel void kernel_mul_mm_f16_f32_v(
        constant mm_args   & args [[buffer(0)]],
        device const char * src0  [[buffer(1)]],
        device const char * src1  [[buffer(2)]],
        device       char * dst   [[buffer(3)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    // sa holds an NR0(64) x NK(32) weight tile, sb an NR1(32) x NK(32)
    // activation tile, both float; the store-back reuses the region as an
    // NR0 x NR1 float tile (8192 B, within sa's span).
    threadgroup float * sa = (threadgroup float *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + sizeof(float) * 64 * 32);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    // nl == 1 for f16: il stays il0 and x advances 2 half4x4 (= NK halves) per
    // K step (the fork's generic il/x update resolved for nl = 1).
    device const half4x4 * x = (device const half4x4 *)(src0 + args.nb01*(r0 + lr0)) + il0;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    simdgroup_float8x8 ma[4];
    simdgroup_float8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // load data and store to threadgroup memory
        {
            // dequantize_f16: a plain 16-element widen-copy of one half4x4.
            float4x4 temp_a = float4x4(*x);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup float2x4 *)(sb + 64*ib + 8*ly) = *((device const float2x4 *) y);
        }

        x += 2;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const float * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const float * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    // Guarded store-back (the fork's bc_out branch): stage the tile in
    // threadgroup memory, then copy only the in-bounds nr0 x nr1 region.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1) {
            device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0;
            device float4 * D4 = (device float4 *) D;

            threadgroup float  * C  = ((threadgroup float *) shmem) + (j*NR0);
            threadgroup float4 * C4 = (threadgroup float4 *) C;

            int i = 0;
            for (; i < nr0/4; i++) {
                *(D4 + i) = *(C4 + i);
            }

            i *= 4;
            for (; i < nr0; i++) {
                *(D + i) = *(C + i);
            }
        }
    }
}
