// MIXED-OPERAND probe variant of f16_t.metal's cooperative-tensor attention
// prefill gemm: the weight tile stays half (the stored dtype — zero extra
// rounding) but the activation tile stays FLOAT instead of being staged to
// half. MetalPerformancePrimitives documents `float(left) x half(right) ->
// float` as a supported matmul2d combination (MPPTensorOpsMatMul2d.h's dtype
// table), which is exactly this kernel's operand order: left = activations
// (float), right = weights (half), f32 accumulate. If it holds most of the
// tensor path's speed it removes the f16 activation rounding that kept
// f16_t.metal opt-in (see that file's header) — the only remaining drift vs
// the classic simdgroup kernel would be tile accumulation order.
//
// TEST-ONLY REACHABILITY: dispatched exclusively through
// dispatch.rs::run_matmul_f16_variant's `TensorMixed` arm, which no production
// selection ever passes. Its own library (pipelines.rs::f16_t_mixed_pipeline),
// compiled lazily on first dispatch — the mm_id_t_hp isolation pattern — so a
// toolchain that rejects mixed-operand matmul2d fails only this probe, never
// the default (or the half-tile tensor) library.
//
// Byte-for-byte f16_t.metal except: sb is a float tile (the staging store is a
// straight float2x4 copy, no half conversion), tB is a float tensor, and the
// destination cooperative tensor is requested with the REAL left/right operand
// order <decltype(tB), decltype(tA)> — f16_t.metal passes <tA, tB>, harmless
// there because both tiles share one type, but with distinct element types the
// order selects the dtype combination. Threadgroup memory stays 8192 B:
// sa 64x32 half = 4096, sb 32x32 float = 4096, and the store-back float tile
// reuses the full region (64x32 float = 8192).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// Pin the library math-mode axis to the value nil compile options resolve to
// today (and that candle's kernels are explicitly compiled with), so a future
// OS default change cannot silently alter this library's codegen. clang
// hard-errors on bad `METAL fp` options, so compiling proves it is honored.
#pragma METAL fp math_mode(fast)

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Byte-for-byte the fork's ggml_metal_kargs_mul_mm (== f16.metal's mm_args and
// dispatch.rs's MmArgs); the host writes the identical layout.
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

// Dense f16-weight gemm, mixed cooperative-tensor operands: half weight tile x
// FLOAT activation tile, f32 accumulate, f32 output. Grid: (ceil(ne1/NR1),
// ceil(ne0/NR0), 1); 128 threads (4 simdgroups); 8192 B threadgroup memory.
kernel void kernel_mul_mm_f16_f32_t_mixed(
        constant mm_args   & args [[buffer(0)]],
        device const char * src0  [[buffer(1)]],
        device const char * src1  [[buffer(2)]],
        device       char * dst   [[buffer(3)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup half  * sa = (threadgroup half  *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + sizeof(half) * 64 * 32);
    threadgroup float * sc = (threadgroup float *)(shmem);

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

    // Dense f16 weight (nl == 1): il stays il0, x advances 2 half4x4 (= NK
    // halves) per K step (f16.metal's dense addressing).
    device const half4x4 * x = (device const half4x4 *)(src0 + args.nb01*(r0 + lr0)) + il0;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    auto tA = tensor<threadgroup half,  dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK,  NR0));
    auto tB = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK ));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NR1, NR0, NK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    // run(sB, sA, cT): left = float activations, right = half weights, so the
    // destination is requested for <left = tB, right = tA> in that order.
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // widen f16 weight + store to threadgroup memory, row-major (NK-strided).
        {
            half4x4 temp_a = half4x4(*x);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = temp_a[i/4][i%4];
            }
        }

        // stage the f32 activation UNROUNDED (the whole point of this probe:
        // no half conversion — a straight float copy).
        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            *(threadgroup float2x4 *)(sb + NK*(8*sy + ly) + 8*sx) = *((device const float2x4 *) y);
        }

        x += 2;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);

        mm.run(sB, sA, cT);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tC);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Dense store-back: sc holds the tile as [token][out] row-major
    // (sc[j*NR0 + i]); each simdgroup drains a token column to its dst row.
    for (short j = sgitg; j < nr1; j += 4) {
        device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i);
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i);
        }
    }
}
