// Vendored Metal-4 cooperative-tensor port of the f16-weight x f32-activation
// attention PREFILL gemm — the tensor analogue of f16.metal's classic simdgroup
// kernel_mul_mm_f16_f32_v. OPT-IN via LAGUNA_ATTN_MM_TENSOR (the classic kernel
// is the shipped default; the tensor path's f16 activation staging flipped a
// 0.6-margin reference decode decision outside the fork envelope — see
// docs/parity.md §3b). When opted in it serves matmul_f16's mm branch
// (ne11 >= 8); the decode gemv (ne11 < 8) never reaches here — it always runs
// f16.metal's classic kernel_mul_mv_f16_f32_v.
//
// One tensor instantiation, half operand tiles: the f32 activation is staged to
// half (matmul2d over HALF cooperative tensors is the only instantiation the
// mm_id.metal probe test validates), the stored f16 weight is widened to half,
// products accumulate in f32, output is f32. So the activation is rounded to
// half here — one extra f16 rounding over the classic kernel's float tiles —
// which puts this path in the fork's own prefill precision class (~2e-4 vs the
// classic kernel; the graduated f16.rs numerics test pins < 5e-4). Attention
// activations are post-RMSNorm and bounded, so no rescale guard is needed (the
// fork stages them half unguarded too).
//
// The tile geometry (NR0=64 out-rows, NR1=32 tokens, NK=32), the ROW-MAJOR
// (NK-strided) tile stores, the matmul2d descriptor and the cooperative-tensor
// store are the validated mm_id.metal::kernel_mul_mm_id_t layout. Only the input
// addressing (dense f16 weight + f32 activation, no per-expert gather / dequant)
// and the output store (dense [t, n_out], no token-slot scatter) are the
// dense-kernel forms from f16.metal.
//
// DELIBERATELY a SEPARATE library from f16.metal: this file needs Metal-4
// (<metal_tensor> + matmul2d), so it is compiled lazily on first tensor-path
// dispatch (src/ops/pipelines.rs::f16_t_pipeline) and the classic f16.metal
// library stays Metal-4-free. Mirrors the mm_id.metal / mm_id_t_hp.metal split.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

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

// Dense f16-weight gemm on the cooperative-tensor path: half operand tiles,
// f32 accumulate, f32 output. Grid: (ceil(ne1/NR1), ceil(ne0/NR0), 1); 128
// threads (4 simdgroups); 8192 B threadgroup memory (sa+sb half tiles, the
// store-back reuses the region as an NR0 x NR1 float tile).
kernel void kernel_mul_mm_f16_f32_t(
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
    threadgroup half  * sb = (threadgroup half  *)(shmem + sizeof(half) * 64 * 32);
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

    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK,  NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK ));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NR1, NR0, NK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

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

        // stage the f32 activation to half (the one f16 rounding of this path).
        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            *(threadgroup half2x4 *)(sb + NK*(8*sy + ly) + 8*sx) = (half2x4)(*((device const float2x4 *) y));
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
