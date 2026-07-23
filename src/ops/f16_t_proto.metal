// PROTOTYPE (not production-wired): Metal-4 cooperative-tensor ports of the
// f16-weight x f32-activation attention prefill gemm (the classic simdgroup
// kernel_mul_mm_f16_f32_v in src/ops/f16.metal). Measures whether an
// mpp::tensor_ops::matmul2d accumulate beats the shipped classic path on the
// dense attention projection shapes, and at what numeric cost. Compiled at
// runtime only from the f16.rs proto bench test module — nothing on the default
// path includes it.
//
// Two variants, mirroring the mm_id.metal tensor precedent exactly:
//   kernel_mul_mm_f16_f32_t     (variant B, analog of mm_id `_t`): half operand
//     tiles. The f32 activation is staged to half (fork-faithful: the fork's
//     only tensor instantiation is half operands), f32 accumulate.
//   kernel_mul_mm_f16_f32_t_hp  (variant A, analog of mm_id `_t_hp`): float
//     operand tiles. The stored f16 weight widened to float and the activation
//     kept UNROUNDED, matmul2d over FLOAT cooperative tensors, f32 accumulate —
//     the "weights are the only f16 rounding" contract, same as the shipped
//     classic kernel. Speculative (matmul2d over float cooperative tensors is
//     validated only for half by the mm_id probe test), which is why it lives in
//     this isolated prototype library.
//
// The tile geometry (NR0=64 out-rows, NR1=32 tokens, NK=32), the ROW-MAJOR
// (NK-strided) tile stores, the matmul2d descriptor and the cooperative-tensor
// store are copied verbatim from mm_id.metal::kernel_mul_mm_id_t (the validated
// layout). Only the input addressing (dense f16 weight + f32 activation, no
// per-expert gather / dequant) and the output store (dense [t, n_out], no
// token-slot scatter) are the dense-kernel forms from f16.metal.

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

// Dense f16-weight gemm on the cooperative-tensor path. S0 is the weight-tile
// element type, S1 the activation-tile element type (half for variant B, float
// for variant A); the accumulate and output are always f32.
template<typename S0, typename S0_4x4, typename S1, typename S1_2x4>
kernel void kernel_mul_mm_f16_f32_t_impl(
        constant mm_args   & args [[buffer(0)]],
        device const char * src0  [[buffer(1)]],
        device const char * src1  [[buffer(2)]],
        device       char * dst   [[buffer(3)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup S0    * sa = (threadgroup S0    *)(shmem);
    threadgroup S1    * sb = (threadgroup S1    *)(shmem + sizeof(S0) * 64 * 32);
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

    auto tA = tensor<threadgroup S0, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK,  NR0));
    auto tB = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK ));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NR1, NR0, NK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // dequantize (f16 widen) + store to threadgroup memory, row-major (NK-strided).
        {
            S0_4x4 temp_a = S0_4x4(*x);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = temp_a[i/4][i%4];
            }
        }

        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            *(threadgroup S1_2x4 *)(sb + NK*(8*sy + ly) + 8*sx) = (S1_2x4)(*((device const float2x4 *) y));
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

// Both instantiations share the same kernel signature, so one decltype names the
// type (the template params only shape the body, exactly as in mm_id.metal).
typedef decltype(kernel_mul_mm_f16_f32_t_impl<half, half4x4, half, half2x4>) mul_mm_f16_t;

// Variant B (`_t`): half operand tiles (activation staged to f16).
template [[host_name("kernel_mul_mm_f16_f32_t")]]    kernel mul_mm_f16_t kernel_mul_mm_f16_f32_t_impl<half,  half4x4,  half,  half2x4>;
// Variant A (`_t_hp`): float operand tiles (activation kept unrounded).
template [[host_name("kernel_mul_mm_f16_f32_t_hp")]] kernel mul_mm_f16_t kernel_mul_mm_f16_f32_t_impl<float, float4x4, float, float2x4>;
