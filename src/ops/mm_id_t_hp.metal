// Split-out `_t_hp` (float-operand cooperative tensor) instantiations of the
// mm_id tensor-path kernel. Kept OUT of mm_id.metal so the DEFAULT prefill
// library carries no float-cooperative-tensor code: matmul2d over FLOAT
// cooperative tensors is speculative (the tensor_matmul2d_probe test validates
// only half operands), and a future toolchain that rejects it would then fail
// only the opt-in LAGUNA_MM_ID_TENSOR_HP path, not the default library.
//
// NOT a standalone translation unit: it references the kernel_mul_mm_id_t
// template, the mul_mm_id_t typedef, the block_q4_K/block_q6_K layouts and the
// dequantize_q4_K/dequantize_q6_K functions, all defined in mm_id.metal.
// src/ops/pipelines.rs compiles it ONLY by concatenating it onto mm_id.metal
// (the TensorHp library), and only on first TensorHp dispatch. The two
// instantiation lines below are byte-identical to the block previously inline in
// mm_id.metal (no kernel math change).
//
// Only instantiated for q4_K/q6_K (the production experts); mm_kernel_instantiated
// in dispatch.rs restricts TensorHp to those dtypes, and the
// instantiation_matrix_matches_metal test cross-checks that against THIS file.

// Re-assert the math-mode pin for the concatenated TensorHp library (mm_id.metal
// already sets it at ITS file scope; repeating the same value here is a no-op
// that keeps this fragment self-describing). See mm_id.metal for the rationale.
#pragma METAL fp math_mode(fast)

template [[host_name("kernel_mul_mm_id_q4_K_f32_t_hp")]] kernel mul_mm_id_t kernel_mul_mm_id_t<float, float4x4, simdgroup_float8x8, float, float2x4, simdgroup_float8x8, block_q4_K, QK_NL, dequantize_q4_K, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f32_t_hp")]] kernel mul_mm_id_t kernel_mul_mm_id_t<float, float4x4, simdgroup_float8x8, float, float2x4, simdgroup_float8x8, block_q6_K, QK_NL, dequantize_q6_K, float, float4x4, float, float2x4>;
