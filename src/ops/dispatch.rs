//! Host-side dispatch plumbing shared by the indexed-MoE matvec/matmul kernels.
//!
//! candle's metallib ships `kernel_mul_mv_id_*` / `kernel_mul_mm_id_*` (the
//! quantized gather matmuls used for the expert FFN) but wires no Rust host code
//! to them. These helpers encode those kernels directly, mirroring the geometry
//! candle uses for its non-indexed `call_quantized_matmul_mv_t` / `_mm_t` and the
//! ggml-metal reference encode functions.

use std::sync::Arc;

use anyhow::{Result, bail};
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, MetalDevice, MetalStorage, Shape, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder};
use candle_metal_kernels::source::Source;
use candle_metal_kernels::utils::EncoderProvider;

use crate::gguf::ExpertStack;
use crate::ops::{MmVariant, pipelines};

/// Build an `MTLSize` without naming `objc2_metal::MTLSize` — laguna does not
/// depend on objc2-metal directly, and a function cannot return the unnameable
/// type. `get_block_dims(1,1,1)` returns `MTLSize { 1, 1, 1 }`; its three fields
/// are public, so we overwrite them with the grid we want.
macro_rules! mtl_size {
    ($w:expr, $h:expr, $d:expr) => {{
        let mut sz = candle_metal_kernels::utils::get_block_dims(1, 1, 1);
        sz.width = $w;
        sz.height = $h;
        sz.depth = $d;
        sz
    }};
}

/// Everything a single indexed dispatch needs, resolved to raw device buffers.
/// Shapes follow the seam contract: weights `[n_expert, n_out, k]`, activations
/// `x` `[t, x_per_row, k]`, ids `[t, top_k]`, output `[t, top_k, n_out]`.
pub(crate) struct IdDispatch<'a> {
    pub weights: &'a Buffer,
    /// Byte offset of the expert stack's first block inside `weights`
    /// (`ExpertStack.base_off`): 0 for a dedicated classic allocation, the
    /// sub-page file offset for an mmap-aliased page-floored view. EVERY
    /// encode that binds `weights` must pass this as the buffer offset.
    pub w_off: usize,
    pub x: &'a Buffer,
    pub x_off: usize,
    pub ids: &'a Buffer,
    pub ids_off: usize,
    pub dst: &'a Buffer,
    pub n_expert: usize,
    pub n_out: usize,
    pub k: usize,
    pub t: usize,
    pub top_k: usize,
    pub x_per_row: usize,
    /// Byte stride between rows of one expert = (k / block_size) * type_size.
    pub bytes_per_row: usize,
    /// Byte stride between experts = n_out * bytes_per_row.
    pub per_expert: usize,
}

/// Threadgroup geometry for the matvec kernels, per quantized dtype. Copied from
/// candle's `call_quantized_matmul_mv_t` table — the id kernels dispatch the same
/// per-dtype `impl_fn`, so they want the same `(nth0, nth1)` threadgroup shape and
/// the same `align` row-block grouping over the output dimension.
struct MvGeom {
    nth0: usize,
    nth1: usize,
    align: usize,
}

fn mv_geom(dt: GgmlDType) -> Result<MvGeom> {
    let g = match dt {
        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q8_0 => MvGeom { nth0: 8, nth1: 8, align: 8 },
        GgmlDType::Q2K => MvGeom { nth0: 2, nth1: 32, align: 4 },
        GgmlDType::Q4K => MvGeom { nth0: 4, nth1: 8, align: 4 },
        GgmlDType::Q3K | GgmlDType::Q5K => MvGeom { nth0: 2, nth1: 32, align: 4 },
        GgmlDType::Q6K => MvGeom { nth0: 2, nth1: 32, align: 2 },
        GgmlDType::F16 | GgmlDType::F32 => MvGeom { nth0: 32, nth1: 1, align: 8 },
        other => bail!("no kernel_mul_mv_id kernel for dtype {other:?}"),
    };
    Ok(g)
}

fn mv_kernel_name(dt: GgmlDType) -> Result<&'static str> {
    let n = match dt {
        GgmlDType::Q4_0 => "kernel_mul_mv_id_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mv_id_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mv_id_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mv_id_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mv_id_q8_0_f32",
        GgmlDType::Q2K => "kernel_mul_mv_id_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mv_id_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mv_id_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mv_id_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mv_id_q6_K_f32",
        GgmlDType::F16 => "kernel_mul_mv_id_f16_f32",
        GgmlDType::F32 => "kernel_mul_mv_id_f32_f32",
        other => bail!("no kernel_mul_mv_id kernel for dtype {other:?}"),
    };
    Ok(n)
}

/// The vendored two-pass `kernel_mul_mm_id_<dtype>_f32` (src/ops/mm_id.metal)
/// is only instantiated for the dtypes the tests and the production Q4_K_M
/// experts use; other dtypes stay on the mv_id path.
pub(crate) fn mm_kernel_name(dt: GgmlDType) -> Result<&'static str> {
    let n = match dt {
        GgmlDType::Q8_0 => "kernel_mul_mm_id_q8_0_f32",
        GgmlDType::Q4K => "kernel_mul_mm_id_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mm_id_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mm_id_q6_K_f32",
        other => bail!("no vendored kernel_mul_mm_id kernel for dtype {other:?}; use mul_mv_id"),
    };
    Ok(n)
}

/// Whether the vendored `kernel_mul_mm_id_<dtype>_f32<variant-suffix>` kernel is
/// actually instantiated in mm_id.metal for this (dtype, variant) pair. The base
/// dtype matrix (q8_0/q4_K/q5_K/q6_K) is `mm_kernel_name`; ON TOP of that, the
/// `_t_hp` (`TensorHp`) variant is instantiated ONLY for the production q4_K/q6_K
/// experts — the other three variants cover the full base matrix. A combo outside
/// this matrix has no pipeline, so `moe` must fall back to mv_id rather than fault
/// the pipeline lookup. Keep in lockstep with the `template [[host_name(...)]]`
/// instantiations in mm_id.metal (the `mm_id::tests::instantiation_matrix_matches_metal`
/// test cross-checks this against the source).
pub(crate) fn mm_kernel_instantiated(dt: GgmlDType, variant: MmVariant) -> bool {
    if mm_kernel_name(dt).is_err() {
        return false;
    }
    match variant {
        MmVariant::TensorHp => matches!(dt, GgmlDType::Q4K | GgmlDType::Q6K),
        MmVariant::Tensor | MmVariant::ClassicHp | MmVariant::ClassicF16 => true,
    }
}

/// The `kernel_mul_mm_id_map0` template is instantiated for these top_k values
/// in mm_id.metal; a top_k outside the set has no map0 pass.
pub(crate) fn map0_kernel_name(top_k: usize) -> Result<String> {
    match top_k {
        1 | 2 | 4 | 5 | 6 | 8 | 10 => Ok(format!("kernel_mul_mm_id_map0_ne20_{top_k}")),
        other => bail!("no kernel_mul_mm_id_map0 instantiation for top_k={other}; use mul_mv_id"),
    }
}

// The 64x32-tile threadgroup memory each mm_id variant reserves (fixed
// regardless of token count — the two-pass row map lives in device scratch, not
// threadgroup memory) is `MmVariant::tile_smem()`: 8192 B for the half-tile
// variants (sa 4096 + sb 2048, store-back float tile reuses the region up to
// NR0*NR1*4 = 8192) and 12288 B for the f32 `_hp` tiles (sa 8192 + sb 4096).
/// Apple-silicon threadgroup memory ceiling; we refuse a launch that would exceed
/// it rather than let the GPU fault.
const MAX_THREADGROUP_SMEM: usize = 32768;

/// `ggml_metal_kargs_mul_mm_id_map0` (ggml-metal-impl.h). `#[repr(C)]` matches
/// the Metal `constant` struct layout byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct Map0Args {
    ne02: i32,
    ne10: i32,
    ne11: i32,
    nb11: u64,
    nb12: u64,
    ne21: i32,
    ne20: i32,
    nb21: u64,
}

/// `ggml_metal_kargs_mul_mm_id` (ggml-metal-impl.h).
#[repr(C)]
#[derive(Clone, Copy)]
struct MmIdArgs {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne11: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne20: i32,
    ne21: i32,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
}

/// `ggml_metal_kargs_mul_mv` (ggml-metal-impl.h). Written to buffer(0) of the
/// vendored plain mat-vec kernels (`kernel_mul_mv_<dtype>_f32_v`). `#[repr(C)]`
/// matches the Metal `constant` struct layout byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct MvArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    nr0: i32,
    r2: i16,
    r3: i16,
}

/// `ggml_metal_kargs_mul_mm` (ggml-metal-impl.h). Written to buffer(0) of the
/// vendored `kernel_mul_mm_f16_f32_v` prefill gemm. `#[repr(C)]` matches the
/// Metal `constant` struct layout byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct MmArgs {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
}

/// `ggml_metal_kargs_mul_mv_id` (ggml-metal-impl.h). Written to buffer(0) of the
/// vendored indexed mat-vec kernels (`kernel_mul_mv_id_<dtype>_f32_v`).
#[repr(C)]
#[derive(Clone, Copy)]
struct MvIdArgs {
    nei0: i32,
    nei1: i32,
    nbi1: u64,
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    ne0: i32,
    ne1: i32,
    nb1: u64,
    nr0: i32,
}

/// Matches the Metal `combine_args` struct (src/ops/combine.metal). `#[repr(C)]`
/// pins the layout byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct CombineArgs {
    top_k: i32,
    n_out: i32,
}

/// candle's `fast_sum` threadgroup width for a `top_k`-wide reduction:
/// `min(pipeline_max, next_pow2(top_k/2))`. The combine kernels reproduce it so
/// the simd_sum lane partition matches candle's reduction order bit-for-bit, but
/// they fold only ONE 32-lane simdgroup (see combine.metal), so a width above 32
/// would silently drop lanes 32.. — `run_combine` bails when this exceeds 32.
fn combine_reduction_width(top_k: usize) -> usize {
    (top_k / 2).next_power_of_two()
}

/// Whether the combine kernel's i32 index math (`down_base = s*top_k*n_out + c`,
/// plus the strided `k*n_out` loads) stays within i32 for the whole grid. The
/// largest flat index into `down` approaches `seq*top_k*n_out`; computed in i64
/// so the check itself cannot overflow. `run_combine` bails when this is false
/// rather than let the kernel wrap to a negative offset.
fn combine_index_fits_i32(seq: usize, top_k: usize, n_out: usize) -> bool {
    (seq as i64) * (top_k as i64) * (n_out as i64) <= i32::MAX as i64
}

/// The fork's host-side mv/mm break-even for the float-family mul_mat path
/// (ggml-metal-ops.cpp `ne11_mm_min`): the tiled matmul kernel is dispatched
/// when the token count EXCEEDS this, the gemv otherwise. `run_matmul_f16`
/// mirrors it. (ggml additionally has small-batch `mul_mv_ext` kernels for
/// ne11 2..8; not vendored — see TODO.md — so those seqs ride the gemv.)
const F16_MM_MIN_SEQ: usize = 8;

/// Fork host constants for `kernel_mul_mv_f16_f32_v` at our shapes: nr0 = 2
/// src0 rows per threadgroup (the only case in ggml's disp switch) and
/// nsg = min(4, ceil(ne00/128)) = 4 simdgroups splitting the K reduction
/// (every attention K is >= 3072). Baked into the kernel as well.
const MV_F16_NR0: usize = 2;
const MV_F16_NSG: usize = 4;

/// ggml's N_R0 * N_SG for the vendored q4_K/q6_K mat-vec kernels (both dtypes
/// carry N_R0 = 2, N_SG = 2 in ggml-metal-impl.h). Rows are grouped into blocks
/// of this many along the output dimension (grid.x), and each threadgroup runs
/// N_SG simdgroups of 32 threads.
const MV_NR0: usize = 2;
const MV_NSG: usize = 2;

/// True iff the vendored ggml-geometry mat-vec kernels exist for `dt` (only the
/// q4_K experts and q6_K experts/lm_head are ported). Other dtypes stay on the
/// candle baked path.
pub fn mv_vendored_supported(dt: GgmlDType) -> bool {
    matches!(dt, GgmlDType::Q4K | GgmlDType::Q6K)
}

fn mv_vendored_id_kernel_name(dt: GgmlDType) -> Result<&'static str> {
    match dt {
        GgmlDType::Q4K => Ok("kernel_mul_mv_id_q4_K_f32_v"),
        GgmlDType::Q6K => Ok("kernel_mul_mv_id_q6_K_f32_v"),
        other => bail!("no vendored kernel_mul_mv_id kernel for dtype {other:?}"),
    }
}

fn mv_vendored_plain_kernel_name(dt: GgmlDType) -> Result<&'static str> {
    match dt {
        GgmlDType::Q4K => Ok("kernel_mul_mv_q4_K_f32_v"),
        GgmlDType::Q6K => Ok("kernel_mul_mv_q6_K_f32_v"),
        other => bail!("no vendored kernel_mul_mv kernel for dtype {other:?}"),
    }
}

/// Encode the vendored `kernel_mul_mv_id_<dtype>_f32_v` (decode path). Same seam
/// contract as `encode_mul_mv_id`, but dispatches our ggml-geometry kernel: each
/// threadgroup covers `MV_NR0*MV_NSG` output rows across `MV_NSG` simdgroups, and
/// grid.z enumerates every (token, slot) pair (the id wrapper decodes z). The
/// argument struct goes to buffer(0) (ggml layout), matching the kernel signature.
pub(crate) fn encode_mul_mv_id_vendored(
    device: &MetalDevice,
    ep: impl EncoderProvider,
    dt: GgmlDType,
    d: &IdDispatch,
) -> Result<()> {
    let name = mv_vendored_id_kernel_name(dt)?;
    let pipeline = pipelines::mv_pipeline(device.device(), name)?;

    let args = MvIdArgs {
        nei0: d.top_k as i32,
        nei1: d.t as i32,
        nbi1: (d.top_k * DType::U32.size_in_bytes()) as u64,
        ne00: d.k as i32,
        ne01: d.n_out as i32,
        ne02: d.n_expert as i32,
        nb00: 0,
        nb01: d.bytes_per_row as u64,
        nb02: d.per_expert as u64,
        ne10: d.k as i32,
        ne11: d.x_per_row as i32,
        ne12: d.t as i32,
        ne13: 1,
        nb10: DType::F32.size_in_bytes() as u64,
        nb11: (d.k * DType::F32.size_in_bytes()) as u64,
        nb12: (d.x_per_row * d.k * DType::F32.size_in_bytes()) as u64,
        ne0: d.n_out as i32,
        ne1: d.top_k as i32,
        nb1: (d.n_out * DType::F32.size_in_bytes()) as u64,
        nr0: MV_NR0 as i32,
    };

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_bytes(0, &args);
    encoder.set_input_buffer(1, Some(d.weights), d.w_off);
    encoder.set_input_buffer(2, Some(d.x), d.x_off);
    encoder.set_output_buffer(3, Some(d.dst), 0);
    encoder.set_input_buffer(4, Some(d.ids), d.ids_off);

    // K-quant grid.x groups n_out rows into MV_NR0*MV_NSG-wide blocks; grid.z
    // walks every (token, slot) pair; threads are MV_NSG simdgroups of 32.
    let grid = mtl_size!(d.n_out.div_ceil(MV_NR0 * MV_NSG), 1, d.top_k * d.t);
    let threads = mtl_size!(32, MV_NSG, 1);
    encoder.dispatch_thread_groups(grid, threads);
    Ok(())
}

/// Encode `kernel_mul_mv_id_<dtype>_f32` (decode path). Each threadgroup along z
/// handles one (token, expert-slot) pair; the kernel reads the expert id from the
/// ids buffer and offsets `weights` by `expert * per_expert`.
pub(crate) fn encode_mul_mv_id(
    device: &MetalDevice,
    ep: impl EncoderProvider,
    dt: GgmlDType,
    d: &IdDispatch,
) -> Result<()> {
    let geom = mv_geom(dt)?;
    let name = mv_kernel_name(dt)?;

    // Kernel argument order mirrors kernel_mul_mv_id's signature exactly.
    let nei0 = d.top_k as i64;
    let nei1 = d.t as i64;
    let nbi1 = (d.top_k * DType::U32.size_in_bytes()) as u64;
    let ne00 = d.k as i64;
    let ne01 = d.n_out as i64;
    let ne02 = d.n_expert as i64;
    let nb00 = 0u64;
    let nb01 = d.bytes_per_row as u64;
    let nb02 = d.per_expert as u64;
    let ne10 = d.k as i64;
    let ne11 = d.x_per_row as i64;
    let ne12 = d.t as i64;
    let ne13 = 1i64;
    let nb10 = DType::F32.size_in_bytes() as u64;
    let nb11 = (d.k * DType::F32.size_in_bytes()) as u64;
    let nb12 = (d.x_per_row * d.k * DType::F32.size_in_bytes()) as u64;
    let ne0 = d.n_out as i64;
    let ne1 = d.top_k as i64;
    let nb1 = (d.n_out * DType::F32.size_in_bytes()) as u64;

    let pipeline = device
        .kernels()
        .load_pipeline(device.device(), Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    candle_metal_kernels::set_params!(
        encoder,
        (
            (d.weights, d.w_off),
            (d.x, d.x_off),
            candle_metal_kernels::Output::new(d.dst),
            (d.ids, d.ids_off),
            nei0,
            nei1,
            nbi1,
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            ne13,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            nb1
        )
    );

    // grid.x groups the n_out rows into `align`-wide blocks; grid.z walks all
    // top_k*t (token, slot) pairs (the id wrapper decodes z into token+slot).
    let grid = mtl_size!(d.n_out.div_ceil(geom.align), 1, d.top_k * d.t);
    let threads = mtl_size!(geom.nth0, geom.nth1, 1);
    encoder.dispatch_thread_groups(grid, threads);
    Ok(())
}

/// Count of 4-byte scratch slots `run` over-allocates on the dst buffer's tail
/// for the mm_id two-pass: the per-expert token count (`tpe`, n_expert i32) then
/// the per-expert compacted token-slot list (`ids-map`, n_expert*t i32). The dst
/// buffer is f32 and these entries are i32 (both 4 bytes), so one slot == one
/// dst element. Living in the dst allocation, the scratch shares its lifetime
/// (the returned tensor keeps it resident) instead of racing the buffer pool.
pub(crate) fn mm_scratch_elems(n_expert: usize, t: usize) -> usize {
    n_expert + n_expert * t
}

/// The live-field subset needed to encode the map0 pass. map0's output (per-expert
/// token count + compacted token-slot list) depends ONLY on the ids and
/// t/top_k/n_expert — the expert count comes from the dispatched thread count and
/// the other `Map0Args` fields (ne10/ne11/nb11/nb12) are not read by the kernel —
/// so ONE map0 pass serves every projection of a MoE block regardless of each
/// projection's k / x_per_row (they differ between gate/up and down).
struct Map0Dispatch<'a> {
    ids: &'a Buffer,
    ids_off: usize,
    n_expert: usize,
    top_k: usize,
    t: usize,
}

/// Byte width of one scratch entry (tpe counts and ids-map slots are both i32).
const MM_SCRATCH_ENTRY_BYTES: usize = 4;

/// Encode the map0 pass: one thread per expert builds that expert's compacted
/// token-slot list (`ids-map`, written at `ids_map_off`) and its token-slot count
/// (`tpe`, written at `tpe_off`) into `scratch`. `tpe` is `n_expert` i32; `ids-map`
/// is `n_expert*t` i32 (see `mm_scratch_elems`). The dead `Map0Args` fields
/// (ne10/ne11/nb11/nb12) are zeroed — the kernel never reads them.
fn encode_map0(
    device: &MetalDevice,
    ep: impl EncoderProvider,
    m: &Map0Dispatch,
    scratch: &Buffer,
    tpe_off: usize,
    ids_map_off: usize,
) -> Result<()> {
    let map0_name = map0_kernel_name(m.top_k)?;
    let map0 = pipelines::mm_id_pipeline(device.device(), &map0_name)?;

    // map0 runs one thread per expert; the ids scratch it reads into holds
    // n_expert * top_k u16 entries.
    let map0_smem = m.n_expert * m.top_k * std::mem::size_of::<u16>();
    if map0_smem > MAX_THREADGROUP_SMEM {
        bail!(
            "kernel_mul_mm_id_map0 needs {map0_smem} bytes of threadgroup memory for \
             n_expert={} top_k={}, over the {MAX_THREADGROUP_SMEM}-byte limit",
            m.n_expert,
            m.top_k
        );
    }
    if m.n_expert > map0.max_total_threads_per_threadgroup() {
        bail!(
            "kernel_mul_mm_id_map0 dispatches {} threads/threadgroup, over the pipeline max {}",
            m.n_expert,
            map0.max_total_threads_per_threadgroup()
        );
    }

    let map0_args = Map0Args {
        ne02: m.n_expert as i32,
        ne10: 0,
        ne11: 0,
        nb11: 0,
        nb12: 0,
        ne21: m.t as i32,
        ne20: m.top_k as i32,
        nb21: (m.top_k * DType::U32.size_in_bytes()) as u64,
    };
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    // buffers: 0=args, 1=ids, 2=tpe out, 3=ids-map out.
    encoder.set_compute_pipeline_state(&map0);
    encoder.set_bytes(0, &map0_args);
    encoder.set_input_buffer(1, Some(m.ids), m.ids_off);
    encoder.set_output_buffer(2, Some(scratch), tpe_off);
    encoder.set_output_buffer(3, Some(scratch), ids_map_off);
    encoder.set_threadgroup_memory_length(0, map0_smem);
    encoder.dispatch_thread_groups(mtl_size!(1, 1, 1), mtl_size!(m.n_expert, 1, 1));
    Ok(())
}

/// Encode the token-grouped matmul pass: each expert's threadgroups cover only
/// its own rows, read from the `tpe`/`ids-map` regions of `scratch` that a prior
/// `encode_map0` wrote. Writes the `[t, top_k, n_out]` result to `d.dst`.
///
/// `variant` picks the mm_id kernel family (tensor `_t` / classic `_hp` / classic
/// f16), threaded in from the single cached read in `ops::mm_id_variant`, never
/// re-read here. It sets the kernel host-name suffix and the tile smem.
///
/// Ordering: `encode_map0` marked tpe/ids-map as outputs and this pass reads them
/// as inputs on the same buffer, so candle inserts the RAW barrier automatically
/// (its Output-mark hazard tracking within an encoder, or the per-encoder fence
/// wait across encoders when the two passes are submitted separately).
fn encode_mm(
    device: &MetalDevice,
    ep: impl EncoderProvider,
    dt: GgmlDType,
    d: &IdDispatch,
    variant: MmVariant,
    scratch: &Buffer,
    tpe_off: usize,
    ids_map_off: usize,
) -> Result<()> {
    let mm_name = format!("{}{}", mm_kernel_name(dt)?, variant.suffix());
    let mm = pipelines::mm_id_pipeline(device.device(), &mm_name)?;

    let nb11 = (d.k * DType::F32.size_in_bytes()) as u64;
    let nb12 = (d.x_per_row * d.k * DType::F32.size_in_bytes()) as u64;

    let mm_args = MmIdArgs {
        ne00: d.k as i32,
        ne02: d.n_expert as i32,
        nb01: d.bytes_per_row as u64,
        nb02: d.per_expert as u64,
        nb03: 0,
        ne11: d.x_per_row as i32,
        nb10: DType::F32.size_in_bytes() as u64,
        nb11,
        nb12,
        nb13: 0,
        ne20: d.top_k as i32,
        ne21: d.t as i32,
        ne0: d.n_out as i32,
        ne1: d.top_k as i32,
        r2: 1,
        r3: 1,
    };
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    // buffers: 0=args, 1=weights, 2=x, 3=tpe, 4=ids-map, 5=dst.
    encoder.set_compute_pipeline_state(&mm);
    encoder.set_bytes(0, &mm_args);
    encoder.set_input_buffer(1, Some(d.weights), d.w_off);
    encoder.set_input_buffer(2, Some(d.x), d.x_off);
    encoder.set_input_buffer(3, Some(scratch), tpe_off);
    encoder.set_input_buffer(4, Some(scratch), ids_map_off);
    encoder.set_output_buffer(5, Some(d.dst), 0);
    encoder.set_threadgroup_memory_length(0, variant.tile_smem());

    // grid: 32-wide token-slot columns, 64-wide n_out rows, one z-slab per expert.
    let grid = mtl_size!(d.t.div_ceil(32), d.n_out.div_ceil(64), d.n_expert);
    encoder.dispatch_thread_groups(grid, mtl_size!(128, 1, 1));
    Ok(())
}

/// Encode the self-contained two-pass indexed matmul (standalone prefill path):
/// map0 then mm, with both scratch regions living at the tail of `d.dst` (offsets
/// past the `[t, top_k, n_out]` output). The returned tensor keeps the whole
/// allocation resident, so the scratch shares its lifetime instead of racing the
/// buffer pool. The shared-map0 production path (`prepare_mm_id_map0` +
/// `run_mm_shared`) uses a dedicated scratch buffer instead.
pub(crate) fn encode_mul_mm_id(
    device: &MetalDevice,
    ep: impl EncoderProvider + Copy,
    dt: GgmlDType,
    d: &IdDispatch,
    variant: MmVariant,
) -> Result<()> {
    let tpe_off = d.t * d.top_k * d.n_out * DType::F32.size_in_bytes();
    let ids_map_off = tpe_off + d.n_expert * MM_SCRATCH_ENTRY_BYTES;
    let m = Map0Dispatch {
        ids: d.ids,
        ids_off: d.ids_off,
        n_expert: d.n_expert,
        top_k: d.top_k,
        t: d.t,
    };
    encode_map0(device, ep, &m, d.dst, tpe_off, ids_map_off)?;
    encode_mm(device, ep, dt, d, variant, d.dst, tpe_off, ids_map_off)?;
    Ok(())
}

/// Wrap a freshly written f32 device buffer as an owned output `Tensor`.
pub(crate) fn output_tensor(
    dst: Arc<Buffer>,
    device: &MetalDevice,
    count: usize,
    shape: impl Into<Shape>,
) -> Tensor {
    let storage = MetalStorage::new(dst, device.clone(), count, DType::F32);
    Tensor::from_storage(
        Storage::Metal(storage),
        shape,
        candle_core::op::BackpropOp::none(),
        false,
    )
}

/// Which id-kernel family to dispatch.
#[derive(Clone, Copy)]
pub(crate) enum Mode {
    /// candle's baked `kernel_mul_mv_id` — one matvec per (token, slot); decode
    /// path, older geometry. Kept as the `LAGUNA_MV_CLASSIC` fallback.
    Mv,
    /// Vendored ggml-geometry `kernel_mul_mv_id_<dtype>_f32_v` — decode path,
    /// current geometry (default for the supported q4_K/q6_K dtypes).
    MvVendored,
    /// `kernel_mul_mm_id` — token-grouped matmul; prefill path.
    Mm,
}

/// A shared map0 scratch (`prepare_mm_id_map0`) plus the geometry it was laid out
/// for. The producer wrote `tpe` (n_expert i32 @ 0) then the `ids-map` at
/// `n_expert * MM_SCRATCH_ENTRY_BYTES` using ITS n_expert/t/top_k; a consumer that
/// recomputed that offset from a different stack's n_expert would read the wrong
/// region. Carrying the geometry lets `run_mm_shared` validate each consuming
/// projection against the producer before it reads the map.
pub(crate) struct Map0Scratch {
    buffer: Arc<Buffer>,
    n_expert: usize,
    t: usize,
    top_k: usize,
}

/// Where `Mode::Mm` reads its map0 scratch from.
enum MmScratch<'a> {
    /// Self-contained: map0 runs here and both scratch regions live at the tail of
    /// the freshly allocated dst (the returned tensor keeps them resident).
    Owned,
    /// Shared: map0 already ran into this dedicated scratch (`prepare_mm_id_map0`),
    /// so only the mm pass runs here, reading tpe @ 0 and ids-map @ n_expert*4 —
    /// after validating this projection's geometry against the producer's.
    Shared(&'a Map0Scratch),
}

/// Validate the seam shapes, resolve every operand to a device buffer, and encode
/// the requested id kernel. Returns the `[t, top_k, n_out]` output tensor.
/// `variant` is only consulted for `Mode::Mm` (which mm_id kernel family);
/// callers pass the cached `ops::mm_id_variant()` in production and an explicit
/// value in A/B tests. `Mode::Mv` ignores it.
pub(crate) fn run(
    stack: &ExpertStack,
    x: &Tensor,
    ids: &Tensor,
    mode: Mode,
    variant: MmVariant,
) -> Result<Tensor> {
    run_inner(stack, x, ids, mode, variant, MmScratch::Owned)
}

/// Run one `Mode::Mm` projection against a shared map0 scratch (`prepare_mm_id_map0`),
/// skipping the map0 pass. Used by `FusedExperts::forward` so the block's three
/// projections build the token-slot map once. `scratch` must stay alive until this
/// dispatch is submitted (the caller holds it across gate/up/down).
pub(crate) fn run_mm_shared(
    stack: &ExpertStack,
    x: &Tensor,
    ids: &Tensor,
    variant: MmVariant,
    scratch: &Map0Scratch,
) -> Result<Tensor> {
    run_inner(stack, x, ids, Mode::Mm, variant, MmScratch::Shared(scratch))
}

fn run_inner(
    stack: &ExpertStack,
    x: &Tensor,
    ids: &Tensor,
    mode: Mode,
    variant: MmVariant,
    scratch: MmScratch,
) -> Result<Tensor> {
    let cdev = x.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("mul_*_id requires x on a Metal device");
    };

    let (t, x_per_row, kx) = x.dims3().map_err(|e| anyhow::anyhow!("x must be rank-3 [t, x_per_row, k]: {e}"))?;
    let (t_ids, top_k) = ids.dims2().map_err(|e| anyhow::anyhow!("ids must be rank-2 [t, top_k]: {e}"))?;

    if x.dtype() != DType::F32 {
        bail!("x must be f32, got {:?}", x.dtype());
    }
    if ids.dtype() != DType::U32 {
        bail!("ids must be u32, got {:?}", ids.dtype());
    }
    if !x.is_contiguous() {
        bail!("x must be contiguous");
    }
    if !ids.is_contiguous() {
        bail!("ids must be contiguous");
    }
    if kx != stack.k {
        bail!("x k ({kx}) does not match expert stack k ({})", stack.k);
    }
    if t_ids != t {
        bail!("ids t ({t_ids}) does not match x t ({t})");
    }
    if x_per_row != 1 && x_per_row != top_k {
        bail!("x_per_row ({x_per_row}) must be 1 (shared row) or top_k ({top_k}) (per-slot row)");
    }

    let dt = stack.dtype;
    let block_size = dt.block_size();
    if !stack.k.is_multiple_of(block_size) {
        bail!("expert stack k ({}) is not a multiple of {dt:?} block size {block_size}", stack.k);
    }
    let bytes_per_row = stack.k / block_size * dt.type_size();
    let per_expert = stack.n_out * bytes_per_row;

    let Some(w_buf) = stack.buffer.as_deref() else {
        bail!("expert stack has no device buffer (not on a Metal device); fused MoE requires Metal");
    };

    let out_count = t * top_k * stack.n_out;
    // Owned Mm over-allocates the dst buffer to hold the two-pass scratch (tpe +
    // ids-map) at its tail; the returned tensor keeps the whole allocation
    // resident, so the scratch shares its lifetime and the pool reuses it once
    // the tensor drops. Shared Mm and the Mv paths write no scratch tail.
    let alloc_count = match (mode, &scratch) {
        (Mode::Mm, MmScratch::Owned) => out_count + mm_scratch_elems(stack.n_expert, t),
        _ => out_count,
    };
    let dst = mdev.new_buffer(alloc_count, DType::F32, "mul_id")?;

    let (x_guard, x_layout) = x.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("x is not on a Metal device");
    };
    let x_buf = x_storage.buffer();
    let x_off = x_layout.start_offset() * DType::F32.size_in_bytes();

    let (ids_guard, ids_layout) = ids.storage_and_layout();
    let Storage::Metal(ids_storage) = &*ids_guard else {
        bail!("ids is not on a Metal device");
    };
    let ids_buf = ids_storage.buffer();
    let ids_off = ids_layout.start_offset() * DType::U32.size_in_bytes();

    let d = IdDispatch {
        weights: w_buf,
        w_off: stack.base_off,
        x: x_buf,
        x_off,
        ids: ids_buf,
        ids_off,
        dst: &dst,
        n_expert: stack.n_expert,
        n_out: stack.n_out,
        k: stack.k,
        t,
        top_k,
        x_per_row,
        bytes_per_row,
        per_expert,
    };
    {
        let cmd = mdev.command_encoder()?;
        match (mode, &scratch) {
            (Mode::Mv, _) => encode_mul_mv_id(mdev, &cmd, dt, &d)?,
            (Mode::MvVendored, _) => encode_mul_mv_id_vendored(mdev, &cmd, dt, &d)?,
            (Mode::Mm, MmScratch::Owned) => encode_mul_mm_id(mdev, &cmd, dt, &d, variant)?,
            (Mode::Mm, MmScratch::Shared(s)) => {
                // The producer laid the ids-map out at `s.n_expert *
                // MM_SCRATCH_ENTRY_BYTES` and sized `tpe`/`ids-map` for its
                // t/top_k; a projection with a different geometry would read the
                // wrong region. Validate before using the producer's n_expert for
                // the offset (guaranteed == stack.n_expert once this passes).
                if s.n_expert != stack.n_expert || s.t != t || s.top_k != top_k {
                    bail!(
                        "shared map0 scratch geometry (n_expert={}, t={}, top_k={}) does not match \
                         this projection (n_expert={}, t={}, top_k={}); the ids-map offset would be wrong",
                        s.n_expert, s.t, s.top_k, stack.n_expert, t, top_k
                    );
                }
                encode_mm(mdev, &cmd, dt, &d, variant, &s.buffer, 0, s.n_expert * MM_SCRATCH_ENTRY_BYTES)?
            }
        }
    }
    drop(x_guard);
    drop(ids_guard);

    Ok(output_tensor(dst, mdev, out_count, (t, top_k, stack.n_out)))
}

/// Allocate the shared map0 scratch for one MoE block and encode the single map0
/// pass from `ids`. The returned buffer holds `tpe` (n_expert i32 @ 0) then the
/// `ids-map` (n_expert*t i32 @ n_expert*4); all three projections read it via
/// `run_mm_shared`. map0's output depends only on ids/t/top_k/n_expert, so one
/// pass serves gate/up/down despite their differing k / x_per_row. The caller
/// keeps the returned buffer alive until the down projection's mm is submitted;
/// candle's per-encoder fences order the mm reads after this write.
pub(crate) fn prepare_mm_id_map0(n_expert: usize, ids: &Tensor) -> Result<Map0Scratch> {
    let cdev = ids.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("prepare_mm_id_map0 requires ids on a Metal device");
    };
    let (t, top_k) = ids.dims2().map_err(|e| anyhow::anyhow!("ids must be rank-2 [t, top_k]: {e}"))?;
    if ids.dtype() != DType::U32 {
        bail!("ids must be u32, got {:?}", ids.dtype());
    }
    if !ids.is_contiguous() {
        bail!("ids must be contiguous");
    }

    let scratch = mdev.new_buffer(mm_scratch_elems(n_expert, t), DType::F32, "mm_id_map0")?;

    let (ids_guard, ids_layout) = ids.storage_and_layout();
    let Storage::Metal(ids_storage) = &*ids_guard else {
        bail!("ids is not on a Metal device");
    };
    let ids_buf = ids_storage.buffer();
    let ids_off = ids_layout.start_offset() * DType::U32.size_in_bytes();

    let m = Map0Dispatch { ids: ids_buf, ids_off, n_expert, top_k, t };
    {
        let cmd = mdev.command_encoder()?;
        encode_map0(mdev, &cmd, &m, &scratch, 0, n_expert * MM_SCRATCH_ENTRY_BYTES)?;
    }
    drop(ids_guard);
    Ok(Map0Scratch { buffer: scratch, n_expert, t, top_k })
}

/// Plain (non-indexed) quantized mat-vec against the vendored ggml-geometry
/// kernel — the lm_head bypass at seq==1. `weight` is a rank-2 `[n_out, k]`
/// quantized tensor's raw device buffer; `x` is `[t, k]` f32 (t small, typically
/// 1). Returns `[t, n_out]` f32. Only q4_K/q6_K are supported (the lm_head is
/// q6_K); callers gate on `mv_vendored_supported` and fall back to QMatMul
/// otherwise.
pub(crate) fn run_plain_mv(
    weight: &Buffer,
    dt: GgmlDType,
    n_out: usize,
    k: usize,
    x: &Tensor,
) -> Result<Tensor> {
    let cdev = x.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("mul_mv requires x on a Metal device");
    };
    let (t, kx) = x.dims2().map_err(|e| anyhow::anyhow!("x must be rank-2 [t, k]: {e}"))?;
    if x.dtype() != DType::F32 {
        bail!("x must be f32, got {:?}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("x must be contiguous");
    }
    if kx != k {
        bail!("x k ({kx}) does not match weight k ({k})");
    }
    if !mv_vendored_supported(dt) {
        bail!("no vendored plain mv kernel for dtype {dt:?}");
    }

    let block_size = dt.block_size();
    if !k.is_multiple_of(block_size) {
        bail!("weight k ({k}) is not a multiple of {dt:?} block size {block_size}");
    }
    let bytes_per_row = k / block_size * dt.type_size();

    let out_count = t * n_out;
    let dst = mdev.new_buffer(out_count, DType::F32, "mul_mv")?;

    let (x_guard, x_layout) = x.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("x is not on a Metal device");
    };
    let x_buf = x_storage.buffer();
    let x_off = x_layout.start_offset() * DType::F32.size_in_bytes();

    let args = MvArgs {
        ne00: k as i32,
        ne01: n_out as i32,
        ne02: 1,
        nb00: 0,
        nb01: bytes_per_row as u64,
        nb02: (n_out * bytes_per_row) as u64,
        nb03: (n_out * bytes_per_row) as u64,
        ne10: k as i32,
        ne11: t as i32,
        ne12: 1,
        nb10: DType::F32.size_in_bytes() as u64,
        nb11: (k * DType::F32.size_in_bytes()) as u64,
        nb12: (t * k * DType::F32.size_in_bytes()) as u64,
        nb13: (t * k * DType::F32.size_in_bytes()) as u64,
        ne0: n_out as i32,
        ne1: t as i32,
        nr0: MV_NR0 as i32,
        r2: 1,
        r3: 1,
    };

    let name = mv_vendored_plain_kernel_name(dt)?;
    let pipeline = pipelines::mv_pipeline(mdev.device(), name)?;
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(weight), 0);
        encoder.set_input_buffer(2, Some(x_buf), x_off);
        encoder.set_output_buffer(3, Some(&dst), 0);

        // K-quant grid.x = ceil(n_out / (MV_NR0*MV_NSG)); grid.y = one column per
        // token row (nr1 == 1 for the quant mv path); threads MV_NSG simdgroups.
        let grid = mtl_size!(n_out.div_ceil(MV_NR0 * MV_NSG), t, 1);
        let threads = mtl_size!(32, MV_NSG, 1);
        encoder.dispatch_thread_groups(grid, threads);
    }
    drop(x_guard);

    Ok(output_tensor(dst, mdev, out_count, (t, n_out)))
}

/// Dense f16-weight x f32-activation matmul against the vendored ggml-geometry
/// kernels (f16.metal) — the attention projections. `weight` is a rank-2
/// `[n_out, k]` dense f16 tensor, `x` is `[t, k]` f32; returns `[t, n_out]` f32
/// with no activation cast and no output rounding (the stored f16 weights are
/// the only f16 in the chain). Dispatches per the fork's host split: the gemv
/// for t <= 8 tokens, the tiled gemm above (`F16_MM_MIN_SEQ`).
pub(crate) fn run_matmul_f16(weight: &Tensor, x: &Tensor) -> Result<Tensor> {
    let kernel = if crate::ops::attn_mm_classic() {
        F16MmKernel::Classic
    } else {
        F16MmKernel::Tensor
    };
    run_matmul_f16_variant(weight, x, kernel)
}

/// Which prefill (ne11 > 8) mm-branch kernel `run_matmul_f16_variant`
/// dispatches. Production only ever selects the first two (`run_matmul_f16`);
/// `TensorMixed` is reachable exclusively from the f16.rs probe tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum F16MmKernel {
    /// Classic simdgroup kernel, float tiles (`f16.metal`) — the
    /// `LAGUNA_ATTN_MM_CLASSIC` kill-switch.
    Classic,
    /// Metal-4 cooperative-tensor kernel, half operand tiles (`f16_t.metal`) —
    /// shipped default.
    Tensor,
    /// Mixed-operand cooperative-tensor probe: half weight tile x FLOAT
    /// activation tile (`f16_t_mixed.metal`). No env switch, no production
    /// selection — constructed only by the f16.rs tests (hence dead in
    /// non-test builds).
    #[allow(dead_code)]
    TensorMixed,
}

/// `run_matmul_f16` with the prefill (ne11 >= 8) mm-branch kernel chosen
/// explicitly. Production derives the kernel from the cached
/// `LAGUNA_ATTN_MM_CLASSIC` kill-switch (`run_matmul_f16`); the f16.rs tests
/// call this with an explicit kernel because the switch is a process-global
/// `OnceLock`. The decode gemv branch is identical for every kernel choice
/// (classic mv).
pub(crate) fn run_matmul_f16_variant(weight: &Tensor, x: &Tensor, kernel: F16MmKernel) -> Result<Tensor> {
    let cdev = x.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("matmul_f16 requires x on a Metal device");
    };
    let (n_out, k) = weight.dims2().map_err(|e| anyhow::anyhow!("weight must be rank-2 [n_out, k]: {e}"))?;
    let (t, kx) = x.dims2().map_err(|e| anyhow::anyhow!("x must be rank-2 [t, k]: {e}"))?;
    if weight.dtype() != DType::F16 {
        bail!("weight must be f16, got {:?}", weight.dtype());
    }
    if x.dtype() != DType::F32 {
        bail!("x must be f32, got {:?}", x.dtype());
    }
    if !weight.is_contiguous() {
        bail!("weight must be contiguous");
    }
    if !x.is_contiguous() {
        bail!("x must be contiguous");
    }
    if kx != k {
        bail!("x k ({kx}) does not match weight k ({k})");
    }
    // Both kernels stream K through vector types (half4/float4 in the gemv,
    // half4x4/float2x4 tiles in the gemm) and skip the fork's bc_inp/K-tail
    // handling, and the gemm's float4 output copy needs 16-byte-aligned dst
    // rows. Every attention shape satisfies these (K multiple of 1024, out
    // dims all multiples of 4).
    if !k.is_multiple_of(32) {
        bail!("matmul_f16 requires k % 32 == 0, got {k}");
    }
    if !n_out.is_multiple_of(4) {
        bail!("matmul_f16 requires n_out % 4 == 0, got {n_out}");
    }

    let out_count = t * n_out;
    let dst = mdev.new_buffer(out_count, DType::F32, "matmul_f16")?;

    let (w_guard, w_layout) = weight.storage_and_layout();
    let Storage::Metal(w_storage) = &*w_guard else {
        bail!("weight is not on a Metal device");
    };
    let w_buf = w_storage.buffer();
    let w_off = w_layout.start_offset() * DType::F16.size_in_bytes();
    // The kernels read the weight through half4/half4x4 device pointers, which
    // Metal requires 16-byte aligned; rows are (k % 8 == 0 checked above), so
    // only a misaligned view start could break it. Production passes whole
    // tensors (offset 0), but `matmul_f16` is pub and a future sliced f16 view
    // would land here.
    if !w_off.is_multiple_of(16) {
        bail!("matmul_f16 requires a 16-byte-aligned weight view, got byte offset {w_off}");
    }

    let (x_guard, x_layout) = x.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("x is not on a Metal device");
    };
    let x_buf = x_storage.buffer();
    let x_off = x_layout.start_offset() * DType::F32.size_in_bytes();
    // The kernels read x through float4/float2x4 device pointers, which Metal
    // requires 16-byte aligned; rows are (k % 4 == 0 checked above), so only a
    // misaligned view start could break it.
    if !x_off.is_multiple_of(16) {
        bail!("matmul_f16 requires a 16-byte-aligned x view, got byte offset {x_off}");
    }

    let nb01 = (k * DType::F16.size_in_bytes()) as u64;
    let nb11 = (k * DType::F32.size_in_bytes()) as u64;

    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        if t > F16_MM_MIN_SEQ {
            // Tiled gemm: 64(out) x 32(token) tiles, 128 threads / 4 simdgroups.
            // Default is the Metal-4 cooperative-tensor kernel (half operand
            // tiles, 8192 B smem); LAGUNA_ATTN_MM_CLASSIC reverts to the classic
            // simdgroup kernel (float tiles, 12288 B; the store-back tile reuses
            // the region); the test-only TensorMixed probe keeps the activation
            // tile float (half sa 4096 + float sb 4096 = the same 8192 B). Same
            // MmArgs, grid and threads for all — only the kernel, its library,
            // and the tile smem differ.
            let args = MmArgs {
                ne00: k as i32,
                ne02: 1,
                nb01,
                nb02: n_out as u64 * nb01,
                nb03: n_out as u64 * nb01,
                ne12: 1,
                nb10: DType::F32.size_in_bytes() as u64,
                nb11,
                nb12: t as u64 * nb11,
                nb13: t as u64 * nb11,
                ne0: n_out as i32,
                ne1: t as i32,
                r2: 1,
                r3: 1,
            };
            let (pipeline, smem) = match kernel {
                F16MmKernel::Classic => {
                    (pipelines::f16_pipeline(mdev.device(), "kernel_mul_mm_f16_f32_v")?, 12288)
                }
                F16MmKernel::Tensor => {
                    (pipelines::f16_t_pipeline(mdev.device(), "kernel_mul_mm_f16_f32_t")?, 8192)
                }
                F16MmKernel::TensorMixed => (
                    pipelines::f16_t_mixed_pipeline(mdev.device(), "kernel_mul_mm_f16_f32_t_mixed")?,
                    8192,
                ),
            };
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_bytes(0, &args);
            encoder.set_input_buffer(1, Some(w_buf), w_off);
            encoder.set_input_buffer(2, Some(x_buf), x_off);
            encoder.set_output_buffer(3, Some(&dst), 0);
            encoder.set_threadgroup_memory_length(0, smem);
            let grid = mtl_size!(t.div_ceil(32), n_out.div_ceil(64), 1);
            encoder.dispatch_thread_groups(grid, mtl_size!(128, 1, 1));
        } else {
            // gemv: NR0 rows per threadgroup, NSG simdgroups splitting K, one
            // grid.y column per token; smem is the cross-simdgroup reduce
            // scratch (NR0 * 32 floats).
            let args = MvArgs {
                ne00: k as i32,
                ne01: n_out as i32,
                ne02: 1,
                nb00: 0,
                nb01,
                nb02: n_out as u64 * nb01,
                nb03: n_out as u64 * nb01,
                ne10: k as i32,
                ne11: t as i32,
                ne12: 1,
                nb10: DType::F32.size_in_bytes() as u64,
                nb11,
                nb12: t as u64 * nb11,
                nb13: t as u64 * nb11,
                ne0: n_out as i32,
                ne1: t as i32,
                nr0: MV_F16_NR0 as i32,
                r2: 1,
                r3: 1,
            };
            let pipeline = pipelines::f16_pipeline(mdev.device(), "kernel_mul_mv_f16_f32_v")?;
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_bytes(0, &args);
            encoder.set_input_buffer(1, Some(w_buf), w_off);
            encoder.set_input_buffer(2, Some(x_buf), x_off);
            encoder.set_output_buffer(3, Some(&dst), 0);
            encoder.set_threadgroup_memory_length(
                0,
                MV_F16_NR0 * 32 * DType::F32.size_in_bytes(),
            );
            let grid = mtl_size!(n_out.div_ceil(MV_F16_NR0), t, 1);
            encoder.dispatch_thread_groups(grid, mtl_size!(32, MV_F16_NSG, 1));
        }
    }
    drop(w_guard);
    drop(x_guard);

    Ok(output_tensor(dst, mdev, out_count, (t, n_out)))
}

/// Fused MoE weighted combine against the vendored `combine.metal` kernels —
/// the routed-expert combine tail of `FusedExperts::forward`. Reads `down`
/// (`[seq, top_k, n_out]` f32 contiguous) once and returns `[seq, n_out]` f32:
///   - `col_l2` = `None`  (rescale-free): `dst[s,c] = Σ_k down[s,k,c] * w[s,k]`
///   - `col_l2` = `Some`  (`[seq, top_k, 1]` f32): the per-column L2 rescale is
///     undone in the same pass — `dst[s,c] = Σ_k down[s,k,c]*col_l2[s,k]*2^-15*w[s,k]`.
/// `weights` is `[seq, top_k]` f32. The launch geometry and per-op rounding
/// mirror candle's strided `sum(1)` exactly, so the result is bit-identical to
/// the candle broadcast/affine/sum chain (see combine.metal / the combine.rs test).
pub(crate) fn run_combine(
    down: &Tensor,
    col_l2: Option<&Tensor>,
    weights: &Tensor,
) -> Result<Tensor> {
    let cdev = down.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("combine requires down on a Metal device");
    };

    let (seq, top_k, n_out) = down
        .dims3()
        .map_err(|e| anyhow::anyhow!("down must be rank-3 [seq, top_k, n_out]: {e}"))?;
    if down.dtype() != DType::F32 {
        bail!("down must be f32, got {:?}", down.dtype());
    }
    if !down.is_contiguous() {
        bail!("down must be contiguous");
    }
    if weights.dims2().map_err(|e| anyhow::anyhow!("weights must be rank-2 [seq, top_k]: {e}"))?
        != (seq, top_k)
    {
        bail!("weights shape {:?} must be [seq, top_k] = [{seq}, {top_k}]", weights.dims());
    }
    if weights.dtype() != DType::F32 {
        bail!("weights must be f32, got {:?}", weights.dtype());
    }
    if !weights.is_contiguous() {
        bail!("weights must be contiguous");
    }
    if let Some(l2) = col_l2 {
        if l2.dims3().map_err(|e| anyhow::anyhow!("col_l2 must be rank-3 [seq, top_k, 1]: {e}"))?
            != (seq, top_k, 1)
        {
            bail!("col_l2 shape {:?} must be [seq, top_k, 1] = [{seq}, {top_k}, 1]", l2.dims());
        }
        if l2.dtype() != DType::F32 {
            bail!("col_l2 must be f32, got {:?}", l2.dtype());
        }
        if !l2.is_contiguous() {
            bail!("col_l2 must be contiguous");
        }
    }

    // The reduction is a single simd_sum over one 32-lane simdgroup, so the
    // candle-matching threadgroup width must not exceed 32 (see combine.metal);
    // a wider width would leave lanes 32.. in a second simdgroup whose partials
    // are never folded in. This is an error, not a fallback — production top_k is
    // 10 (width 8); a top_k needing width > 32 (i.e. >= 66) is out of contract.
    let width_hint = combine_reduction_width(top_k);
    if width_hint > 32 {
        bail!(
            "combine top_k={top_k} needs threadgroup width {width_hint} > 32; the single-simdgroup \
             simd_sum reduction would silently drop lanes 32.."
        );
    }
    // The kernels address `down` with i32 index math; a grid whose flat element
    // count exceeds i32::MAX would wrap to a negative offset.
    if !combine_index_fits_i32(seq, top_k, n_out) {
        bail!(
            "combine index math overflows i32: seq={seq} top_k={top_k} n_out={n_out} \
             (seq*top_k*n_out = {} exceeds i32::MAX)",
            (seq as i64) * (top_k as i64) * (n_out as i64)
        );
    }

    let name = if col_l2.is_some() {
        "kernel_moe_combine_rescale"
    } else {
        "kernel_moe_combine"
    };
    let pipeline = pipelines::combine_pipeline(mdev.device(), name)?;

    let out_length = seq * n_out;
    let dst = mdev.new_buffer(out_length, DType::F32, "combine")?;

    // Resolve operand buffers. `storage_and_layout` guards must outlive the encode.
    let (down_guard, down_layout) = down.storage_and_layout();
    let Storage::Metal(down_storage) = &*down_guard else {
        bail!("down is not on a Metal device");
    };
    let down_buf = down_storage.buffer();
    let down_off = down_layout.start_offset() * DType::F32.size_in_bytes();

    let (w_guard, w_layout) = weights.storage_and_layout();
    let Storage::Metal(w_storage) = &*w_guard else {
        bail!("weights is not on a Metal device");
    };
    let w_buf = w_storage.buffer();
    let w_off = w_layout.start_offset() * DType::F32.size_in_bytes();

    // The optional col_l2 guard is bound for the whole encode when present.
    let l2_resolved = match col_l2 {
        Some(l2) => {
            let (guard, layout) = l2.storage_and_layout();
            let off = layout.start_offset() * DType::F32.size_in_bytes();
            Some((guard, off))
        }
        None => None,
    };

    let args = CombineArgs {
        top_k: top_k as i32,
        n_out: n_out as i32,
    };
    // candle's `fast_sum_f32_strided` launch: out_length threadgroups, block_dim
    // = min(pipeline max, next_pow2(top_k/2)); reproduced so the simd_sum lane
    // partition (and thus the reduction order) is identical. The width guard
    // above pins `combine_reduction_width(top_k)` <= 32, so this stays within one
    // simdgroup.
    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        combine_reduction_width(top_k),
    );
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(down_buf), down_off);
        if let Some((l2_guard, l2_off)) = &l2_resolved {
            let Storage::Metal(l2_storage) = &**l2_guard else {
                bail!("col_l2 is not on a Metal device");
            };
            encoder.set_input_buffer(2, Some(l2_storage.buffer()), *l2_off);
            encoder.set_input_buffer(3, Some(w_buf), w_off);
            encoder.set_output_buffer(4, Some(&dst), 0);
        } else {
            encoder.set_input_buffer(2, Some(w_buf), w_off);
            encoder.set_output_buffer(3, Some(&dst), 0);
        }
        encoder.dispatch_thread_groups(mtl_size!(out_length, 1, 1), mtl_size!(width, 1, 1));
    }
    drop(down_guard);
    drop(w_guard);
    drop(l2_resolved);

    Ok(output_tensor(dst, mdev, out_length, (seq, n_out)))
}

/// Matches the Metal `attn_gate_args` struct (src/ops/attn_glue.metal).
/// `#[repr(C)]` pins the layout byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct AttnGateArgs {
    n_head: i32,
    seq: i32,
    head_dim: i32,
}

/// Matches the Metal `permute_args` struct (src/ops/attn_glue.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct PermuteArgs {
    d0: i32,
    d1: i32,
    d2: i32,
}

/// Matches the Metal `rope_args` struct (src/ops/rope.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct RopeArgs {
    heads: i32,
    seq: i32,
    head_dim: i32,
    n_rot: i32,
    pos: i32,
}

/// Matches the Metal `flash_attn_params` struct (src/ops/flash.metal).
/// `#[repr(C)]` pins the layout byte-for-byte: twelve 4-byte fields (48 bytes,
/// a multiple of 8) followed by eight `i64` element strides — no implicit
/// padding on either side.
#[repr(C)]
#[derive(Clone, Copy)]
struct FlashAttnArgs {
    gqa_factor: i32,
    scale: f32,
    nk: i32,
    nq_aligned: i32,
    nk_aligned: i32,
    ql_rem: i32,
    kl_rem: i32,
    kl: i32,
    q_off: i32,
    k_off: i32,
    window: i32,
    disable_skip: i32,
    q_stride_h: i64,
    q_stride_r: i64,
    k_stride_h: i64,
    k_stride_r: i64,
    v_stride_h: i64,
    v_stride_r: i64,
    o_stride_h: i64,
    o_stride_r: i64,
}

/// Linear one-thread-per-element launch shared by the attention-glue kernels:
/// `n` threads in threadgroups of up to 256 (bounds-checked in the kernels, so
/// the rounded-up tail is harmless).
fn dispatch_linear(
    encoder: &ComputeCommandEncoder,
    pipeline: &candle_metal_kernels::metal::ComputePipeline,
    n: usize,
) {
    let width = pipeline.max_total_threads_per_threadgroup().min(256);
    let grid = mtl_size!(n.div_ceil(width), 1, 1);
    let threads = mtl_size!(width, 1, 1);
    encoder.dispatch_thread_groups(grid, threads);
}

/// The attention-glue kernels address their tensors with i32 index math; refuse
/// a launch whose flat element count would wrap. (Production maxima are ~5M.)
fn glue_index_fits_i32(n: usize) -> Result<()> {
    if n > i32::MAX as usize {
        bail!("attn-glue index math overflows i32: {n} elements exceed i32::MAX");
    }
    Ok(())
}

/// Overflow-checked product of size components. The glue-op guards
/// (`glue_index_fits_i32`, table-extent bounds) must see the TRUE mathematical
/// value: an unchecked usize product wraps in release builds, which could carry
/// a wrapped (small) count past the guard. Not reachable from real tensors
/// (candle cannot hold one that large), but the guards should not be
/// circumventable in principle.
fn checked_elems(parts: &[usize], what: &str) -> Result<usize> {
    let mut n = 1usize;
    for &p in parts {
        n = n
            .checked_mul(p)
            .ok_or_else(|| anyhow::anyhow!("{what}: element count {parts:?} overflows usize"))?;
    }
    Ok(n)
}

/// Fused softplus output gate against the `kernel_attn_gate_*` pair
/// (attn_glue.metal): `dst[h,s,d] = attn[h,s,d] * softplus_chain(gate[s,h])`,
/// replacing the 10-dispatch candle chain (softplus + transpose/reshape +
/// broadcast_mul) with one pass over `attn`. `attn` is `[n_head, seq,
/// head_dim]` contiguous, f32 or f16 (the decode path's raw sdpa output — the
/// f16 variant widens in-kernel, exact, so it is bit-identical to `cast_f32` +
/// the f32 variant); `gate` is `[seq, n_head]` f32 contiguous (the g_proj
/// output layout). Output is always f32. The per-op rounding mirrors candle's
/// chain exactly, so the result is bit-identical (see attn_glue.metal / the
/// attn_glue.rs tests).
pub(crate) fn run_attn_gate(attn: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let cdev = attn.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("attn_gate requires attn on a Metal device");
    };

    let (n_head, seq, head_dim) = attn
        .dims3()
        .map_err(|e| anyhow::anyhow!("attn must be rank-3 [n_head, seq, head_dim]: {e}"))?;
    let kernel_name = match attn.dtype() {
        DType::F32 => "kernel_attn_gate_f32",
        DType::F16 => "kernel_attn_gate_f16",
        dt => bail!("attn must be f32 or f16, got {dt:?}"),
    };
    if !attn.is_contiguous() {
        bail!("attn must be contiguous");
    }
    if gate.dims2().map_err(|e| anyhow::anyhow!("gate must be rank-2 [seq, n_head]: {e}"))?
        != (seq, n_head)
    {
        bail!("gate shape {:?} must be [seq, n_head] = [{seq}, {n_head}]", gate.dims());
    }
    if gate.dtype() != DType::F32 {
        bail!("gate must be f32, got {:?}", gate.dtype());
    }
    if !gate.is_contiguous() {
        bail!("gate must be contiguous");
    }
    if !attn.device().same_device(gate.device()) {
        bail!("attn and gate must live on the same Metal device");
    }
    let n = checked_elems(&[n_head, seq, head_dim], "attn_gate")?;
    glue_index_fits_i32(n)?;

    let pipeline = pipelines::attn_glue_pipeline(mdev.device(), kernel_name)?;
    let dst = mdev.new_buffer(n, DType::F32, "attn_gate")?;

    let (attn_guard, attn_layout) = attn.storage_and_layout();
    let Storage::Metal(attn_storage) = &*attn_guard else {
        bail!("attn is not on a Metal device");
    };
    let (gate_guard, gate_layout) = gate.storage_and_layout();
    let Storage::Metal(gate_storage) = &*gate_guard else {
        bail!("gate is not on a Metal device");
    };

    let args = AttnGateArgs {
        n_head: n_head as i32,
        seq: seq as i32,
        head_dim: head_dim as i32,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(
            1,
            Some(attn_storage.buffer()),
            attn_layout.start_offset() * attn.dtype().size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(gate_storage.buffer()),
            gate_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(3, Some(&dst), 0);
        dispatch_linear(encoder, &pipeline, n);
    }
    drop(attn_guard);
    drop(gate_guard);

    Ok(output_tensor(dst, mdev, n, (n_head, seq, head_dim)))
}

/// Fused transpose(0,1)+contiguous with optional dtype conversion, against the
/// `kernel_permute_cast_*` family (attn_glue.metal): `x` `[d0, d1, d2]`
/// contiguous becomes `[d1, d0, d2]` contiguous in ONE pass, converting per
/// `out_dtype` (f32→f32 copy, f32→f16 RTNE, f16→f32 exact — candle's cast
/// scalar). `d0 == 1` degenerates to a plain (optionally casting) copy, which
/// is how the shape-preserving `cast_*` wrappers use it.
pub(crate) fn run_permute_cast(x: &Tensor, out_dtype: DType) -> Result<Tensor> {
    let cdev = x.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("permute_cast requires x on a Metal device");
    };

    let (d0, d1, d2) = x
        .dims3()
        .map_err(|e| anyhow::anyhow!("x must be rank-3 [d0, d1, d2]: {e}"))?;
    if !x.is_contiguous() {
        bail!("x must be contiguous");
    }
    let name = match (x.dtype(), out_dtype) {
        (DType::F32, DType::F32) => "kernel_permute_cast_f32_f32",
        (DType::F32, DType::F16) => "kernel_permute_cast_f32_f16",
        (DType::F16, DType::F32) => "kernel_permute_cast_f16_f32",
        (from, to) => bail!("no permute_cast kernel for {from:?} -> {to:?}"),
    };
    let n = checked_elems(&[d0, d1, d2], "permute_cast")?;
    glue_index_fits_i32(n)?;

    let pipeline = pipelines::attn_glue_pipeline(mdev.device(), name)?;
    let dst = mdev.new_buffer(n, out_dtype, "permute_cast")?;

    let (x_guard, x_layout) = x.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("x is not on a Metal device");
    };

    let args = PermuteArgs {
        d0: d0 as i32,
        d1: d1 as i32,
        d2: d2 as i32,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(
            1,
            Some(x_storage.buffer()),
            x_layout.start_offset() * x.dtype().size_in_bytes(),
        );
        encoder.set_output_buffer(2, Some(&dst), 0);
        dispatch_linear(encoder, &pipeline, n);
    }
    drop(x_guard);

    let storage = MetalStorage::new(dst, mdev.clone(), n, out_dtype);
    Ok(Tensor::from_storage(
        Storage::Metal(storage),
        (d1, d0, d2),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// Vendored partial-rotary NEOX rope against the `kernel_rope_neox_*` pair
/// (rope.metal): rotates the first `n_rot` dims of `x` `[heads, seq, head_dim]`
/// f32 with candle's by-halves rope math and passes the rest through, in ONE
/// read+write of `x` — replacing the narrow/contiguous/rope/cat chain.
/// `out_dtype` picks the store width (f32, or f16 — the rotation still runs in
/// f32 and only the final store rounds, one RTNE rounding, bit-identical to
/// f32-rope + `cast_f16`; pass-through dims round the same way). `cos`/`sin`
/// are the full `[max_ctx, n_rot/2]` f32 tables; `pos` selects the starting
/// row. Bit-identical to the candle chain (see rope.metal / the attn_glue.rs
/// tests).
pub(crate) fn run_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    pos: usize,
    n_rot: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    let cdev = x.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("rope requires x on a Metal device");
    };

    let (heads, seq, head_dim) = x
        .dims3()
        .map_err(|e| anyhow::anyhow!("x must be rank-3 [heads, seq, head_dim]: {e}"))?;
    if x.dtype() != DType::F32 {
        bail!("x must be f32, got {:?}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("x must be contiguous");
    }
    if n_rot == 0 || n_rot % 2 != 0 || n_rot > head_dim {
        bail!("n_rot ({n_rot}) must be even and in 2..=head_dim ({head_dim})");
    }
    let kernel_name = match out_dtype {
        DType::F32 => "kernel_rope_neox_f32",
        DType::F16 => "kernel_rope_neox_f16",
        dt => bail!("rope output dtype must be f32 or f16, got {dt:?}"),
    };
    let half = n_rot / 2;
    // Checked: a caller-supplied pos near usize::MAX must not wrap the row
    // bound (release builds wrap unchecked usize adds).
    let end = pos
        .checked_add(seq)
        .ok_or_else(|| anyhow::anyhow!("rope pos + seq ({pos} + {seq}) overflows usize"))?;
    for (name, t) in [("cos", cos), ("sin", sin)] {
        let (rows, cols) = t
            .dims2()
            .map_err(|e| anyhow::anyhow!("{name} must be rank-2 [max_ctx, n_rot/2]: {e}"))?;
        if cols != half {
            bail!("{name} has {cols} columns, expected n_rot/2 = {half}");
        }
        if end > rows {
            bail!("{name} has {rows} rows, need pos + seq = {end}");
        }
        if t.dtype() != DType::F32 {
            bail!("{name} must be f32, got {:?}", t.dtype());
        }
        if !t.is_contiguous() {
            bail!("{name} must be contiguous");
        }
        if !x.device().same_device(t.device()) {
            bail!("{name} must live on the same Metal device as x");
        }
    }
    let n = checked_elems(&[heads, seq, head_dim], "rope")?;
    glue_index_fits_i32(n)?;
    glue_index_fits_i32(checked_elems(&[end, half], "rope tables")?)?;

    let pipeline = pipelines::rope_pipeline(mdev.device(), kernel_name)?;
    let dst = mdev.new_buffer(n, out_dtype, "rope")?;

    let (x_guard, x_layout) = x.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("x is not on a Metal device");
    };
    let (cos_guard, cos_layout) = cos.storage_and_layout();
    let Storage::Metal(cos_storage) = &*cos_guard else {
        bail!("cos is not on a Metal device");
    };
    let (sin_guard, sin_layout) = sin.storage_and_layout();
    let Storage::Metal(sin_storage) = &*sin_guard else {
        bail!("sin is not on a Metal device");
    };

    let args = RopeArgs {
        heads: heads as i32,
        seq: seq as i32,
        head_dim: head_dim as i32,
        n_rot: n_rot as i32,
        pos: pos as i32,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(
            1,
            Some(x_storage.buffer()),
            x_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(cos_storage.buffer()),
            cos_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            3,
            Some(sin_storage.buffer()),
            sin_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(4, Some(&dst), 0);
        dispatch_linear(encoder, &pipeline, n);
    }
    drop(x_guard);
    drop(cos_guard);
    drop(sin_guard);

    let storage = MetalStorage::new(dst, mdev.clone(), n, out_dtype);
    Ok(Tensor::from_storage(
        Storage::Metal(storage),
        (heads, seq, head_dim),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// The flash prefill kernel's fixed tile geometry (flash.metal): BQ=32 query
/// rows per block, BK=16 key columns per block, head_dim locked at 128.
const FLASH_BQ: usize = 32;
const FLASH_BK: usize = 16;
const FLASH_BD: usize = 128;

/// Vendored flash-attention prefill against the `kernel_flash_attn_*` family
/// (flash.metal — the modified copy of candle's MLX steel attention kernel).
/// `q` is `[n_head, seq, 128]` f32 contiguous (the rope output); `k`/`v` are
/// `[n_kv, K, 128]` f16 cache views that may be HEAD-STRIDED (rows within a
/// head contiguous, head_dim stride 1; the head axis may carry the cache's
/// max_ctx gap — passed to the kernel as explicit strides, never forced
/// contiguous). Masking runs in-kernel: query row i (absolute `pos + i`) sees
/// key column j (absolute `k_off + j`) iff it is not future and within
/// `window` (None = full attention) — exactly `kv_cache::attn_mask_for`'s
/// rule. Returns `[n_head, seq, 128]` f32 contiguous. `disable_skip` defeats
/// the block-level skip bounds (test-only; the skip is exact — see flash.metal).
pub(crate) fn run_flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    pos: usize,
    k_off: usize,
    window: Option<usize>,
    scale: f32,
    disable_skip: bool,
) -> Result<Tensor> {
    let cdev = q.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("flash_attn requires q on a Metal device");
    };

    let (n_head, seq, head_dim) = q
        .dims3()
        .map_err(|e| anyhow::anyhow!("q must be rank-3 [n_head, seq, head_dim]: {e}"))?;
    if q.dtype() != DType::F32 {
        bail!("q must be f32, got {:?}", q.dtype());
    }
    if !q.is_contiguous() {
        bail!("q must be contiguous");
    }
    if head_dim != FLASH_BD {
        bail!("flash_attn is compiled for head_dim {FLASH_BD}, got {head_dim}");
    }
    if seq == 0 {
        bail!("flash_attn requires at least one query row");
    }

    let (n_kv, k_len, _) = check_flash_kv(k, "k", head_dim, &cdev)?;
    let (n_kv_v, v_len, _) = check_flash_kv(v, "v", head_dim, &cdev)?;
    if (n_kv_v, v_len) != (n_kv, k_len) {
        bail!("k [{n_kv}, {k_len}] and v [{n_kv_v}, {v_len}] disagree on shape");
    }
    if n_kv == 0 || !n_head.is_multiple_of(n_kv) {
        bail!("n_head ({n_head}) must be a positive multiple of n_kv ({n_kv})");
    }

    // The mask semantics require every query's own key in range: row i's own
    // key sits at column `pos + i - k_off`, which must lie in [0, K). A row
    // with NO visible key would divide 0/0 in the softmax normalizer.
    if k_off > pos {
        bail!("k_off ({k_off}) exceeds pos ({pos}): query rows before the key range");
    }
    let q_end = pos
        .checked_add(seq)
        .ok_or_else(|| anyhow::anyhow!("pos + seq ({pos} + {seq}) overflows usize"))?;
    let k_end = k_off
        .checked_add(k_len)
        .ok_or_else(|| anyhow::anyhow!("k_off + K ({k_off} + {k_len}) overflows usize"))?;
    if q_end > k_end {
        bail!(
            "query rows reach absolute position {q_end} but keys end at {k_end}: \
             each query's own key must be present"
        );
    }
    // The kernel does its position math in i32.
    for (what, val) in [("pos + seq", q_end), ("k_off + K", k_end)] {
        if i32::try_from(val).is_err() {
            bail!("flash_attn {what} ({val}) overflows the kernel's i32 position math");
        }
    }
    let window = match window {
        None => i32::MAX,
        Some(0) => bail!("flash_attn window must be >= 1"),
        Some(w) => i32::try_from(w).unwrap_or(i32::MAX),
    };

    let out_count = checked_elems(&[n_head, seq, head_dim], "flash_attn")?;
    let dst = mdev.new_buffer(out_count, DType::F32, "flash_attn")?;

    let (q_guard, q_layout) = q.storage_and_layout();
    let Storage::Metal(q_storage) = &*q_guard else {
        bail!("q is not on a Metal device");
    };
    let (k_guard, k_layout) = k.storage_and_layout();
    let Storage::Metal(k_storage) = &*k_guard else {
        bail!("k is not on a Metal device");
    };
    let (v_guard, v_layout) = v.storage_and_layout();
    let Storage::Metal(v_storage) = &*v_guard else {
        bail!("v is not on a Metal device");
    };

    let nq = seq.div_ceil(FLASH_BQ);
    let nk = k_len.div_ceil(FLASH_BK);
    let align_q = seq.is_multiple_of(FLASH_BQ);
    let align_k = k_len.is_multiple_of(FLASH_BK);
    let name = match (align_q, align_k) {
        (true, true) => "kernel_flash_attn_q1_k1",
        (true, false) => "kernel_flash_attn_q1_k0",
        (false, true) => "kernel_flash_attn_q0_k1",
        (false, false) => "kernel_flash_attn_q0_k0",
    };
    let pipeline = pipelines::flash_pipeline(mdev.device(), name)?;

    let i64_stride = |s: usize, what: &str| -> Result<i64> {
        i64::try_from(s).map_err(|_| anyhow::anyhow!("flash_attn {what} stride {s} overflows i64"))
    };
    let args = FlashAttnArgs {
        gqa_factor: (n_head / n_kv) as i32,
        scale,
        nk: nk as i32,
        nq_aligned: (seq / FLASH_BQ) as i32,
        nk_aligned: (k_len / FLASH_BK) as i32,
        ql_rem: (seq % FLASH_BQ) as i32,
        kl_rem: (k_len % FLASH_BK) as i32,
        kl: k_len as i32,
        q_off: pos as i32,
        k_off: k_off as i32,
        window,
        disable_skip: disable_skip as i32,
        q_stride_h: i64_stride(q_layout.stride()[0], "q head")?,
        q_stride_r: i64_stride(q_layout.stride()[1], "q row")?,
        k_stride_h: i64_stride(k_layout.stride()[0], "k head")?,
        k_stride_r: i64_stride(k_layout.stride()[1], "k row")?,
        v_stride_h: i64_stride(v_layout.stride()[0], "v head")?,
        v_stride_r: i64_stride(v_layout.stride()[1], "v row")?,
        o_stride_h: (seq * head_dim) as i64,
        o_stride_r: head_dim as i64,
    };

    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(
            1,
            Some(q_storage.buffer()),
            q_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(k_storage.buffer()),
            k_layout.start_offset() * DType::F16.size_in_bytes(),
        );
        encoder.set_input_buffer(
            3,
            Some(v_storage.buffer()),
            v_layout.start_offset() * DType::F16.size_in_bytes(),
        );
        encoder.set_output_buffer(4, Some(&dst), 0);
        // One threadgroup per (query block, query head); 4 simdgroups.
        let grid = mtl_size!(nq, n_head, 1);
        encoder.dispatch_thread_groups(grid, mtl_size!(32, 4, 1));
    }
    drop(q_guard);
    drop(k_guard);
    drop(v_guard);

    Ok(output_tensor(dst, mdev, out_count, (n_head, seq, head_dim)))
}

/// Validate one flash k/v cache view: rank-3 `[n_kv, K, head_dim]` f16 with
/// head_dim stride 1 and contiguous rows (stride `head_dim`); the head stride
/// is free (the cache's max_ctx gap). Returns (n_kv, K, head stride).
fn check_flash_kv(
    t: &Tensor,
    what: &str,
    head_dim: usize,
    q_device: &Device,
) -> Result<(usize, usize, usize)> {
    let (n_kv, len, hd) = t
        .dims3()
        .map_err(|e| anyhow::anyhow!("{what} must be rank-3 [n_kv, K, head_dim]: {e}"))?;
    if t.dtype() != DType::F16 {
        bail!("{what} must be f16, got {:?}", t.dtype());
    }
    if hd != head_dim {
        bail!("{what} head_dim {hd} != q head_dim {head_dim}");
    }
    if len == 0 {
        bail!("{what} has no keys");
    }
    let stride = t.layout().stride();
    if stride[2] != 1 || stride[1] != head_dim {
        bail!(
            "{what} must have contiguous rows (strides [_, {head_dim}, 1]), got {:?}",
            stride
        );
    }
    if stride[0] < len * head_dim {
        bail!("{what} head stride {} overlaps its {len} rows", stride[0]);
    }
    if !t.device().same_device(q_device) {
        bail!("{what} must live on the same Metal device as q");
    }
    Ok((n_kv, len, stride[0]))
}

#[cfg(test)]
mod combine_guard_tests {
    use super::{combine_index_fits_i32, combine_reduction_width};

    /// The i32 index guard: the combine kernels address `down` with i32 math
    /// (`down_base = s*top_k*n_out + c`), so the grid's flat element count must
    /// stay within i32. Tested directly — the overflowing case (seq ≈ 70k at
    /// top_k=10 / n_out=3072) is a ~8.6TB tensor that cannot be allocated.
    #[test]
    fn index_guard_rejects_i32_overflow() {
        // Production decode/prefill geometry stays well within i32.
        assert!(combine_index_fits_i32(1, 10, 3072));
        assert!(combine_index_fits_i32(4096, 10, 3072)); // 125.8M < 2.1B
        // Just under and just over i32::MAX with a top_k=10 / n_out=3072 row.
        let per_seq = 10 * 3072; // 30720 elements per seq row
        let max_ok = i32::MAX as usize / per_seq; // largest seq that still fits
        assert!(combine_index_fits_i32(max_ok, 10, 3072));
        assert!(!combine_index_fits_i32(max_ok + 1, 10, 3072));
    }

    /// The single-simdgroup width guard threshold: `next_pow2(top_k/2)` must stay
    /// <= 32. Production top_k=10 gives width 8; top_k=66 is the first that needs
    /// width 64 (66/2=33 → next_pow2 64).
    #[test]
    fn reduction_width_threshold() {
        assert_eq!(combine_reduction_width(10), 8);
        assert_eq!(combine_reduction_width(64), 32); // 64/2=32, still one simdgroup
        assert_eq!(combine_reduction_width(65), 32); // 65/2=32
        assert_eq!(combine_reduction_width(66), 64); // 66/2=33 → 64, over the limit
        assert!(combine_reduction_width(66) > 32);
        assert!(combine_reduction_width(10) <= 32);
    }
}

#[cfg(test)]
pub(crate) mod testutil {
    use anyhow::Result;
    use candle_core::quantized::{GgmlDType, QStorage, QTensor};
    use candle_core::{Device, Tensor};
    use std::sync::Arc;

    use crate::gguf::ExpertStack;

    /// Build an expert stack `[n_expert, n_out, k]` on `device` by quantizing a
    /// fixed pseudo-random f32 tensor to `dt`. Returns the stack plus the
    /// dequantized-then-reread weights the kernel effectively sees, so the oracle
    /// compares against the same rounding the kernel does.
    pub(crate) fn build_stack(
        device: &Device,
        dt: GgmlDType,
        n_expert: usize,
        n_out: usize,
        k: usize,
        seed: u64,
    ) -> Result<(ExpertStack, Vec<f32>)> {
        let w = pseudo_random(n_expert * n_out * k, seed, -1.0, 1.0);
        let w_t = Tensor::from_vec(w, (n_expert, n_out, k), device)?;
        let qt = QTensor::quantize(&w_t, dt)?;
        // What the kernel actually multiplies: the quantized weights, dequantized.
        let deq = qt
            .dequantize(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // Mirror the production `expert_stack` load path exactly: upload the
        // quantized bytes once via `from_data`, retain the buffer handle for the
        // fused kernels, then MOVE that storage into the QTensor. The qtensor and
        // the retained `buffer` must share one allocation — if `qtensor` came from
        // a separate `quantize` instead, the shared buffer's only pool reference
        // would hit strong_count 1 and candle's `drop_unused_buffers` (triggered
        // by any readback) would evict it from the residency set, so a later fused
        // dispatch reads a non-resident buffer. (Test-only difference from
        // production: the bytes come from `qt.data()`, not the GGUF file.)
        let storage = QStorage::from_data(qt.data()?, device, dt)?;
        let buffer = match &storage {
            QStorage::Metal(qms) => Some(Arc::new(qms.buffer().clone())),
            _ => None,
        };
        let qtensor = Arc::new(QTensor::new(storage, (n_expert, n_out, k))?);
        let stack = ExpertStack {
            qtensor: Some(qtensor),
            buffer,
            base_off: 0,
            mmap: None,
            dtype: dt,
            n_expert,
            n_out,
            k,
        };
        Ok((stack, deq))
    }

    /// Deterministic reference: for each (token, slot) select expert `ids[token][slot]`,
    /// pick x row `slot` (per-slot) or `0` (shared) per `x_per_row`, and dot each of
    /// the expert's `n_out` rows with it. Layout matches the kernel output
    /// `[t, top_k, n_out]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn oracle(
        deq_weights: &[f32],
        x: &[f32],
        ids: &[u32],
        n_out: usize,
        k: usize,
        t: usize,
        top_k: usize,
        x_per_row: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; t * top_k * n_out];
        for token in 0..t {
            for slot in 0..top_k {
                let e = ids[token * top_k + slot] as usize;
                let x_row = if x_per_row == 1 { 0 } else { slot };
                let x_base = (token * x_per_row + x_row) * k;
                for o in 0..n_out {
                    let w_base = (e * n_out + o) * k;
                    let mut acc = 0f32;
                    for i in 0..k {
                        acc += deq_weights[w_base + i] * x[x_base + i];
                    }
                    out[(token * top_k + slot) * n_out + o] = acc;
                }
            }
        }
        out
    }

    /// Relative L2 error between two equal-length slices.
    pub(crate) fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
        let mut num = 0f64;
        let mut den = 0f64;
        for (g, w) in got.iter().zip(want) {
            num += (*g as f64 - *w as f64).powi(2);
            den += (*w as f64).powi(2);
        }
        (num / den.max(1e-30)).sqrt() as f32
    }

    pub(crate) fn max_abs(got: &[f32], want: &[f32]) -> f32 {
        got.iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max)
    }

    /// Small xorshift so tests do not depend on rand's distributions.
    pub(crate) fn pseudo_random(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
            out.push(lo + (hi - lo) * u as f32);
        }
        out
    }

    pub(crate) fn random_ids(t: usize, top_k: usize, n_expert: usize, seed: u64) -> Vec<u32> {
        let r = pseudo_random(t * top_k, seed, 0.0, n_expert as f32);
        r.into_iter()
            .map(|v| (v as usize % n_expert) as u32)
            .collect()
    }

    /// Ids with `top_k` DISTINCT experts per token — the invariant real top-k
    /// routing always satisfies (argsort top-k never repeats an index). The
    /// two-pass mm_id kernel relies on it: map0 collapses each token's slots for
    /// an expert into one row, so a token selecting the same expert twice would
    /// lose a slot. mv_id has no such requirement, but distinct ids exercise both.
    pub(crate) fn distinct_ids(t: usize, top_k: usize, n_expert: usize, seed: u64) -> Vec<u32> {
        assert!(top_k <= n_expert, "cannot pick {top_k} distinct of {n_expert} experts");
        let mut out = Vec::with_capacity(t * top_k);
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        for _ in 0..t {
            let mut chosen: Vec<u32> = Vec::with_capacity(top_k);
            while chosen.len() < top_k {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let e = (s % n_expert as u64) as u32;
                if !chosen.contains(&e) {
                    chosen.push(e);
                }
            }
            out.extend_from_slice(&chosen);
        }
        out
    }
}
