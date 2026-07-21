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

fn mm_kernel_name(dt: GgmlDType) -> Result<&'static str> {
    let n = match dt {
        GgmlDType::Q4_0 => "kernel_mul_mm_id_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mm_id_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mm_id_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mm_id_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mm_id_q8_0_f32",
        GgmlDType::Q2K => "kernel_mul_mm_id_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mm_id_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mm_id_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mm_id_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mm_id_q6_K_f32",
        GgmlDType::F16 => "kernel_mul_mm_id_f16_f32",
        GgmlDType::F32 => "kernel_mul_mm_id_f32_f32",
        other => bail!("no kernel_mul_mm_id kernel for dtype {other:?}"),
    };
    Ok(n)
}

/// Matrix-tile threadgroup memory the mm kernel always reserves (sa: 4096 half +
/// sb: 4096 float), before the per-launch id-remap scratch.
const MM_TILE_SMEM: usize = 8192;
/// Apple-silicon threadgroup memory ceiling; we refuse a launch that would exceed
/// it rather than let the GPU fault.
const MAX_THREADGROUP_SMEM: usize = 32768;

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
            d.weights,
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

/// Encode `kernel_mul_mm_id_<dtype>_f32` (prefill path). One threadgroup slab per
/// expert (grid.z); the kernel scans the ids buffer into a threadgroup row-id map
/// so each expert's rows are dequantized once for all tokens that selected it.
pub(crate) fn encode_mul_mm_id(
    device: &MetalDevice,
    ep: impl EncoderProvider,
    dt: GgmlDType,
    d: &IdDispatch,
) -> Result<()> {
    let name = mm_kernel_name(dt)?;

    // Worst case every token routes every slot to one expert, so that expert's
    // row-id map (and the column grid) must cover all top_k*t pairs.
    let max_rows = d.top_k * d.t;
    // The row-id map (ushort2 per pair) shares the 32KB threadgroup budget with
    // the 8KB matrix tile, capping top_k*t at (32768-8192)/4 = 6144. WP9 owns the
    // prefill chunk sizing; this documents the invariant the runtime guard below
    // enforces (512 tokens * top_k 10 = 5120 fits).
    debug_assert!(
        max_rows <= (MAX_THREADGROUP_SMEM - MM_TILE_SMEM) / std::mem::size_of::<u32>(),
        "mul_mm_id row-id map for top_k*t={max_rows} exceeds the {MAX_THREADGROUP_SMEM}-byte \
         threadgroup budget (cap {})",
        (MAX_THREADGROUP_SMEM - MM_TILE_SMEM) / std::mem::size_of::<u32>()
    );
    let smem = MM_TILE_SMEM + max_rows * std::mem::size_of::<u32>();
    if smem > MAX_THREADGROUP_SMEM {
        bail!(
            "mul_mm_id needs {smem} bytes of threadgroup memory for top_k*t={max_rows}, \
             over the {MAX_THREADGROUP_SMEM}-byte limit; use mul_mv_id or a smaller prefill chunk"
        );
    }

    let nei0 = d.top_k as i64;
    let nei1 = d.t as i64;
    let nbi1 = (d.top_k * DType::U32.size_in_bytes()) as u64;
    let ne00 = d.k as i64;
    let ne02 = d.n_expert as i64;
    let nb01 = d.bytes_per_row as u64;
    let nb02 = d.per_expert as u64;
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
            d.weights,
            (d.x, d.x_off),
            candle_metal_kernels::Output::new(d.dst),
            (d.ids, d.ids_off),
            nei0,
            nei1,
            nbi1,
            ne00,
            ne02,
            nb01,
            nb02,
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

    encoder.set_threadgroup_memory_length(0, smem);

    // BLOCK_SIZE_N=32 columns (token-slots), BLOCK_SIZE_M=64 rows (n_out), one
    // slab per expert in z.
    let grid = mtl_size!(max_rows.div_ceil(32), d.n_out.div_ceil(64), d.n_expert);
    let threads = mtl_size!(128, 1, 1);
    encoder.dispatch_thread_groups(grid, threads);
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
    /// `kernel_mul_mv_id` — one matvec per (token, slot); decode path.
    Mv,
    /// `kernel_mul_mm_id` — token-grouped matmul; prefill path.
    Mm,
}

/// Validate the seam shapes, resolve every operand to a device buffer, and encode
/// the requested id kernel. Returns the `[t, top_k, n_out]` output tensor.
pub(crate) fn run(stack: &ExpertStack, x: &Tensor, ids: &Tensor, mode: Mode) -> Result<Tensor> {
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
    let dst = mdev.new_buffer(out_count, DType::F32, "mul_id")?;

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
        match mode {
            Mode::Mv => encode_mul_mv_id(mdev, &cmd, dt, &d)?,
            Mode::Mm => encode_mul_mm_id(mdev, &cmd, dt, &d)?,
        }
    }
    drop(x_guard);
    drop(ids_guard);

    Ok(output_tensor(dst, mdev, out_count, (t, top_k, stack.n_out)))
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
        // Mirror the production load path: upload the quantized bytes once and
        // retain the buffer handle for the fused kernels. (Test-only difference:
        // the bytes come from `qt.data()` rather than the GGUF file.)
        let storage = QStorage::from_data(qt.data()?, device, dt)?;
        let buffer = match &storage {
            QStorage::Metal(qms) => Some(Arc::new(qms.buffer().clone())),
            _ => None,
        };
        let stack = ExpertStack {
            qtensor: Arc::new(qt),
            buffer,
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
}
