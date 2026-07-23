//! Runtime compilation and caching of laguna's vendored Metal kernels.
//!
//! candle's baked metallib is fixed at its build; our vendored kernel sources
//! (`src/ops/*.metal`) are compiled against the live device once each and the
//! resulting pipelines cached by name. `ComputePipeline`/`Library` are
//! `Send + Sync + Clone`, so a process-global cache keyed by the device's
//! registry id serves every dispatch.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use candle_metal_kernels::metal::{ComputePipeline, Device, Library};

/// Vendored kernel source, compiled at runtime (candle's metallib carries no
/// Rust wiring for these and cannot be extended at build time). This is the
/// DEFAULT prefill library; it instantiates every mm_id variant EXCEPT the
/// float-operand cooperative-tensor `_t_hp` one (split into `mm_id_t_hp.metal`),
/// so a toolchain that rejects float `matmul2d` operands cannot break it.
const MM_ID_SOURCE: &str = include_str!("mm_id.metal");
/// The `_t_hp` (float-operand cooperative tensor) instantiations, split out of
/// `mm_id.metal` so the default library carries no float-cooperative-tensor
/// code. Not a standalone translation unit — it references that file's template,
/// typedef and dequant definitions, so it is compiled by concatenating it onto
/// `MM_ID_SOURCE` (see `mm_id_t_hp_source`), and only on first TensorHp dispatch.
const MM_ID_T_HP_INSTANTIATIONS: &str = include_str!("mm_id_t_hp.metal");
/// Vendored ggml-geometry mat-vec kernels (decode expert gather + lm_head).
/// Deliberately separate from `mm_id.metal` so it carries no Metal-4 tensor
/// dependency.
const MV_SOURCE: &str = include_str!("mv.metal");
/// Vendored f16-weight x f32-activation matmul kernels (attention projections).
/// Separate from both `mm_id.metal` (no Metal-4 tensor dependency) and
/// `mv.metal` (attention-critical vs MoE-decode-critical: neither library can
/// break the other).
const F16_SOURCE: &str = include_str!("f16.metal");
/// Vendored Metal-4 cooperative-tensor attention prefill gemm (the tensor
/// analogue of `f16.metal`'s classic `kernel_mul_mm_f16_f32_v`). Separate from
/// `f16.metal` so that file stays Metal-4-free: this library is compiled lazily,
/// only on the first tensor-path dispatch (`f16_t_pipeline`), mirroring the
/// `mm_id_t_hp` split. The tensor path is opt-in (`LAGUNA_ATTN_MM_TENSOR`); with
/// the classic default the mm branch never asks for this library, so it never
/// compiles.
const F16_T_SOURCE: &str = include_str!("f16_t.metal");
/// PROBE: the mixed-operand (half weight tile x FLOAT activation tile)
/// cooperative-tensor variant of `f16_t.metal`. Test-only reachability
/// (`run_matmul_f16_variant`'s `TensorMixed` arm); its own lazily compiled
/// library — the `mm_id_t_hp` isolation pattern — so a toolchain that rejects
/// mixed-operand `matmul2d` fails only this probe, never the default or the
/// half-tile tensor library.
const F16_T_MIXED_SOURCE: &str = include_str!("f16_t_mixed.metal");
/// Vendored fused MoE weighted-combine kernels (the routed-expert combine tail).
/// Own library (no Metal-4 dependency); compiled with FP contraction disabled so
/// its per-op rounding stays bit-identical to the candle broadcast/affine/sum
/// chain it replaces (see combine.metal).
const COMBINE_SOURCE: &str = include_str!("combine.metal");
/// Vendored fused attention-glue kernels (softplus gate + permute/cast copies).
/// Own library (no Metal-4 dependency); compiled with FP contraction disabled so
/// its per-op rounding stays bit-identical to the candle elementwise/copy chains
/// it replaces (see attn_glue.metal).
const ATTN_GLUE_SOURCE: &str = include_str!("attn_glue.metal");
/// Vendored NEOX rope kernel with internal partial rotary. A SEPARATE library
/// from attn_glue.metal because that file's fp pragmas are file-scoped and the
/// rope rotation must instead compile under the same default math mode as
/// candle's own rope kernel to stay bit-identical (see rope.metal).
const ROPE_SOURCE: &str = include_str!("rope.metal");

/// The concatenated source for the TensorHp library: the shared mm_id template
/// portion plus the split-out `_t_hp` instantiations. Built once on first use,
/// so the (potentially unsupported) float-cooperative-tensor code is only ever
/// handed to the Metal compiler when TensorHp is actually selected.
fn mm_id_t_hp_source() -> &'static str {
    static SRC: OnceLock<String> = OnceLock::new();
    SRC.get_or_init(|| format!("{MM_ID_SOURCE}\n{MM_ID_T_HP_INSTANTIATIONS}"))
}

struct Cache {
    /// One compiled library per (device registry id, source key).
    libraries: HashMap<(u64, &'static str), Library>,
    /// Pipelines keyed by (device registry id, function name). Function names
    /// are unique across our sources, so the source key is not part of the key.
    pipelines: HashMap<(u64, String), ComputePipeline>,
}

fn cache() -> &'static Mutex<Cache> {
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        Mutex::new(Cache {
            libraries: HashMap::new(),
            pipelines: HashMap::new(),
        })
    })
}

/// Fetch (compiling and caching on first use) the compute pipeline for `name`
/// from vendored `source` (labelled `source_key` for the library cache) on
/// `device`.
fn compiled_pipeline(
    device: &Device,
    source: &str,
    source_key: &'static str,
    name: &str,
) -> Result<ComputePipeline> {
    // The vendored kernels' bit-identity contracts assume candle's kernels
    // compile in candle's default fast-math mode; our sources are pinned
    // `math_mode(fast)`. A falsy CANDLE_METAL_ENABLE_FAST_MATH would move
    // candle to Relaxed/Precise while ours stay fast — a silent break of every
    // bitwise contract, so refuse to run rather than warn (the same
    // fail-closed stance the parity provenance system takes).
    if crate::ops::candle_fast_math_disabled() {
        anyhow::bail!(
            "CANDLE_METAL_ENABLE_FAST_MATH is set falsy: candle would compile its Metal kernels \
             Relaxed/Precise while laguna's vendored libraries are pinned math_mode(fast), \
             silently breaking their bitwise-identity contracts (combine/attn_glue/rope). \
             Unset the variable — the two compile modes cannot be mixed."
        );
    }
    let key = device.registry_id();
    let mut cache = cache().lock().unwrap();

    if let Some(p) = cache.pipelines.get(&(key, name.to_string())) {
        return Ok(p.clone());
    }

    let lib_key = (key, source_key);
    if !cache.libraries.contains_key(&lib_key) {
        let lib = device
            .new_library_with_source(source, None)
            .map_err(|e| anyhow::anyhow!("compiling vendored {source_key}.metal: {e}"))?;
        cache.libraries.insert(lib_key, lib);
    }
    let lib = &cache.libraries[&lib_key];

    let func = lib
        .get_function(name, None)
        .map_err(|e| anyhow::anyhow!("locating `{name}` in vendored {source_key}.metal: {e}"))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .with_context(|| format!("building pipeline for `{name}`"))?;

    cache
        .pipelines
        .insert((key, name.to_string()), pipeline.clone());
    Ok(pipeline)
}

/// Pipeline for a `mm_id.metal` kernel (two-pass indexed matmul).
///
/// The float-operand cooperative-tensor `_t_hp` kernels live in a separate
/// library, compiled lazily by concatenating `mm_id_t_hp.metal` onto the shared
/// source. Routing on the kernel name keeps the default library free of
/// float-cooperative-tensor code: a `_t_hp` compile failure surfaces only here,
/// on the TensorHp path, and never touches the default prefill library.
pub(crate) fn mm_id_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    if name.ends_with("_t_hp") {
        compiled_pipeline(device, mm_id_t_hp_source(), "mm_id_t_hp", name)
    } else {
        compiled_pipeline(device, MM_ID_SOURCE, "mm_id", name)
    }
}

/// Pipeline for a `mv.metal` kernel (vendored ggml-geometry mat-vec).
pub(crate) fn mv_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, MV_SOURCE, "mv", name)
}

/// Pipeline for an `f16.metal` kernel (vendored f16-weight attention matmul).
pub(crate) fn f16_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, F16_SOURCE, "f16", name)
}

/// Pipeline for an `f16_t.metal` kernel (Metal-4 cooperative-tensor attention
/// prefill gemm). Its own library, compiled lazily on first use so the classic
/// `f16.metal` library carries no Metal-4 dependency.
pub(crate) fn f16_t_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, F16_T_SOURCE, "f16_t", name)
}

/// Pipeline for an `f16_t_mixed.metal` kernel (the mixed-operand matmul2d
/// probe). Own library, compiled lazily on first (test-only) dispatch.
pub(crate) fn f16_t_mixed_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, F16_T_MIXED_SOURCE, "f16_t_mixed", name)
}

/// Pipeline for a `combine.metal` kernel (vendored fused MoE weighted combine).
pub(crate) fn combine_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, COMBINE_SOURCE, "combine", name)
}

/// Pipeline for an `attn_glue.metal` kernel (fused softplus gate / permute-cast).
pub(crate) fn attn_glue_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, ATTN_GLUE_SOURCE, "attn_glue", name)
}

/// Pipeline for a `rope.metal` kernel (vendored partial-rotary NEOX rope).
pub(crate) fn rope_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, ROPE_SOURCE, "rope", name)
}
