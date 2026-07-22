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
/// Rust wiring for these and cannot be extended at build time).
const MM_ID_SOURCE: &str = include_str!("mm_id.metal");
/// Vendored ggml-geometry mat-vec kernels (decode expert gather + lm_head).
/// Deliberately separate from `mm_id.metal` so it carries no Metal-4 tensor
/// dependency.
const MV_SOURCE: &str = include_str!("mv.metal");

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
pub(crate) fn mm_id_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, MM_ID_SOURCE, "mm_id", name)
}

/// Pipeline for a `mv.metal` kernel (vendored ggml-geometry mat-vec).
pub(crate) fn mv_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, MV_SOURCE, "mv", name)
}
