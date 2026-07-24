use std::fmt::Write as _;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result, bail, ensure};
use candle_core::quantized::gguf_file::Content;
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, MetalDevice, MetalStorage, Module, Storage, Tensor};
use candle_metal_kernels::metal::Buffer;
use candle_nn::RmsNorm;

/// `LAGUNA_LOAD_CLASSIC` reverts the model load from the default mmap aliasing
/// (the big weights — expert stacks, f16 attention planes — alias the GGUF's
/// page cache through no-copy Metal buffer views, so a warm load takes seconds)
/// to the legacy full-copy load (read every tensor into a Vec, upload via
/// `QStorage::from_data`). Consulted once at `open`: a classic-opened `GgufFile`
/// carries no mapping, so every downstream loader copies.
///
/// PRESENCE-BASED and cached (read once), like the sibling `ops::*` switches
/// (`flash_classic`, `attn_glue_classic`): any value enables it — only leaving
/// it unset keeps the mmap path.
pub fn load_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("LAGUNA_LOAD_CLASSIC").is_some())
}

unsafe extern "C" {
    /// Mach's exported page-size global (libSystem, always linked): 16384 on
    /// Apple silicon, but read at runtime rather than hardcoded.
    /// `newBufferWithBytesNoCopy` requires page-aligned pointers and
    /// page-multiple lengths, so `MmapSource::view` floors/ceils with this.
    safe static vm_page_size: usize;
}

/// The GGUF file mapped read-only, plus the raw Metal device and residency set
/// its no-copy buffer views hang off. Built once per `open` on Metal (unless
/// `LAGUNA_LOAD_CLASSIC`) and shared by every aliased tensor.
///
/// LIFETIME INVARIANT: the mapping must outlive every Metal view buffer created
/// over it. Views are created with `deallocator: None`, so dropping a `Buffer`
/// never unmaps — but dropping this struct DOES (memmap2 unmaps on drop), which
/// would leave a live view (and the GPU) reading unmapped pages. Everything
/// holding an aliased view therefore keeps a clone of the owning
/// `Arc<MmapSource>`: `ExpertStack.mmap` per stack, and `LagunaModel` holds one
/// for the aliased attention tensors (whose `Tensor`s cannot carry it).
pub struct MmapSource {
    map: memmap2::Mmap,
    /// The candle Metal device views are created on. View buffers are batch-
    /// registered in ITS queue-attached residency set (the vendored candle
    /// patch `register_external_buffers`), so aliased weights stay permanently
    /// GPU-resident exactly like candle's own pool buffers — without this the
    /// 70GB working set pays per-command-buffer residency bookkeeping,
    /// measured at ~10% of sustained 4k prefill (docs/log.md mmap entry).
    /// Residency is perf-only: setBuffer-bound buffers are made resident per
    /// command buffer regardless, so unregistered views compute correctly.
    mdev: MetalDevice,
    /// Views created but not yet registered resident; drained into
    /// `registered` by `register_views` (LagunaModel::load calls it once after
    /// all weights are built — one batch, one residency-set commit).
    pending: Mutex<Vec<Arc<Buffer>>>,
    /// Views currently registered in the device's residency set. Drop
    /// unregisters them — the set RETAINS its allocations, so without this a
    /// dumped model would leave every view's MTLBuffer (and its GPU mapping)
    /// alive forever. Load→drop cycles (e.g. a serve-then-unload server) are
    /// leak-free: drop order guarantees this struct outlives all view holders,
    /// so by the time Drop runs the set holds the only remaining retains.
    registered: Mutex<Vec<Arc<Buffer>>>,
}

impl Drop for MmapSource {
    fn drop(&mut self) {
        // Quiesce before unmapping: an in-flight command buffer retains the
        // view MTLBuffers it has bound, but a buffer being alive does NOT keep
        // the underlying pages mapped (`deallocator: None`) — the munmap when
        // `self.map` drops below would yank pages the GPU may still be
        // reading. Errors are ignored: teardown must not panic.
        let _ = self.mdev.wait_until_completed();
        let registered = self.registered.get_mut().unwrap_or_else(|e| e.into_inner());
        self.mdev.unregister_external_buffers(registered.iter().map(|b| b.as_ref()));
    }
}

impl MmapSource {
    /// Maps `path` read-only for aliasing on `device` (must be Metal).
    pub fn open(path: &Path, device: &Device) -> Result<Arc<Self>> {
        let Device::Metal(mdev) = device else {
            bail!("mmap aliasing requires a Metal device");
        };
        let file = File::open(path).with_context(|| format!("opening {} for mmap", path.display()))?;
        // SAFETY: the mapping is read-only, and the GGUF file being truncated or
        // rewritten under a running process is out of contract (the same
        // assumption llama.cpp's mmap loader makes).
        let map = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("mmapping {}", path.display()))?;
        // Cheap prefetch hint; harmless if the kernel ignores it.
        let _ = map.advise(memmap2::Advice::WillNeed);
        Ok(Arc::new(Self {
            map,
            mdev: mdev.clone(),
            pending: Mutex::new(Vec::new()),
            registered: Mutex::new(Vec::new()),
        }))
    }

    /// One no-copy Metal buffer aliasing bytes `[abs_off, abs_off + len)` of the
    /// mapping: page-floored start, page-ceiled length (Metal requires both),
    /// candle's `RESOURCE_OPTIONS` (Shared + hazard-untracked, same as every
    /// candle allocation so the encoder fence discipline stays uniform).
    /// Returns the buffer plus `base_off`, the tensor's byte offset inside the
    /// view — always < page size, and 32-byte aligned because GGUF aligns
    /// tensor data to 32. Overlapping views over one mapping are legal (the
    /// fork's giant per-file views rely on the same property).
    fn view(&self, abs_off: usize, len: usize) -> Result<(Arc<Buffer>, usize)> {
        ensure!(
            abs_off.checked_add(len).is_some_and(|end| end <= self.map.len()),
            "mmap view [{abs_off}, +{len}) exceeds the {}-byte mapping",
            self.map.len()
        );
        let page = vm_page_size;
        let start = abs_off / page * page;
        let base_off = abs_off - start;
        // Page-ceiled: the kernel maps whole pages, so a tail past EOF inside
        // the last page is mapped (zero-filled) and safe to cover.
        let view_len = (base_off + len).div_ceil(page) * page;
        let ptr = std::ptr::NonNull::new(unsafe { self.map.as_ptr().add(start) } as *mut std::ffi::c_void)
            .context("mmap base pointer is null")?;
        // SAFETY: `ptr` is page-aligned inside the mapping and `view_len` is a
        // page multiple; the bytes stay valid as long as `self.map` lives (the
        // Arc<MmapSource> lifetime invariant above). `deallocator: None` means
        // Metal never frees the pages — unmapping stays the Mmap drop's job.
        let raw = unsafe {
            use objc2_metal::MTLDevice as _;
            self.mdev.device().as_ref().newBufferWithBytesNoCopy_length_options_deallocator(
                ptr,
                view_len,
                candle_metal_kernels::RESOURCE_OPTIONS,
                None,
            )
        }
        .with_context(|| format!("newBufferWithBytesNoCopy failed for {view_len} bytes at {start}"))?;
        let buffer = Arc::new(Buffer::new(raw));
        // Collected for batch residency registration — see `pending`/
        // `register_views`. Per-view registration (a synchronous residency-set
        // commit each) measured ~7s of load across the 381 views.
        self.pending.lock().unwrap_or_else(|e| e.into_inner()).push(buffer.clone());
        Ok((buffer, base_off))
    }

    /// Registers every not-yet-registered view in the device's queue-attached
    /// residency set, one batch + one commit. LagunaModel::load calls this
    /// once after all weights are built; Drop unregisters everything this
    /// registered.
    pub fn register_views(&self) {
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        let mut registered = self.registered.lock().unwrap_or_else(|e| e.into_inner());
        self.mdev.register_external_buffers(pending.iter().map(|b| b.as_ref()));
        registered.append(&mut pending);
    }
}

/// An opened GGUF: parsed header plus the file handle tensors are read from,
/// and (on Metal, unless `LAGUNA_LOAD_CLASSIC`) the whole file mapped for the
/// no-copy alias load path.
pub struct GgufFile {
    pub content: Content,
    pub device: Device,
    pub path: PathBuf,
    file: Mutex<File>,
    mmap: Option<Arc<MmapSource>>,
}

impl GgufFile {
    /// The alias-load mapping, present on Metal unless `LAGUNA_LOAD_CLASSIC`.
    /// A holder of aliased weights whose tensors cannot carry the Arc
    /// themselves (the attention planes) must clone this and keep it alive —
    /// see `MmapSource`'s lifetime invariant.
    pub fn mmap_source(&self) -> Option<&Arc<MmapSource>> {
        self.mmap.as_ref()
    }
}

pub fn metal_device() -> Result<Device> {
    Ok(Device::new_metal(0)?)
}

pub fn open(path: impl AsRef<Path>, device: &Device) -> Result<Arc<GgufFile>> {
    let path = path.as_ref();
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let content = Content::read(&mut file).with_context(|| format!("parsing GGUF {}", path.display()))?;
    let mmap = if matches!(device, Device::Metal(_)) && !load_classic() {
        Some(MmapSource::open(path, device)?)
    } else {
        None
    };
    Ok(Arc::new(GgufFile {
        content,
        device: device.clone(),
        path: path.to_path_buf(),
        file: Mutex::new(file),
        mmap,
    }))
}

/// A quantized (or dense — QMatMul dequantizes F32/F16 sources) linear layer.
pub struct QLinear {
    inner: QMatMul,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl QLinear {
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // candle's Metal quantized matmul rebuilds the input layout from its
        // SHAPE (quantized/metal.rs, call_quantized_matmul_mm_t) and so silently
        // drops any storage start_offset — a dim-0 narrowed view reads the wrong
        // rows. `.contiguous()` cannot repair that (an offset-only view still
        // passes the contiguity check and no-ops), so genuinely materialize such
        // inputs via the zeros_like + slice_set blit (Tensor::copy is a shallow
        // Arc clone on Metal — see CLAUDE.md).
        let x = if !x.is_contiguous() { x.contiguous()? } else { x.clone() };
        let x = if x.layout().start_offset() != 0 {
            let out = x.zeros_like()?;
            out.slice_set(&x, 0, 0)?;
            out
        } else {
            x
        };
        self.inner.forward(&x)
    }

    /// Wraps an already-loaded rank-2 weight `[out_dim, in_dim]` as a linear layer,
    /// for callers holding a QTensor directly rather than a GGUF tensor name.
    pub fn from_qtensor(qt: Arc<QTensor>) -> Result<Self> {
        let dims = qt.shape().dims().to_vec();
        let [out_dim, in_dim] = dims[..] else {
            bail!("QLinear source is not a rank-2 weight: {dims:?}");
        };
        Ok(QLinear { inner: QMatMul::from_arc(qt)?, in_dim, out_dim })
    }
}

/// Stacked per-expert weights kept in their quantized GGUF layout,
/// `[n_expert, n_out, k]`, whose device buffer the `ops::{mv_id,mm_id}`
/// kernels index directly by expert id.
pub struct ExpertStack {
    /// The stack as a QTensor, on the classic copying load. `None` on the mmap
    /// alias load — the fused Metal kernels read only `buffer` plus the
    /// shape/dtype fields, and every consumer that needs real QTensors
    /// (`ReferenceExperts`, `split_expert_stack`) loads its own via
    /// `expert_qtensors`, which always copies.
    pub qtensor: Option<Arc<QTensor>>,
    /// The stack's quantized bytes as a raw device buffer, for the fused
    /// `ops::{mv_id,mm_id}` kernels. Classic load: a retained handle to the
    /// SAME `MTLBuffer` that backs `qtensor` (both were cloned from one
    /// `QStorage`), so the fused path indexes the resident weights with no
    /// second upload. Mmap load: a page-floored no-copy view over the GGUF
    /// mapping (see `MmapSource::view`). `None` off Metal — the Reference
    /// runner uses `expert_qtensors`.
    pub buffer: Option<Arc<Buffer>>,
    /// Byte offset of the stack's first block inside `buffer`: 0 on the classic
    /// path (dedicated allocation), the sub-page remainder of the tensor's file
    /// offset on the mmap path (< page size, 32-byte aligned per GGUF data
    /// alignment). Every kernel dispatch consuming `buffer` must add it
    /// (dispatch.rs `IdDispatch.w_off`).
    pub base_off: usize,
    /// Keeps the file mapping (and its residency set) alive while `buffer`
    /// aliases it — `MmapSource`'s lifetime invariant. `None` on the classic
    /// path.
    pub mmap: Option<Arc<MmapSource>>,
    pub dtype: GgmlDType,
    pub n_expert: usize,
    pub n_out: usize,
    pub k: usize,
}

/// The vendored q8_0 attention decode gemv covers output rows in `N_R0`-row
/// groups (`kernel_mul_mv_q8_0_f32_attn`, N_R0_Q8_0 = 2 in ggml-metal-impl.h;
/// mirrored by `ops`'s `MV_Q8_NR0`). ggml guards only the STORE against a ragged
/// final group — the COMPUTE reads the whole `N_R0`-row group — so the classic
/// private weight copy is padded up to a whole multiple of this many rows.
const Q8_DECODE_NR0: usize = 2;

/// The raw q8_0 bytes of one attention projection weight `[out_dim, in_dim]`,
/// as a device buffer for the vendored decode gemv (`ops::matmul_q8`). Loaded
/// ONLY for a checkpoint that stores its attention weights q8_0 (the unsloth
/// UD-Q4_K_XL file); the dense f16 plane that carries the prefill/mm path lives
/// alongside it in `Proj`. Mmap load: a page-floored no-copy view over the GGUF
/// mapping, with `base_off` the tensor's byte offset inside the view. Classic
/// load: a dedicated private buffer (`base_off` 0).
pub struct AttnQ8 {
    /// The q8_0 bytes as a raw device buffer (aliased view or private copy).
    pub buffer: Arc<Buffer>,
    /// Byte offset of the weight's first block inside `buffer`: 0 on the classic
    /// path, the sub-page remainder of the tensor's file offset on the mmap path
    /// (< page size, 32-byte aligned per GGUF data alignment). The decode gemv
    /// dispatch binds `buffer` at this offset.
    pub base_off: usize,
    /// Keeps the file mapping (and its residency set) alive while `buffer`
    /// aliases it — `MmapSource`'s lifetime invariant. `None` on the classic path.
    pub mmap: Option<Arc<MmapSource>>,
    pub out_dim: usize,
    pub in_dim: usize,
}

/// VarBuilder-shaped accessor: `w.pp("blk.0").qlinear("attn_q")` reads `blk.0.attn_q.weight`.
#[derive(Clone)]
pub struct Weights {
    src: Arc<GgufFile>,
    prefix: String,
}

impl Weights {
    pub fn from_gguf(src: Arc<GgufFile>) -> Self {
        Self { src, prefix: String::new() }
    }

    pub fn pp(&self, p: impl AsRef<str>) -> Weights {
        let prefix = if self.prefix.is_empty() {
            p.as_ref().to_string()
        } else {
            format!("{}.{}", self.prefix, p.as_ref())
        };
        Self { src: self.src.clone(), prefix }
    }

    pub fn device(&self) -> &Device {
        &self.src.device
    }

    pub fn has(&self, name: &str) -> bool {
        self.src.content.tensor_infos.contains_key(&self.name(name))
    }

    fn name(&self, n: &str) -> String {
        if self.prefix.is_empty() { format!("{n}.weight") } else { format!("{}.{n}.weight", self.prefix) }
    }

    pub fn qtensor(&self, name: &str) -> Result<Arc<QTensor>> {
        let full = self.name(name);
        let mut file = self.src.file.lock().unwrap();
        let qt = self
            .src
            .content
            .tensor(&mut *file, &full, &self.src.device)
            .with_context(|| format!("loading tensor {full}"))?;
        Ok(Arc::new(qt))
    }

    pub fn qlinear(&self, name: &str) -> Result<QLinear> {
        let qt = self.qtensor(name)?;
        let dims = qt.shape().dims().to_vec();
        let [out_dim, in_dim] = dims[..] else {
            bail!("{} is not a rank-2 weight: {dims:?}", self.name(name));
        };
        Ok(QLinear { inner: QMatMul::from_arc(qt)?, in_dim, out_dim })
    }

    /// Loads a rank-2 weight `[out_dim, in_dim]` as a `QLinear`, additionally
    /// returning a retained handle to its Metal buffer (or `None` off Metal) so a
    /// caller can dispatch the vendored plain mat-vec kernel over the SAME
    /// allocation the `QLinear` uses — no second upload. Same zero-copy
    /// construction as `expert_stack`: read the quantized bytes ourselves, upload
    /// once via `QStorage::from_data`, retain the buffer BEFORE the storage moves
    /// into the `QTensor`. Used for the lm_head (q6_K) decode bypass; a non-Metal
    /// device yields `buffer = None` and the caller stays on `QLinear::forward`.
    pub fn qlinear_with_buffer(&self, name: &str) -> Result<(QLinear, Option<Arc<Buffer>>, GgmlDType)> {
        let full = self.name(name);
        let info = self
            .src
            .content
            .tensor_infos
            .get(&full)
            .with_context(|| format!("tensor {full} not found"))?;
        let dims = info.shape.dims().to_vec();
        let [out_dim, in_dim] = dims[..] else {
            bail!("{full} is not a rank-2 weight: {dims:?}");
        };
        let dtype = info.ggml_dtype;
        let block = dtype.block_size();
        let elems = out_dim * in_dim;
        if !elems.is_multiple_of(block) {
            bail!("{full}: {elems} elements not a multiple of {dtype:?} block size {block}");
        }
        let size_in_bytes = elems / block * dtype.type_size();
        let tensor_start = self.src.content.tensor_data_offset + info.offset;

        let mut raw = vec![0u8; size_in_bytes];
        {
            let mut file = self.src.file.lock().unwrap();
            file.seek(SeekFrom::Start(tensor_start))
                .with_context(|| format!("seeking to {full}"))?;
            file.read_exact(&mut raw)
                .with_context(|| format!("reading {full} ({size_in_bytes} bytes)"))?;
        }

        let storage = QStorage::from_data(std::borrow::Cow::Owned(raw), &self.src.device, dtype)?;
        let buffer = match &storage {
            QStorage::Metal(qms) => Some(Arc::new(qms.buffer().clone())),
            _ => None,
        };
        let qtensor = Arc::new(QTensor::new(storage, (out_dim, in_dim))?);
        let qlinear = QLinear { inner: QMatMul::from_arc(qtensor)?, in_dim, out_dim };
        Ok((qlinear, buffer, dtype))
    }

    pub fn rms_norm(&self, name: &str, eps: f64) -> Result<RmsNorm> {
        let w = self.dense_f32(name)?;
        Ok(RmsNorm::new(w, eps))
    }

    /// A small tensor needed densely on-device (norm weights, router, exp_probs_b),
    /// dequantized to f32 whatever its stored dtype.
    pub fn dense_f32(&self, name: &str) -> Result<Tensor> {
        let qt = self.qtensor(name)?;
        Ok(qt.dequantize(&self.src.device)?.to_dtype(candle_core::DType::F32)?)
    }

    /// A rank-2 weight `[out_dim, in_dim]` as a dense f16 tensor, for layers
    /// whose matmuls run natively in f16 (the GGUF stores the attention weights
    /// as F16, so this keeps them at their stored precision — `QMatMul` would
    /// dequantize them to f32 and double the streamed bytes).
    ///
    /// On the mmap load an F16-stored weight ALIASES the file's page cache
    /// (`f16_alias_tensor`) instead of being read + re-uploaded: the GGUF F16
    /// bytes ARE the dense plane, and the classic `dequantize_f16` round-trip
    /// (f16→f32→f16) is exact, so the two paths are bit-identical. Any other
    /// stored dtype (or a classic open) takes the copying path, whose f32
    /// intermediate is dropped before this returns; no f32 copy stays alive.
    pub fn dense_f16(&self, name: &str) -> Result<Tensor> {
        if let Some(src) = self.src.mmap.as_ref() {
            let full = self.name(name);
            let info = self
                .src
                .content
                .tensor_infos
                .get(&full)
                .with_context(|| format!("tensor {full} not found"))?;
            if info.ggml_dtype == GgmlDType::F16 {
                let dims = info.shape.dims().to_vec();
                let [out_dim, in_dim] = dims[..] else {
                    bail!("{full} is not a rank-2 weight: {dims:?}");
                };
                let tensor_start = (self.src.content.tensor_data_offset + info.offset) as usize;
                return f16_alias_tensor(src, &self.src.device, tensor_start, out_dim, in_dim)
                    .with_context(|| format!("mmap-aliasing {full}"));
            }
        }
        let qt = self.qtensor(name)?;
        let t = qt.dequantize_f16(&self.src.device)?;
        let dims = t.dims().to_vec();
        let [_out_dim, _in_dim] = dims[..] else {
            bail!("{} is not a rank-2 weight: {dims:?}", self.name(name));
        };
        Ok(t)
    }

    /// One attention projection weight `[out_dim, in_dim]`: the dense f16 plane
    /// (`dense_f16` — the prefill/mm path) plus, ONLY when the GGUF stores the
    /// weight as Q8_0, the raw q8_0 bytes as an `AttnQ8` for the vendored decode
    /// gemv (`ops::matmul_q8`). An F16-stored weight (the official checkpoint)
    /// returns `None` for the q8 handle and stays on the f16 path everywhere —
    /// byte-identical to today, no extra load work.
    ///
    /// The f16 plane and the q8_0 bytes are two views of the SAME weight: the plane
    /// is `dequantize_f16` of the q8_0 tensor, the alias is the untouched q8_0
    /// bytes. The two decode paths across the `Q8_DECODE_MAX_SEQ` seq boundary are
    /// NOT bit-identical: the f16 plane rounds each dequantized value `d·q_i` to
    /// f16 (one extra rounding per weight element) and the seq > boundary path
    /// multiplies those f16 weights by the f32 activation, whereas the seq <=
    /// boundary q8 gemv multiplies the raw int8 quants by `d` and accumulates in
    /// f32 with no such rounding. Both inherit q8_0's quantization error, so they
    /// agree to that plus the f16 plane's per-element rounding — the same
    /// numerical class as the f16 path's OWN gemv/gemm split (already ulp-different
    /// across its seq boundary). The discontinuity is accepted design: the decode
    /// parity tier grades greedy/perplexity statistically, not bitwise.
    pub fn attn_proj(&self, name: &str) -> Result<(Tensor, Option<AttnQ8>)> {
        let f16 = self.dense_f16(name)?;
        let full = self.name(name);
        let info = self
            .src
            .content
            .tensor_infos
            .get(&full)
            .with_context(|| format!("tensor {full} not found"))?;
        if info.ggml_dtype != GgmlDType::Q8_0 {
            return Ok((f16, None));
        }
        let dims = info.shape.dims().to_vec();
        let [out_dim, in_dim] = dims[..] else {
            bail!("{full} is not a rank-2 weight: {dims:?}");
        };
        let dtype = info.ggml_dtype;
        let block = dtype.block_size();
        let elems = out_dim * in_dim;
        if !elems.is_multiple_of(block) {
            bail!("{full}: {elems} elements not a multiple of {dtype:?} block size {block}");
        }
        let size_in_bytes = elems / block * dtype.type_size();
        let tensor_start = (self.src.content.tensor_data_offset + info.offset) as usize;

        let (buffer, base_off, mmap) = match self.src.mmap.as_ref() {
            Some(src) => {
                let (buffer, base_off) = src
                    .view(tensor_start, size_in_bytes)
                    .with_context(|| format!("mmap-aliasing {full}"))?;
                // The gemv walks whole q8_0 blocks (half delta + int8 quants), so
                // the bound offset needs 2-byte alignment; every real GGUF is
                // 32-aligned (this guards hand-crafted files).
                ensure!(
                    base_off.is_multiple_of(2),
                    "{full}: mmap base_off {base_off} is not 2-byte aligned"
                );
                (buffer, base_off, Some(src.clone()))
            }
            None => {
                // Classic copy: read the raw q8_0 bytes into a fresh private buffer
                // (base_off 0). The dense f16 plane above is an independent
                // allocation on this path, so no allocation is shared.
                //
                // The decode gemv reads output rows in Q8_DECODE_NR0-row groups and
                // (ggml convention) reads the whole group even when the final one is
                // ragged — only the STORE is row-guarded. `new_buffer_with_data`
                // allocates exactly the data length (the mmap view is page-padded,
                // this is not), so an odd out_dim would put the last group's second
                // row past the buffer. Pad up to a whole Q8_DECODE_NR0-row multiple
                // with zeros; the padding rows are read and discarded by the guard.
                let bytes_per_row = in_dim / block * dtype.type_size();
                let padded_rows = out_dim.div_ceil(Q8_DECODE_NR0) * Q8_DECODE_NR0;
                let mut raw = vec![0u8; padded_rows * bytes_per_row];
                {
                    let mut file = self.src.file.lock().unwrap();
                    file.seek(SeekFrom::Start(tensor_start as u64))
                        .with_context(|| format!("seeking to {full}"))?;
                    file.read_exact(&mut raw[..size_in_bytes])
                        .with_context(|| format!("reading {full} ({size_in_bytes} bytes)"))?;
                }
                let Device::Metal(mdev) = &self.src.device else {
                    bail!("q8_0 attention weights require a Metal device");
                };
                let buffer = mdev.new_buffer_with_data(&raw)?;
                (buffer, 0usize, None)
            }
        };
        Ok((f16, Some(AttnQ8 { buffer, base_off, mmap, out_dim, in_dim })))
    }

    /// Loads a stacked expert tensor `[n_expert, n_out, k]` for the fused MoE
    /// kernels. Default (mmap open): the quantized bytes stay in the file's
    /// page cache and the kernels read them through a no-copy view. Classic
    /// open: read + upload once, sharing the allocation with a wrapping
    /// QTensor.
    pub fn expert_stack(&self, name: &str) -> Result<ExpertStack> {
        match self.src.mmap.as_ref() {
            Some(src) => self.expert_stack_mmap(name, src),
            None => self.expert_stack_classic(name),
        }
    }

    /// The classic copying load: the fused MoE kernels and the wrapping
    /// `QTensor` share ONE device allocation. We read the quantized bytes from
    /// the file ourselves, upload them once via `QStorage::from_data`, retain a
    /// handle to that buffer, and only then wrap the same storage in a
    /// `QTensor` — so no second copy of the (large) expert weights ever lands
    /// in VRAM.
    fn expert_stack_classic(&self, name: &str) -> Result<ExpertStack> {
        let full = self.name(name);
        let info = self
            .src
            .content
            .tensor_infos
            .get(&full)
            .with_context(|| format!("expert stack tensor {full} not found"))?;
        let dims = info.shape.dims().to_vec();
        let [n_expert, n_out, k] = dims[..] else {
            bail!("{full} is not a rank-3 expert stack: {dims:?}");
        };
        let dtype = info.ggml_dtype;
        let block = dtype.block_size();
        let elems = n_expert * n_out * k;
        if !elems.is_multiple_of(block) {
            bail!("{full}: {elems} elements not a multiple of {dtype:?} block size {block}");
        }
        let size_in_bytes = elems / block * dtype.type_size();
        let tensor_start = self.src.content.tensor_data_offset + info.offset;

        let mut raw = vec![0u8; size_in_bytes];
        {
            let mut file = self.src.file.lock().unwrap();
            file.seek(SeekFrom::Start(tensor_start))
                .with_context(|| format!("seeking to {full}"))?;
            file.read_exact(&mut raw)
                .with_context(|| format!("reading {full} ({size_in_bytes} bytes)"))?;
        }

        let storage = QStorage::from_data(std::borrow::Cow::Owned(raw), &self.src.device, dtype)?;
        // Retain the storage's buffer before it moves into the QTensor: cloning a
        // candle Buffer retains the underlying MTLBuffer (no data copy), so this
        // handle and the QTensor point at the same allocation.
        let buffer = match &storage {
            QStorage::Metal(qms) => Some(Arc::new(qms.buffer().clone())),
            _ => None,
        };
        let qtensor = Arc::new(QTensor::new(storage, (n_expert, n_out, k))?);
        Ok(ExpertStack { qtensor: Some(qtensor), buffer, base_off: 0, mmap: None, dtype, n_expert, n_out, k })
    }

    /// The mmap alias load: the stack's quantized bytes are never read into
    /// host memory — the fused kernels index a page-floored no-copy view of the
    /// GGUF mapping, with `base_off` marking the stack's first block inside the
    /// view. The GGUF stores expert stacks expert-major-contiguous (the classic
    /// path does no re-layout either), so the view IS the stack. No QTensor is
    /// built: nothing on the fused Metal path reads one, and the consumers that
    /// need real QTensors (`ReferenceExperts`, `split_expert_stack`) load
    /// theirs via `expert_qtensors`.
    fn expert_stack_mmap(&self, name: &str, src: &Arc<MmapSource>) -> Result<ExpertStack> {
        let full = self.name(name);
        let info = self
            .src
            .content
            .tensor_infos
            .get(&full)
            .with_context(|| format!("expert stack tensor {full} not found"))?;
        let dims = info.shape.dims().to_vec();
        let [n_expert, n_out, k] = dims[..] else {
            bail!("{full} is not a rank-3 expert stack: {dims:?}");
        };
        let dtype = info.ggml_dtype;
        let block = dtype.block_size();
        let elems = n_expert * n_out * k;
        if !elems.is_multiple_of(block) {
            bail!("{full}: {elems} elements not a multiple of {dtype:?} block size {block}");
        }
        let size_in_bytes = elems / block * dtype.type_size();
        let tensor_start = (self.src.content.tensor_data_offset + info.offset) as usize;

        let (buffer, base_off) = src
            .view(tensor_start, size_in_bytes)
            .with_context(|| format!("mmap-aliasing {full}"))?;
        // Same contract the f16 alias path enforces: the expert kernels do
        // vector loads from the bound offset, so it must stay 16-byte aligned
        // (every real GGUF is 32-aligned; this guards hand-crafted files).
        ensure!(
            base_off.is_multiple_of(16),
            "{full}: mmap base_off {base_off} is not 16-byte aligned"
        );
        Ok(ExpertStack {
            qtensor: None,
            buffer: Some(buffer),
            base_off,
            mmap: Some(src.clone()),
            dtype,
            n_expert,
            n_out,
            k,
        })
    }

    /// Loads a tensor by its fully-qualified GGUF name (dtype suffix included),
    /// bypassing the implicit `.weight` suffix that `qtensor` appends.
    fn qtensor_named(&self, full: &str) -> Result<Arc<QTensor>> {
        let mut file = self.src.file.lock().unwrap();
        let qt = self
            .src
            .content
            .tensor(&mut *file, full, &self.src.device)
            .with_context(|| format!("loading tensor {full}"))?;
        Ok(Arc::new(qt))
    }

    /// Dense f32 for a small tensor whose GGUF suffix may be `.weight` or `.bias`.
    /// The routing score-correction bias `exp_probs_b` is stored bias-suffixed
    /// (`blk.N.exp_probs_b.bias`); everything else uses `.weight`. Prefers
    /// `.weight` when present, otherwise falls back to `.bias`.
    pub fn dense_f32_biasable(&self, name: &str) -> Result<Tensor> {
        let weight = self.name(name);
        let full = if self.src.content.tensor_infos.contains_key(&weight) {
            weight
        } else if self.prefix.is_empty() {
            format!("{name}.bias")
        } else {
            format!("{}.{name}.bias", self.prefix)
        };
        let qt = self.qtensor_named(&full)?;
        Ok(qt.dequantize(&self.src.device)?.to_dtype(candle_core::DType::F32)?)
    }

    /// Slices a stacked expert tensor `[n_expert, n_out, k]` into `n_expert`
    /// per-expert rank-2 QTensors `[n_out, k]`, keeping the quantized bytes
    /// (no dequantization). The stack is contiguous in expert-major order, so
    /// each expert's footprint is a fixed byte stride: its `n_out * k` elements
    /// form a whole number of quantization blocks.
    pub fn expert_qtensors(&self, name: &str) -> Result<Vec<Arc<QTensor>>> {
        let qt = self.qtensor(name)?;
        let dims = qt.shape().dims().to_vec();
        let [n_expert, n_out, k] = dims[..] else {
            bail!("{} is not a rank-3 expert stack: {dims:?}", self.name(name));
        };
        split_expert_stack(&qt, n_expert, n_out, k, &self.src.device)
    }
}

/// Byte-slices a stacked expert QTensor `[n_expert, n_out, k]` into per-expert
/// `[n_out, k]` QTensors, preserving the quantized layout. Kept free-standing so
/// callers holding a stack directly (not via GGUF) can reuse it.
pub fn split_expert_stack(
    stack: &QTensor,
    n_expert: usize,
    n_out: usize,
    k: usize,
    device: &Device,
) -> Result<Vec<Arc<QTensor>>> {
    let dtype = stack.dtype();
    let block = dtype.block_size();
    let type_size = dtype.type_size();
    let per_expert_elems = n_out * k;
    if per_expert_elems % block != 0 {
        bail!("expert size {per_expert_elems} is not a multiple of block size {block}");
    }
    let stride = per_expert_elems / block * type_size;
    let data = stack.data()?;
    if data.len() != stride * n_expert {
        bail!(
            "stacked expert data is {} bytes, expected {n_expert} x {stride}",
            data.len()
        );
    }
    let mut out = Vec::with_capacity(n_expert);
    for e in 0..n_expert {
        // to_vec() gives a fresh, over-aligned allocation, satisfying the block
        // struct's alignment requirement on the Metal/CUDA load paths.
        let bytes = data[e * stride..(e + 1) * stride].to_vec();
        let storage = QStorage::from_data(std::borrow::Cow::Owned(bytes), device, dtype)?;
        out.push(Arc::new(QTensor::new(storage, (n_out, k))?));
    }
    Ok(out)
}

/// Wraps `[out_dim, in_dim]` f16 bytes at absolute file offset `abs_off` of
/// `src`'s mapping as a dense f16 tensor WITHOUT copying: page-floored no-copy
/// view → whole-view 1-D f16 `MetalStorage` → `narrow` off the sub-page
/// `base_off` → `reshape`. The result is a contiguous VIEW whose layout
/// start_offset is `base_off / 2` elements; the vendored `matmul_f16` dispatch
/// honors `layout.start_offset()` and requires the byte offset 16-aligned,
/// which the GGUF's 32-byte tensor-data alignment guarantees.
fn f16_alias_tensor(
    src: &MmapSource,
    device: &Device,
    abs_off: usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<Tensor> {
    let Device::Metal(mdev) = device else {
        bail!("f16 mmap aliasing requires a Metal device");
    };
    let elems = out_dim * in_dim;
    let (buffer, base_off) = src.view(abs_off, elems * DType::F16.size_in_bytes())?;
    // base_off is 32-byte aligned (GGUF data alignment), so it is a whole
    // number of f16 elements and satisfies matmul_f16's 16-byte view check.
    ensure!(
        base_off.is_multiple_of(16),
        "f16 alias at file offset {abs_off} is not 16-byte aligned inside its page (base_off {base_off})"
    );
    let count = buffer.length() / DType::F16.size_in_bytes(); // page-multiple, exact
    let storage = MetalStorage::new(buffer, mdev.clone(), count, DType::F16);
    let whole = Tensor::from_storage(
        Storage::Metal(storage),
        count,
        candle_core::op::BackpropOp::none(),
        false,
    );
    // The returned tensor's storage IS the no-copy view for the model's
    // lifetime: with the view registered in the queue-attached residency set,
    // Shared page-cache-backed planes stream at full rate (measured even with
    // the classic loader's driver-allocated buffers; docs/log.md mmap entry).
    // Do NOT "fix" this with Tensor::copy() — Metal's try_clone at the pinned
    // rev is a shallow Arc clone, not a data copy.
    Ok(whole
        .narrow(0, base_off / DType::F16.size_in_bytes(), elems)?
        .reshape((out_dim, in_dim))?)
}

/// Human-readable metadata + tensor listing for `laguna inspect`.
pub fn describe(content: &Content) -> String {
    let mut out = String::new();
    let mut keys: Vec<_> = content.metadata.keys().collect();
    keys.sort();
    for k in keys {
        let v = &content.metadata[k];
        let mut s = format!("{v:?}");
        if s.len() > 120 {
            let cut = (0..=117).rev().find(|&i| s.is_char_boundary(i)).unwrap_or(0);
            s.truncate(cut);
            s.push_str("...");
        }
        let _ = writeln!(out, "{k} = {s}");
    }
    let mut infos: Vec<_> = content.tensor_infos.iter().collect();
    infos.sort_by(|a, b| a.0.cmp(b.0));
    let _ = writeln!(out, "\n{} tensors:", infos.len());
    for (name, info) in infos {
        let _ = writeln!(out, "{name}  {:?}  {:?}", info.shape.dims(), info.ggml_dtype);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::gguf_file;

    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// QLinear::forward must read the RIGHT rows from an offset view: candle's
    /// Metal quantized matmul drops the input's storage start_offset (it rebuilds
    /// the layout from the shape), so a dim-0 `narrow` fed straight through would
    /// silently multiply the wrong rows. The guard materializes such views; this
    /// asserts narrow(0, 1, n) == the same rows copied into fresh storage.
    #[test]
    fn qlinear_forward_honors_row_offset_views() {
        let device = metal_device().unwrap();
        let (out_dim, in_dim, rows) = (16usize, 256usize, 4usize);

        let w = Tensor::from_vec(fill(out_dim * in_dim, 0xA7), (out_dim, in_dim), &Device::Cpu).unwrap();
        let qt = QTensor::quantize(&w.to_device(&device).unwrap(), GgmlDType::Q8_0).unwrap();
        let lin = QLinear::from_qtensor(Arc::new(qt)).unwrap();

        let x = Tensor::from_vec(fill(rows * in_dim, 0xB3), (rows, in_dim), &device).unwrap();
        let tail_view = x.narrow(0, 1, rows - 1).unwrap();
        // Genuinely materialized copy of the same rows (offset 0 storage).
        let tail_rows = Tensor::from_vec(
            x.to_device(&Device::Cpu).unwrap().to_vec2::<f32>().unwrap()[1..].concat(),
            (rows - 1, in_dim),
            &device,
        )
        .unwrap();

        let via_view = lin.forward(&tail_view).unwrap().to_device(&Device::Cpu).unwrap().to_vec2::<f32>().unwrap();
        let via_copy = lin.forward(&tail_rows).unwrap().to_device(&Device::Cpu).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(via_view, via_copy, "offset view multiplied different rows than its materialized copy");
    }

    /// The fused expert stack must load as ONE device allocation shared with its
    /// QTensor — no second upload of the (large) weights. Writes a synthetic GGUF,
    /// loads it through the real `expert_stack` path, and checks the buffer is a
    /// single stack in size and holds the correct weights (the fused matvec through
    /// it matches dequantizing the stack's QTensor).
    #[test]
    fn expert_stack_loads_single_shared_buffer() {
        let device = metal_device().unwrap();
        let (n_expert, n_out, k) = (4usize, 8usize, 256usize);
        let dt = GgmlDType::Q4K;

        let w = Tensor::from_vec(fill(n_expert * n_out * k, 0xE1), (n_expert, n_out, k), &Device::Cpu).unwrap();
        let qt_cpu = QTensor::quantize(&w, dt).unwrap();
        let path = std::env::temp_dir().join(format!("laguna_expert_stack_{}.gguf", std::process::id()));
        {
            let mut f = File::create(&path).unwrap();
            gguf_file::write(&mut f, &[], &[("ffn_gate_exps.weight", &qt_cpu)]).unwrap();
        }

        let gguf = open(&path, &device).unwrap();
        let weights = Weights::from_gguf(gguf);
        // The CLASSIC path explicitly: this test pins the shared-allocation
        // invariant of the copying load (the default mmap path is covered by
        // `expert_stack_mmap_matches_classic`).
        let stack = weights.expert_stack_classic("ffn_gate_exps").unwrap();

        // One buffer, sized to exactly one stack (a double upload would not change
        // this length, but the size check catches a wrong-tensor / wrong-dtype load
        // and pairs with the by-construction single `from_data` in `expert_stack`).
        let expected = n_expert * n_out * k / dt.block_size() * dt.type_size();
        let buf = stack.buffer.as_ref().expect("expert stack has a Metal buffer");
        assert_eq!(buf.length(), expected, "fused buffer must be one stack");
        let qtensor = stack.qtensor.as_ref().expect("classic load carries a QTensor");
        assert_eq!(qtensor.storage_size_in_bytes(), expected);
        assert_eq!(stack.dtype, dt);
        assert_eq!(stack.base_off, 0, "classic load starts at the buffer head");

        // The shared buffer carries the right weights: a fused gather-matvec through
        // stack.buffer matches a CPU reference over the dequantized QTensor.
        let deq = qtensor.dequantize(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (t, top_k) = (2usize, 2usize);
        let x_vec = fill(t * k, 0xC0FFEE);
        let x = Tensor::from_vec(x_vec.clone(), (t, 1, k), &device).unwrap();
        let ids_v: Vec<u32> = vec![0, 3, 1, 2];
        let ids = Tensor::from_vec(ids_v.clone(), (t, top_k), &device).unwrap();
        let got = crate::ops::mul_mv_id(&stack, &x, &ids).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let mut want = vec![0f32; t * top_k * n_out];
        for token in 0..t {
            for slot in 0..top_k {
                let e = ids_v[token * top_k + slot] as usize;
                for o in 0..n_out {
                    let mut acc = 0f32;
                    for i in 0..k {
                        acc += deq[(e * n_out + o) * k + i] * x_vec[token * k + i];
                    }
                    want[(token * top_k + slot) * n_out + o] = acc;
                }
            }
        }
        let (mut num, mut den) = (0f64, 0f64);
        for (g, wv) in got.iter().zip(&want) {
            num += (*g as f64 - *wv as f64).powi(2);
            den += (*wv as f64).powi(2);
        }
        let rel = (num / den.max(1e-30)).sqrt();
        assert!(rel < 1e-3, "fused-through-shared-buffer rel_l2 {rel} too high");

        let _ = std::fs::remove_file(&path);
    }

    /// The alias-load mapping for `path`: reuse the GgufFile's own (default
    /// open), or map explicitly when the test env forced `LAGUNA_LOAD_CLASSIC`.
    fn mmap_source_for(gguf: &GgufFile, path: &Path, device: &Device) -> Arc<MmapSource> {
        match gguf.mmap_source() {
            Some(s) => s.clone(),
            None => MmapSource::open(path, device).unwrap(),
        }
    }

    /// An mmap-aliased f16 plane must be BITWISE identical to uploading the
    /// same bytes, both at rest and through the vendored matmul_f16 kernels
    /// (whose weight extraction must honor the view's nonzero start_offset).
    /// The plane sits at a non-page-aligned (but 32-aligned, as in a GGUF)
    /// offset, so the view is page-floored and base_off is nonzero.
    #[test]
    fn mmap_alias_f16_matches_upload() {
        let device = metal_device().unwrap();
        let (out_dim, in_dim) = (24usize, 64usize);
        let elems = out_dim * in_dim;
        let header = 96usize; // 32-aligned like GGUF tensor data, not page-aligned

        let vals: Vec<half::f16> =
            fill(elems, 0xF16).iter().map(|v| half::f16::from_f32(*v)).collect();
        let path = std::env::temp_dir().join(format!("laguna_mmap_f16_{}.bin", std::process::id()));
        {
            let mut bytes = vec![0u8; header];
            for v in &vals {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(&path, &bytes).unwrap();
        }

        let src = MmapSource::open(&path, &device).unwrap();
        // View plumbing: page-floored start, sub-page base_off, page-multiple length.
        let (buf, base_off) = src.view(header, elems * 2).unwrap();
        assert_eq!(base_off, header, "view must page-floor to offset 0 here");
        assert_eq!(buf.length() % vm_page_size, 0, "no-copy view length must be page-multiple");

        let aliased = f16_alias_tensor(&src, &device, header, out_dim, in_dim).unwrap();
        // Exercise the full residency lifecycle: batch-register the views (as
        // LagunaModel::load does) so the end-of-test drop runs the unregister
        // path against a set that actually holds them.
        src.register_views();
        assert!(aliased.is_contiguous());
        assert_eq!(aliased.dims(), &[out_dim, in_dim]);
        let uploaded = Tensor::from_vec(vals.clone(), (out_dim, in_dim), &device).unwrap();

        let a: Vec<half::f16> = aliased.flatten_all().unwrap().to_vec1().unwrap();
        let u: Vec<half::f16> = uploaded.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (x, y)) in a.iter().zip(&u).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "aliased f16 bytes differ at element {i}");
        }

        // Through the vendored kernels, both the decode gemv (t=1) and the
        // prefill mm (t=16 > F16_MM_MIN_SEQ) branches: identical kernels over
        // identical bytes, only the weight buffer offset differs — bitwise.
        for t in [1usize, 16] {
            let x = Tensor::from_vec(fill(t * in_dim, 0xAB + t as u64), (t, in_dim), &device).unwrap();
            let got: Vec<f32> = crate::ops::matmul_f16(&aliased, &x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let want: Vec<f32> = crate::ops::matmul_f16(&uploaded, &x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "matmul_f16 t={t} differs at element {i}: aliased {g} vs uploaded {w}"
                );
            }
        }

        let _ = std::fs::remove_file(&path);
    }

    /// The mmap-aliased expert stack must be BITWISE identical to the classic
    /// upload through BOTH fused kernel families (mv_id decode gather, mm_id
    /// two-pass prefill): same kernels, same bytes, only the weight buffer
    /// binding offset (`base_off`) differs. The synthetic GGUF's small header
    /// puts the tensor at a non-page-aligned offset, so base_off is exercised
    /// nonzero.
    #[test]
    fn expert_stack_mmap_matches_classic() {
        let device = metal_device().unwrap();
        let (n_expert, n_out, k) = (4usize, 8usize, 256usize);
        let dt = GgmlDType::Q4K;

        let w = Tensor::from_vec(fill(n_expert * n_out * k, 0xA11A5), (n_expert, n_out, k), &Device::Cpu).unwrap();
        let qt_cpu = QTensor::quantize(&w, dt).unwrap();
        let path = std::env::temp_dir().join(format!("laguna_mmap_stack_{}.gguf", std::process::id()));
        {
            let mut f = File::create(&path).unwrap();
            gguf_file::write(&mut f, &[], &[("ffn_up_exps.weight", &qt_cpu)]).unwrap();
        }

        let gguf = open(&path, &device).unwrap();
        let weights = Weights::from_gguf(gguf.clone());
        let classic = weights.expert_stack_classic("ffn_up_exps").unwrap();
        let src = mmap_source_for(&gguf, &path, &device);
        let aliased = weights.expert_stack_mmap("ffn_up_exps", &src).unwrap();

        assert!(aliased.qtensor.is_none(), "mmap stack must not build a QTensor");
        assert!(aliased.mmap.is_some(), "mmap stack must keep the mapping alive");
        assert_ne!(
            aliased.base_off, 0,
            "the synthetic GGUF's tensor offset must exercise a nonzero base_off"
        );
        assert_eq!(aliased.base_off % 32, 0, "GGUF data alignment makes base_off 32-aligned");

        let (t, top_k) = (4usize, 2usize);
        let x = Tensor::from_vec(fill(t * k, 0xBEEF), (t, 1, k), &device).unwrap();
        let ids = Tensor::from_vec(vec![0u32, 3, 1, 2, 2, 0, 3, 1], (t, top_k), &device).unwrap();

        let read = |t: Tensor| -> Vec<f32> { t.flatten_all().unwrap().to_vec1().unwrap() };
        let assert_bitwise = |got: &[f32], want: &[f32], path_name: &str| {
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{path_name} differs at element {i}: aliased {g} vs classic {w}"
                );
            }
        };

        // Decode gather (mv_id; vendored or classic per env — same for both stacks).
        let got = read(crate::ops::mul_mv_id(&aliased, &x, &ids).unwrap());
        let want = read(crate::ops::mul_mv_id(&classic, &x, &ids).unwrap());
        assert_bitwise(&got, &want, "mv_id");

        // Prefill two-pass matmul (mm_id, active variant — same for both stacks).
        let got = read(crate::ops::mul_mm_id(&aliased, &x, &ids).unwrap());
        let want = read(crate::ops::mul_mm_id(&classic, &x, &ids).unwrap());
        assert_bitwise(&got, &want, "mm_id");

        let _ = std::fs::remove_file(&path);
    }
}
