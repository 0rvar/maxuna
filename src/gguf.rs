use std::fmt::Write as _;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use candle_core::quantized::gguf_file::Content;
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{Device, Module, Tensor};
use candle_metal_kernels::metal::Buffer;
use candle_nn::RmsNorm;

/// An opened GGUF: parsed header plus the file handle tensors are read from.
pub struct GgufFile {
    pub content: Content,
    pub device: Device,
    pub path: PathBuf,
    file: Mutex<File>,
}

pub fn metal_device() -> Result<Device> {
    Ok(Device::new_metal(0)?)
}

pub fn open(path: impl AsRef<Path>, device: &Device) -> Result<Arc<GgufFile>> {
    let path = path.as_ref();
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let content = Content::read(&mut file).with_context(|| format!("parsing GGUF {}", path.display()))?;
    Ok(Arc::new(GgufFile {
        content,
        device: device.clone(),
        path: path.to_path_buf(),
        file: Mutex::new(file),
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
        self.inner.forward(x)
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

/// Stacked per-expert weights kept in their quantized GGUF layout:
/// one 3D QTensor `[n_expert, n_out, k]` whose device buffer the `ops::{mv_id,mm_id}`
/// kernels index directly by expert id.
pub struct ExpertStack {
    pub qtensor: Arc<QTensor>,
    /// The stack's quantized bytes as a raw device buffer, for the fused
    /// `ops::{mv_id,mm_id}` kernels. This is a retained handle to the SAME
    /// `MTLBuffer` that backs `qtensor` (both were cloned from one `QStorage`),
    /// so the fused path indexes the resident weights with no second upload.
    /// `None` off Metal — the Reference runner uses `qtensor`/`expert_qtensors`.
    pub buffer: Option<Arc<Buffer>>,
    pub dtype: GgmlDType,
    pub n_expert: usize,
    pub n_out: usize,
    pub k: usize,
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

    /// Loads a stacked expert tensor `[n_expert, n_out, k]` such that the fused
    /// MoE kernels and the wrapping `QTensor` share ONE device allocation. We read
    /// the quantized bytes from the file ourselves, upload them once via
    /// `QStorage::from_data`, retain a handle to that buffer, and only then wrap
    /// the same storage in a `QTensor` — so no second copy of the (large) expert
    /// weights ever lands in VRAM.
    pub fn expert_stack(&self, name: &str) -> Result<ExpertStack> {
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
        Ok(ExpertStack { qtensor, buffer, dtype, n_expert, n_out, k })
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
        let stack = weights.expert_stack("ffn_gate_exps").unwrap();

        // One buffer, sized to exactly one stack (a double upload would not change
        // this length, but the size check catches a wrong-tensor / wrong-dtype load
        // and pairs with the by-construction single `from_data` in `expert_stack`).
        let expected = n_expert * n_out * k / dt.block_size() * dt.type_size();
        let buf = stack.buffer.as_ref().expect("expert stack has a Metal buffer");
        assert_eq!(buf.length(), expected, "fused buffer must be one stack");
        assert_eq!(stack.qtensor.storage_size_in_bytes(), expected);
        assert_eq!(stack.dtype, dt);

        // The shared buffer carries the right weights: a fused gather-matvec through
        // stack.buffer matches a CPU reference over the dequantized QTensor.
        let deq = stack.qtensor.dequantize(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
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
}
