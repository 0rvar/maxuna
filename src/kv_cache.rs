use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::config::LagunaConfig;

/// Per-layer KV cache. Full-attention layers preallocate to `max_ctx` and grow
/// via slice_set; SWA layers keep a fixed ring of `sliding_window` slots.
/// Cache dtype is f16 (sdpa runs in f16).
///
/// Positions are absolute. The caller drives the cache sequentially, so the
/// number of tokens stored before an `append` equals the `pos` handed to the
/// matching `attn_mask` call — this invariant is what lets `attn_mask` describe
/// the exact key ordering `append` returns without sharing extra state.
///
/// K/V are laid out `[n_kv_head, slot, head_dim]`, matching what `attention.rs`
/// feeds into sdpa after adding a leading batch dim.
pub enum LayerCache {
    /// Full attention: every past key is kept, keys returned in position order.
    Full { k: Tensor, v: Tensor, len: usize },
    /// Sliding-window attention: a ring of `window` slots keyed by `pos % window`.
    Swa { k: Tensor, v: Tensor, len: usize, window: usize },
}

impl LayerCache {
    pub fn new(cfg: &LagunaConfig, il: usize, max_ctx: usize, device: &Device) -> Result<Self> {
        let n_kv_head = cfg.n_kv_head;
        let head_dim = cfg.head_dim;
        let alloc = |slots: usize| Tensor::zeros((n_kv_head, slots, head_dim), DType::F16, device);
        if cfg.is_full_attn(il) {
            Ok(LayerCache::Full { k: alloc(max_ctx)?, v: alloc(max_ctx)?, len: 0 })
        } else {
            let window = cfg.sliding_window;
            Ok(LayerCache::Swa { k: alloc(window)?, v: alloc(window)?, len: 0, window })
        }
    }

    /// Append k/v ([n_kv_head, seq, head_dim] f16) and return the effective
    /// cached (k, v) views to attend over. For SWA prefill (seq > 1) the view is
    /// ordered oldest→newest so it lines up with `attn_mask`; for single-token
    /// decode the ring is returned as-is (softmax is order-independent when no
    /// mask is applied).
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq = k.dim(1)?;
        match self {
            LayerCache::Full { k: kb, v: vb, len } => {
                kb.slice_set(k, 1, *len)?;
                vb.slice_set(v, 1, *len)?;
                *len += seq;
                Ok((kb.narrow(1, 0, *len)?, vb.narrow(1, 0, *len)?))
            }
            LayerCache::Swa { k: kr, v: vr, len, window } => {
                let w = *window;
                if seq == 1 {
                    let idx = *len % w;
                    kr.slice_set(k, 1, idx)?;
                    vr.slice_set(v, 1, idx)?;
                    *len += 1;
                    let view = (*len).min(w);
                    Ok((kr.narrow(1, 0, view)?, vr.narrow(1, 0, view)?))
                } else {
                    // Effective view = existing valid keys (oldest→newest) ++ new keys.
                    let eff_k = prepend_existing(kr, k, *len, w)?;
                    let eff_v = prepend_existing(vr, v, *len, w)?;
                    write_ring_multi(kr, k, *len, w)?;
                    write_ring_multi(vr, v, *len, w)?;
                    *len += seq;
                    Ok((eff_k, eff_v))
                }
            }
        }
    }

    /// Attention mask for `seq_len` new queries at absolute position `pos`,
    /// or None when a single decode token needs no mask.
    ///
    /// The mask is additive (0.0 to attend, -inf to block) with shape
    /// `[seq_len, k_seq]`, where column `c` corresponds to the same key
    /// `append` places at position `c` in the returned view.
    pub fn attn_mask(&self, seq_len: usize, pos: usize) -> Result<Option<Tensor>> {
        if seq_len == 1 {
            return Ok(None);
        }
        let device = match self {
            LayerCache::Full { k, .. } | LayerCache::Swa { k, .. } => k.device().clone(),
        };
        let mask = match self {
            LayerCache::Full { .. } => {
                // Keys 0..pos+seq_len in position order; block strictly-future keys.
                let k_seq = pos + seq_len;
                let mut data = vec![0f32; seq_len * k_seq];
                for qi in 0..seq_len {
                    let q_abs = pos + qi;
                    for kj in 0..k_seq {
                        if kj > q_abs {
                            data[qi * k_seq + kj] = f32::NEG_INFINITY;
                        }
                    }
                }
                Tensor::from_vec(data, (seq_len, k_seq), &device)?
            }
            LayerCache::Swa { window, .. } => {
                // Columns are the m surviving past keys (abs pos-m..pos-1) then
                // the seq new keys (abs pos..pos+seq-1). Block future keys and any
                // key older than the window (matches llama.cpp is_masked_swa
                // STANDARD: q_abs - k_abs >= n_swa is masked).
                let w = *window;
                let m = pos.min(w);
                let k_seq = m + seq_len;
                let mut data = vec![0f32; seq_len * k_seq];
                for qi in 0..seq_len {
                    let q_abs = pos + qi;
                    for c in 0..k_seq {
                        let k_abs = pos - m + c;
                        if k_abs > q_abs || q_abs - k_abs >= w {
                            data[qi * k_seq + c] = f32::NEG_INFINITY;
                        }
                    }
                }
                Tensor::from_vec(data, (seq_len, k_seq), &device)?
            }
        };
        Ok(Some(mask))
    }

    pub fn reset(&mut self) {
        match self {
            LayerCache::Full { len, .. } => *len = 0,
            LayerCache::Swa { len, .. } => *len = 0,
        }
    }
}

/// The `min(len, window)` most recent ring entries, reordered oldest→newest.
fn ordered_existing(ring: &Tensor, len: usize, w: usize) -> Result<Option<Tensor>> {
    if len == 0 {
        Ok(None)
    } else if len <= w {
        Ok(Some(ring.narrow(1, 0, len)?))
    } else {
        // Oldest surviving key sits at ring index len % w; rotate so it leads.
        let s = len % w;
        Ok(Some(Tensor::cat(&[&ring.narrow(1, s, w - s)?, &ring.narrow(1, 0, s)?], 1)?))
    }
}

/// Concatenate the ordered surviving ring entries in front of the new keys.
fn prepend_existing(ring: &Tensor, new: &Tensor, len: usize, w: usize) -> Result<Tensor> {
    match ordered_existing(ring, len, w)? {
        None => Ok(new.clone()),
        Some(existing) => Ok(Tensor::cat(&[&existing, new], 1)?),
    }
}

/// Write new keys (abs positions len..len+seq) into the ring, wrap-aware,
/// leaving the ring holding the most recent `min(len+seq, w)` keys.
fn write_ring_multi(ring: &Tensor, new: &Tensor, len: usize, w: usize) -> Result<()> {
    let seq = new.dim(1)?;
    if seq >= w {
        // Only the last w new keys survive; place them at their ring indices.
        let last = new.narrow(1, seq - w, w)?;
        write_wrapped(ring, &last, (len + seq - w) % w, w)
    } else {
        write_wrapped(ring, new, len % w, w)
    }
}

/// slice_set `src` (length <= w) into `ring` starting at `start`, splitting at
/// the ring boundary when it wraps.
fn write_wrapped(ring: &Tensor, src: &Tensor, start: usize, w: usize) -> Result<()> {
    let s = src.dim(1)?;
    if start + s <= w {
        ring.slice_set(&src.contiguous()?, 1, start)?;
    } else {
        let first = w - start;
        ring.slice_set(&src.narrow(1, 0, first)?.contiguous()?, 1, start)?;
        ring.slice_set(&src.narrow(1, first, s - first)?.contiguous()?, 1, 0)?;
    }
    Ok(())
}
