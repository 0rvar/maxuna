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

/// The per-layer state the attention-mask math reads — the attention kind, plus
/// the window for SWA. Nothing else in a `LayerCache` affects the mask, so a
/// mask built from a `MaskKind` is identical across every layer of that kind
/// and can be hoisted out of the per-layer loop.
#[derive(Clone, Copy)]
pub enum MaskKind {
    Full,
    Swa { window: usize },
}

/// Attention mask for `seq_len` new queries at absolute position `pos`, or None
/// when a single decode token needs no mask.
///
/// The mask is additive (0.0 to attend, -inf to block) with shape
/// `[seq_len, k_seq]`, where column `c` corresponds to the same key `append`
/// places at position `c` in the returned view. This is a pure function of
/// (kind, seq_len, pos) — it reads no cache contents — so a caller can build it
/// once per forward and share it across every layer of the same kind.
pub fn attn_mask_for(
    kind: MaskKind,
    seq_len: usize,
    pos: usize,
    device: &Device,
) -> Result<Option<Tensor>> {
    if seq_len == 1 {
        return Ok(None);
    }
    let mask = match kind {
        MaskKind::Full => {
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
            Tensor::from_vec(data, (seq_len, k_seq), device)?
        }
        MaskKind::Swa { window: w } => {
            // Columns are the m surviving past keys (abs pos-m..pos-1) then the
            // seq new keys (abs pos..pos+seq-1). Block future keys and any key
            // older than the window (matches llama.cpp is_masked_swa STANDARD:
            // q_abs - k_abs >= n_swa is masked).
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
            Tensor::from_vec(data, (seq_len, k_seq), device)?
        }
    };
    Ok(Some(mask))
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

    /// Convenience wrapper over `attn_mask_for` for a single cache. Production
    /// prefill hoists the build out of the per-layer loop and calls the free
    /// function once per kind; single-cache callers (tests, benches) use this.
    /// See `attn_mask_for` for the mask semantics.
    pub fn attn_mask(&self, seq_len: usize, pos: usize) -> Result<Option<Tensor>> {
        attn_mask_for(self.mask_kind(), seq_len, pos, &self.device())
    }

    /// The mask kind for this cache — the only per-layer state the mask reads.
    pub fn mask_kind(&self) -> MaskKind {
        match self {
            LayerCache::Full { .. } => MaskKind::Full,
            LayerCache::Swa { window, .. } => MaskKind::Swa { window: *window },
        }
    }

    /// The device the cache tensors live on.
    pub fn device(&self) -> Device {
        match self {
            LayerCache::Full { k, .. } | LayerCache::Swa { k, .. } => k.device().clone(),
        }
    }

    pub fn reset(&mut self) {
        match self {
            LayerCache::Full { len, .. } => *len = 0,
            LayerCache::Swa { len, .. } => *len = 0,
        }
    }

    /// Number of positions currently stored (absolute count of appends, the same
    /// `len` both variants advance). All layers are driven in lockstep, so this
    /// is identical across every cache after a shared forward.
    pub fn len(&self) -> usize {
        match self {
            LayerCache::Full { len, .. } | LayerCache::Swa { len, .. } => *len,
        }
    }

    /// Snapshot the state a `span`-token verify forward is about to clobber, so
    /// `rollback` can restore it on a partial accept. Taken BEFORE the verify
    /// forward, at the current length `len0`.
    ///
    /// Full-attention layers write to fresh absolute slots [len0, len0+span) and
    /// never overwrite live data, so their rollback is a pure length truncation —
    /// no data snapshot (`LayerCheckpoint::Full`). SWA-ring layers overwrite the
    /// slots positions [len0, len0+span) map to, evicting whatever those slots
    /// held; the snapshot deep-copies that pre-verify K/V so rejected slots can be
    /// put back byte-for-byte.
    pub fn checkpoint(&self, span: usize) -> Result<LayerCheckpoint> {
        match self {
            LayerCache::Full { .. } => Ok(LayerCheckpoint::Full),
            LayerCache::Swa { k, v, len, window } => {
                let (len0, w) = (*len, *window);
                Ok(LayerCheckpoint::Swa {
                    k: snapshot_ring(k, len0, span, w)?,
                    v: snapshot_ring(v, len0, span, w)?,
                })
            }
        }
    }

    /// Roll a verify forward back to `len0 + commit` (0 <= commit <= span), where
    /// `len0`/`span` are the checkpoint's. Full layers truncate the length,
    /// discarding the rejected tail (its stale slots sit past `len` and are
    /// overwritten by the next append). SWA layers restore the ring slots the
    /// rejected positions [len0+commit, len0+span) overwrote to their pre-verify
    /// contents, then lower the length. After this, the cache is byte-identical to
    /// one that only ever appended the committed prefix.
    pub fn rollback(
        &mut self,
        ckpt: &LayerCheckpoint,
        len0: usize,
        span: usize,
        commit: usize,
    ) -> Result<()> {
        match (self, ckpt) {
            (LayerCache::Full { len, .. }, LayerCheckpoint::Full) => {
                *len = len0 + commit;
                Ok(())
            }
            (LayerCache::Swa { k: kr, v: vr, len, window }, LayerCheckpoint::Swa { k: sk, v: sv }) => {
                let w = *window;
                let rejected = span - commit;
                if rejected > 0 {
                    // The snapshot is stored in position order [len0, len0+span);
                    // the rejected tail is its suffix [commit, span). Write it back
                    // to the ring at the rejected positions' slots (wrap-aware).
                    let start = (len0 + commit) % w;
                    write_wrapped(kr, &sk.narrow(1, commit, rejected)?, start, w)?;
                    write_wrapped(vr, &sv.narrow(1, commit, rejected)?, start, w)?;
                }
                *len = len0 + commit;
                Ok(())
            }
            _ => anyhow::bail!("kv_rollback: checkpoint kind does not match cache kind"),
        }
    }
}

/// Per-layer rollback state produced by `LayerCache::checkpoint`. Full layers
/// carry no data (truncation restores them); SWA layers carry the pre-verify K/V
/// of the ring slots the verify span overwrites, deep-copied into independent
/// storage and laid out in position order `[n_kv_head, span, head_dim]`.
pub enum LayerCheckpoint {
    Full,
    Swa { k: Tensor, v: Tensor },
}

/// A whole-model KV rollback point: the length at checkpoint time, the verify
/// span, and the per-layer snapshots (one entry per cache, in layer order).
pub struct KvCheckpoint {
    pub(crate) len0: usize,
    pub(crate) span: usize,
    pub(crate) layers: Vec<LayerCheckpoint>,
}

impl KvCheckpoint {
    pub(crate) fn new(len0: usize, span: usize, layers: Vec<LayerCheckpoint>) -> Self {
        Self { len0, span, layers }
    }

    /// The verify span this checkpoint covers (commit on rollback must be <= it).
    pub fn span(&self) -> usize {
        self.span
    }
}

/// Deep-copy the ring slots that positions [len0, len0+span) occupy, returned in
/// POSITION order as `[n_kv_head, span, head_dim]` (wrap-aware). Independent of
/// the ring's storage: on Metal at our pinned candle rev `Tensor::copy()` /
/// `contiguous()` on an already-contiguous view is a shallow Arc clone that
/// copies no data (CLAUDE.md trap), so `materialize` forces a real allocation.
fn snapshot_ring(ring: &Tensor, len0: usize, span: usize, w: usize) -> Result<Tensor> {
    let start = len0 % w;
    let seg1 = span.min(w - start);
    let block = if seg1 == span {
        ring.narrow(1, start, span)?
    } else {
        // Span straddles the ring boundary: [start, w) then [0, span - seg1).
        Tensor::cat(&[&ring.narrow(1, start, seg1)?, &ring.narrow(1, 0, span - seg1)?], 1)?
    };
    materialize(&block)
}

/// Force `t` into freshly allocated, independent device storage, byte-for-byte.
/// `contiguous()` first (needed for `slice_set`, and a no-op-or-shallow view is
/// fine here — we immediately blit its bytes elsewhere); then `slice_set` into a
/// fresh zero tensor, which copies the raw bits (unlike `affine`, which would
/// canonicalize -0.0).
fn materialize(t: &Tensor) -> Result<Tensor> {
    let src = t.contiguous()?;
    let dst = src.zeros_like()?;
    dst.slice_set(&src, 1, 0)?;
    Ok(dst)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Prefer the real target device (Metal) so the shallow-copy trap the
    /// snapshot guards against is actually exercised; fall back to CPU where
    /// Metal is unavailable (the rollback math is device-agnostic).
    fn dev() -> Device {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    }

    /// Deterministic pseudo-random f32s (LCG, no deps), in roughly [-0.5, 0.5].
    fn seeded(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / u32::MAX as f32) - 0.5
            })
            .collect()
    }

    /// K/V for absolute positions `start..start+len` as `[n_kv, len, hd]` f16,
    /// each position's row keyed by that position so identical positions produce
    /// identical bits across independent cache builds.
    fn kv_block(start: usize, len: usize, n_kv: usize, hd: usize, dev: &Device) -> (Tensor, Tensor) {
        let build = |base: u64| {
            let rows: Vec<Tensor> = (start..start + len)
                .map(|p| {
                    Tensor::from_vec(seeded(n_kv * hd, base + p as u64), (n_kv, 1, hd), dev).unwrap()
                })
                .collect();
            Tensor::cat(&rows.iter().collect::<Vec<_>>(), 1)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap()
        };
        (build(1000), build(9000))
    }

    fn fresh_swa(window: usize, n_kv: usize, hd: usize, dev: &Device) -> LayerCache {
        let z = || Tensor::zeros((n_kv, window, hd), DType::F16, dev).unwrap();
        LayerCache::Swa { k: z(), v: z(), len: 0, window }
    }

    fn fresh_full(max_ctx: usize, n_kv: usize, hd: usize, dev: &Device) -> LayerCache {
        let z = || Tensor::zeros((n_kv, max_ctx, hd), DType::F16, dev).unwrap();
        LayerCache::Full { k: z(), v: z(), len: 0 }
    }

    fn append_range(c: &mut LayerCache, start: usize, len: usize, n_kv: usize, hd: usize, dev: &Device) {
        if len == 0 {
            return;
        }
        let (k, v) = kv_block(start, len, n_kv, hd, dev);
        c.append(&k, &v).unwrap();
    }

    /// Raw bits of every element (f16 widened to f32 losslessly, then bit-cast),
    /// so the comparison catches sign-of-zero differences a float `==` would miss.
    fn bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect()
    }

    fn ring_kv(c: &LayerCache) -> (Tensor, Tensor) {
        match c {
            LayerCache::Full { k, v, .. } | LayerCache::Swa { k, v, .. } => (k.clone(), v.clone()),
        }
    }

    /// Test 1: a full-attention layer truncated mid-stream and re-extended equals
    /// a fresh cache built from the surviving prefix plus the same continuation —
    /// over the valid `[0, len)` view (stale slots past `len` are ignored by
    /// every reader and reclaimed by the next append).
    #[test]
    fn full_layer_truncate_matches_fresh() {
        let dev = dev();
        let (n_kv, hd, max_ctx) = (2, 4, 64);
        for commit in [0usize, 3, 8] {
            let span = 8;
            let len0 = 12;
            let n_new = 4;

            let mut a = fresh_full(max_ctx, n_kv, hd, &dev);
            append_range(&mut a, 0, len0, n_kv, hd, &dev);
            let lc = a.checkpoint(span).unwrap();
            append_range(&mut a, len0, span, n_kv, hd, &dev); // verify span
            a.rollback(&lc, len0, span, commit).unwrap();
            append_range(&mut a, len0 + commit, n_new, n_kv, hd, &dev);

            let mut b = fresh_full(max_ctx, n_kv, hd, &dev);
            append_range(&mut b, 0, len0 + commit, n_kv, hd, &dev);
            append_range(&mut b, len0 + commit, n_new, n_kv, hd, &dev);

            let end = len0 + commit + n_new;
            assert_eq!(a.len(), end, "commit {commit}");
            assert_eq!(b.len(), end, "commit {commit}");
            let (ka, va) = ring_kv(&a);
            let (kb, vb) = ring_kv(&b);
            let view = |t: &Tensor| t.narrow(1, 0, end).unwrap();
            assert_eq!(bits(&view(&ka)), bits(&view(&kb)), "commit {commit} K");
            assert_eq!(bits(&view(&va)), bits(&view(&vb)), "commit {commit} V");
        }
    }

    /// Test 2: SWA ring checkpoint/rollback is byte-exact across the interesting
    /// regimes — (a) len0 < window (slots empty pre-wrap), (b) len0 > window with
    /// the span fully inside the current turn of the ring, (c) span straddling the
    /// ring boundary — for every commit in 0..=span and with the rejected tail
    /// both re-covered and not re-covered by the subsequent appends (n_new). The
    /// property: rollback(commit) then any continuation == never appending the
    /// rejected positions. The FULL ring is compared, unwritten slots included.
    #[test]
    fn swa_ring_rollback_is_bit_exact() {
        let dev = dev();
        let (n_kv, hd) = (2, 4);
        // (window, prefix_len == len0, span)
        for &(window, len0, span) in &[(16usize, 5usize, 4usize), (8, 12, 3), (8, 14, 4)] {
            for commit in 0..=span {
                for &n_new in &[0usize, 2] {
                    let mut a = fresh_swa(window, n_kv, hd, &dev);
                    append_range(&mut a, 0, len0, n_kv, hd, &dev);
                    let lc = a.checkpoint(span).unwrap();
                    append_range(&mut a, len0, span, n_kv, hd, &dev); // verify span
                    a.rollback(&lc, len0, span, commit).unwrap();
                    append_range(&mut a, len0 + commit, n_new, n_kv, hd, &dev);

                    let mut b = fresh_swa(window, n_kv, hd, &dev);
                    append_range(&mut b, 0, len0, n_kv, hd, &dev);
                    append_range(&mut b, len0, commit, n_kv, hd, &dev); // committed prefix
                    append_range(&mut b, len0 + commit, n_new, n_kv, hd, &dev);

                    let label = format!("window {window} len0 {len0} span {span} commit {commit} n_new {n_new}");
                    assert_eq!(a.len(), b.len(), "{label}: len");
                    let (ka, va) = ring_kv(&a);
                    let (kb, vb) = ring_kv(&b);
                    assert_eq!(bits(&ka), bits(&kb), "{label}: K");
                    assert_eq!(bits(&va), bits(&vb), "{label}: V");
                }
            }
        }
    }

    /// Test 3: the snapshot is a real copy, not a Metal shallow Arc clone. Take a
    /// checkpoint, then overwrite the very slots it captured (by running the verify
    /// span) and assert the snapshot's bytes are unchanged. If `materialize`
    /// regressed to a shallow view aliasing the ring, the append would mutate the
    /// snapshot and this fails.
    #[test]
    fn swa_snapshot_is_a_real_copy() {
        let dev = dev();
        let (n_kv, hd, window, len0, span) = (2, 4, 8, 12, 4);
        let mut c = fresh_swa(window, n_kv, hd, &dev);
        append_range(&mut c, 0, len0, n_kv, hd, &dev);
        let lc = c.checkpoint(span).unwrap();
        let (before_k, before_v) = match &lc {
            LayerCheckpoint::Swa { k, v } => (bits(k), bits(v)),
            LayerCheckpoint::Full => unreachable!("SWA cache yields an SWA checkpoint"),
        };
        // Clobber the captured slots.
        append_range(&mut c, len0, span, n_kv, hd, &dev);
        let (after_k, after_v) = match &lc {
            LayerCheckpoint::Swa { k, v } => (bits(k), bits(v)),
            LayerCheckpoint::Full => unreachable!(),
        };
        assert_eq!(before_k, after_k, "snapshot K aliased the ring (shallow copy)");
        assert_eq!(before_v, after_v, "snapshot V aliased the ring (shallow copy)");
    }
}
