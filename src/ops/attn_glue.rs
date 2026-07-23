//! Host side of the fused attention-glue ops (softplus gate, permute/cast
//! copies, partial-rotary rope). Kernel-side rounding contracts live in
//! attn_glue.metal / rope.metal; each op here is bit-identical to the candle
//! chain it replaces, proven by the bitwise tests below.

use anyhow::Result;
use candle_core::{DType, Tensor};

use crate::ops::dispatch;

/// Fused softplus output gate against the `kernel_attn_gate_*` pair
/// (attn_glue.metal): `out[h,s,d] = attn[h,s,d] * softplus_chain(
/// gate_logits[s,h])`, replacing the candle chain in `AttnBlock::forward`
/// (softplus's 8 elementwise dispatches + the gate transpose/reshape copy +
/// the broadcast_mul) with ONE pass over `attn`. `attn` is `[n_head, seq,
/// head_dim]` contiguous, f32 or f16 (the decode path feeds the raw f16 sdpa
/// output; the kernel widens it in-register — exact — so the f16 input is
/// bit-identical to `cast_f32` + the f32 input, proven by
/// `attn_gate_f16_input_matches_widened_bitwise`); `gate_logits` is
/// `[seq, n_head]` f32 contiguous (the g_proj output). Output is always f32.
/// Bit-identical to the candle chain (`attn_gate_matches_candle_bitwise`
/// proves it), so the fused path is safe under every parity tier. Metal only;
/// the caller's kill-switch is the candle chain (`LAGUNA_ATTN_GLUE_CLASSIC`).
pub fn attn_gate(attn: &Tensor, gate_logits: &Tensor) -> Result<Tensor> {
    dispatch::run_attn_gate(attn, gate_logits)
}

/// Fused `transpose(0,1)+contiguous`: `[d0, d1, d2]` f32 contiguous →
/// `[d1, d0, d2]` f32 contiguous in one pass (a pure permutation copy, so
/// bit-identity is structural). Replaces candle's strided-copy `contiguous()`
/// on a transposed view.
pub fn permute_01(x: &Tensor) -> Result<Tensor> {
    dispatch::run_permute_cast(x, DType::F32)
}

/// Fused `transpose(0,1)+contiguous+to_dtype(F16)`: same permutation with the
/// f32→f16 conversion (round-to-nearest-even, candle's cast scalar) folded into
/// the single pass — one memory pass where the candle chain takes two.
pub fn permute_01_f16(x: &Tensor) -> Result<Tensor> {
    dispatch::run_permute_cast(x, DType::F16)
}

/// Shape-preserving f32→f16 cast (round-to-nearest-even, candle's cast scalar)
/// through the permute kernel's `d0 == 1` degenerate copy. Requires a
/// contiguous input; returns a contiguous tensor of the same shape.
pub fn cast_f16(x: &Tensor) -> Result<Tensor> {
    cast_to(x, DType::F16)
}

/// Shape-preserving f16→f32 cast (exact) — the post-sdpa widening.
pub fn cast_f32(x: &Tensor) -> Result<Tensor> {
    cast_to(x, DType::F32)
}

fn cast_to(x: &Tensor, out_dtype: DType) -> Result<Tensor> {
    let flat = x.reshape((1, 1, x.elem_count()))?;
    Ok(dispatch::run_permute_cast(&flat, out_dtype)?.reshape(x.dims())?)
}

/// Vendored partial-rotary NEOX rope against the `kernel_rope_neox_*` pair
/// (rope.metal): rotates the first `n_rot` dims of `x` `[heads, seq,
/// head_dim]` f32 (pair (i, i + n_rot/2), candle's by-halves math verbatim)
/// and passes dims >= n_rot through, in ONE read+write — replacing
/// `Rope::rotate`'s narrow/contiguous/rope/cat chain. `out_dtype` picks the
/// store width: f32, or f16 — the rotation still runs in f32 and only the
/// final store rounds (one RTNE rounding, pass-through dims included), so the
/// f16 store is bit-identical to the f32 store + `cast_f16`
/// (`rope_f16_store_matches_cast_bitwise` proves it) and folds the standalone
/// post-rope cast away. `cos`/`sin` are the full precomputed
/// `[max_ctx, n_rot/2]` f32 tables; `pos` is the absolute position of `x`'s
/// first token. Bit-identical to the candle chain
/// (`rope_matches_candle_bitwise` proves it). Metal only; kill-switch is the
/// candle chain (`LAGUNA_ATTN_GLUE_CLASSIC`).
pub fn rope_neox(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    pos: usize,
    n_rot: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    dispatch::run_rope(x, cos, sin, pos, n_rot, out_dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;

    /// Deterministic pseudo-random f32s in [lo, hi] (xorshift, no deps).
    fn rand(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = (s >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
                lo + (hi - lo) * u as f32
            })
            .collect()
    }

    fn assert_bits_eq_f32(got: &Tensor, want: &Tensor, what: &str) {
        assert_eq!(got.dims(), want.dims(), "{what}: shape mismatch");
        assert_eq!(got.dtype(), DType::F32, "{what}: dtype");
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (a, b)) in g.iter().zip(w.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{what}: element {i} differs (fused {a:?} bits {:#010x}, candle {b:?} bits {:#010x})",
                a.to_bits(),
                b.to_bits(),
            );
        }
    }

    /// f16 tensors are compared through the exact f16→f32 widening: it is
    /// injective, so f32-bit equality of the widened values is f16-bit equality
    /// of the originals (no dependency on the `half` crate needed).
    fn assert_bits_eq_f16(got: &Tensor, want: &Tensor, what: &str) {
        assert_eq!(got.dims(), want.dims(), "{what}: shape mismatch");
        assert_eq!(got.dtype(), DType::F16, "{what}: dtype");
        assert_eq!(want.dtype(), DType::F16, "{what}: reference dtype");
        assert_bits_eq_f32(
            &got.to_dtype(DType::F32).unwrap(),
            &want.to_dtype(DType::F32).unwrap(),
            what,
        );
    }

    /// The exact candle softplus-gate chain `AttnBlock::forward` runs today
    /// (attention.rs `softplus` + the transpose/reshape + broadcast_mul) — the
    /// ground truth `attn_gate` must reproduce bit-for-bit.
    fn candle_gate_chain(attn: &Tensor, gate_logits: &Tensor, n_head: usize, seq: usize) -> Tensor {
        let x = gate_logits;
        let ax = x.abs().unwrap();
        let relu = x.broadcast_add(&ax).unwrap().affine(0.5, 0.0).unwrap();
        let tail = ax.neg().unwrap().exp().unwrap().affine(1.0, 1.0).unwrap().log().unwrap();
        let sp = relu.broadcast_add(&tail).unwrap();
        let gate = sp.transpose(0, 1).unwrap().reshape((n_head, seq, 1)).unwrap();
        attn.broadcast_mul(&gate).unwrap()
    }

    /// UNIT 1: the fused gate kernel must reproduce the live candle softplus +
    /// broadcast_mul chain BIT-FOR-BIT (f32::to_bits, not a tolerance) across
    /// the production seq / head-count grid, with gate logits spanning both
    /// softplus branches (large negative → pure tail, large positive → pure
    /// relu, and the mixed region). Bit-identity is the whole justification for
    /// shipping the fused kernel on every parity tier — never loosen this to a
    /// tolerance.
    #[test]
    fn attn_gate_matches_candle_bitwise() {
        let device = metal_device().unwrap();
        let head_dim = 128usize;

        for &seq in &[1usize, 8, 512] {
            for &n_head in &[48usize, 72] {
                let attn_v = rand(0x1000 + seq as u64 * 7 + n_head as u64, n_head * seq * head_dim, -4.0, 4.0);
                // Gate logits: wide span (±60) exercises exp underflow-to-0 and
                // the relu-dominant branch on top of the realistic ±20 range.
                let gate_v = rand(0x2000 + seq as u64 + n_head as u64, seq * n_head, -60.0, 60.0);

                let attn = Tensor::from_vec(attn_v, (n_head, seq, head_dim), &device).unwrap();
                let gate = Tensor::from_vec(gate_v, (seq, n_head), &device).unwrap();

                let fused = attn_gate(&attn, &gate).unwrap();
                let want = candle_gate_chain(&attn, &gate, n_head, seq);
                assert_bits_eq_f32(&fused, &want, &format!("gate seq={seq} n_head={n_head}"));
            }
        }
    }

    /// UNIT 2: the permute(+cast) kernels must reproduce candle's
    /// transpose+contiguous (+to_dtype) chains bit-for-bit: f32→f32 and f32→f16
    /// permutes over the q/k/v prefill shapes, the d0==1 plain casts both ways,
    /// and the f16→f32 widening. seq=1 covers the decode-degenerate layout.
    #[test]
    fn permute_cast_matches_candle_bitwise() {
        let device = metal_device().unwrap();
        // [seq, heads, head_dim] source shapes: q (48/72 heads), k/v (8 heads),
        // at prefill and decode seq.
        for &(d0, d1, d2) in &[
            (512usize, 48usize, 128usize),
            (512, 72, 128),
            (512, 8, 128),
            (8, 48, 128),
            (1, 72, 128),
            (1, 8, 128),
        ] {
            let v = rand(0x3000 + (d0 * d1) as u64, d0 * d1 * d2, -8.0, 8.0);
            let x = Tensor::from_vec(v, (d0, d1, d2), &device).unwrap();
            let label = format!("permute [{d0},{d1},{d2}]");

            let want = x.transpose(0, 1).unwrap().contiguous().unwrap();
            assert_bits_eq_f32(&permute_01(&x).unwrap(), &want, &label);

            let want16 = want.to_dtype(DType::F16).unwrap();
            assert_bits_eq_f16(&permute_01_f16(&x).unwrap(), &want16, &format!("{label} +f16"));

            // Plain casts (d0 == 1 path) preserve shape and match to_dtype.
            assert_bits_eq_f16(
                &cast_f16(&x).unwrap(),
                &x.to_dtype(DType::F16).unwrap(),
                &format!("{label} cast_f16"),
            );
            let x16 = x.to_dtype(DType::F16).unwrap();
            assert_bits_eq_f32(
                &cast_f32(&x16).unwrap(),
                &x16.to_dtype(DType::F32).unwrap(),
                &format!("{label} cast_f32"),
            );
        }
    }

    /// The exact candle chain `Rope::rotate` runs today (narrow + contiguous +
    /// candle rope + cat + squeeze/contiguous) — the ground truth `rope_neox`
    /// must reproduce bit-for-bit.
    fn candle_rope_chain(x: &Tensor, cos: &Tensor, sin: &Tensor, pos: usize, n_rot: usize) -> Tensor {
        let (_, seq, head_dim) = x.dims3().unwrap();
        let cos = cos.narrow(0, pos, seq).unwrap();
        let sin = sin.narrow(0, pos, seq).unwrap();
        let x = x.unsqueeze(0).unwrap();
        let rotated = candle_nn::rotary_emb::rope(
            &x.narrow(3, 0, n_rot).unwrap().contiguous().unwrap(),
            &cos,
            &sin,
        )
        .unwrap();
        let out = if n_rot < head_dim {
            let pass = x.narrow(3, n_rot, head_dim - n_rot).unwrap();
            Tensor::cat(&[&rotated, &pass], 3).unwrap()
        } else {
            rotated
        };
        out.squeeze(0).unwrap().contiguous().unwrap()
    }

    /// UNIT 3: the fused partial-rotary rope must reproduce the live candle
    /// chain BIT-FOR-BIT for both layer kinds — full-attention (n_rot=64 of
    /// 128, pass-through upper half) and SWA (n_rot=128, no pass-through) —
    /// across decode/small/prefill seq and a nonzero position (table-row
    /// offset). Table CONTENT is irrelevant to the identity (both sides read
    /// the same rows), so the tables are pseudo-random at cos/sin magnitudes.
    #[test]
    fn rope_matches_candle_bitwise() {
        let device = metal_device().unwrap();
        let head_dim = 128usize;
        let max_ctx = 1024usize;

        for &(n_rot, heads, kind) in &[(64usize, 48usize, "full"), (128, 72, "swa")] {
            let half = n_rot / 2;
            // YaRN-scaled tables exceed 1.0 in magnitude; span that.
            let cos = Tensor::from_vec(
                rand(0x4000 + n_rot as u64, max_ctx * half, -1.5, 1.5),
                (max_ctx, half),
                &device,
            )
            .unwrap();
            let sin = Tensor::from_vec(
                rand(0x5000 + n_rot as u64, max_ctx * half, -1.5, 1.5),
                (max_ctx, half),
                &device,
            )
            .unwrap();

            for &seq in &[1usize, 8, 512] {
                for &pos in &[0usize, 509] {
                    let v = rand(0x6000 + seq as u64 * 3 + pos as u64, heads * seq * head_dim, -6.0, 6.0);
                    let x = Tensor::from_vec(v, (heads, seq, head_dim), &device).unwrap();

                    let fused = rope_neox(&x, &cos, &sin, pos, n_rot, DType::F32).unwrap();
                    let want = candle_rope_chain(&x, &cos, &sin, pos, n_rot);
                    assert_bits_eq_f32(
                        &fused,
                        &want,
                        &format!("rope {kind} seq={seq} pos={pos}"),
                    );
                }
            }
        }
    }

    /// The KV heads never rope in production, but k roping uses the same kernel
    /// at 8 heads — cover that geometry too (regression net for the row/head
    /// index math at a head count that does not divide the others).
    #[test]
    fn rope_kv_head_geometry() {
        let device = metal_device().unwrap();
        let (heads, head_dim, n_rot, max_ctx) = (8usize, 128usize, 64usize, 600usize);
        let half = n_rot / 2;
        let cos = Tensor::from_vec(rand(11, max_ctx * half, -1.5, 1.5), (max_ctx, half), &device).unwrap();
        let sin = Tensor::from_vec(rand(12, max_ctx * half, -1.5, 1.5), (max_ctx, half), &device).unwrap();
        for &(seq, pos) in &[(1usize, 42usize), (17, 3)] {
            let x = Tensor::from_vec(
                rand(13 + seq as u64, heads * seq * head_dim, -6.0, 6.0),
                (heads, seq, head_dim),
                &device,
            )
            .unwrap();
            let fused = rope_neox(&x, &cos, &sin, pos, n_rot, DType::F32).unwrap();
            let want = candle_rope_chain(&x, &cos, &sin, pos, n_rot);
            assert_bits_eq_f32(&fused, &want, &format!("rope kv seq={seq} pos={pos}"));
        }
    }

    /// UNIT 3b: the rope f16-store variant must be BITWISE equal to the f32
    /// store followed by candle's f32→f16 cast (RTNE) — the fold that lets k
    /// (always) and decode q flow out of rope pre-narrowed, deleting the
    /// standalone post-rope casts. Covers both layer kinds (partial rotary
    /// n_rot=64 with pass-through upper dims, and full-width n_rot=128), the
    /// q and k prefill head counts, the decode shape, and pos > 0. Never
    /// loosen to a tolerance — bit-identity is what keeps the fold off the
    /// parity gates' books.
    #[test]
    fn rope_f16_store_matches_cast_bitwise() {
        let device = metal_device().unwrap();
        let head_dim = 128usize;
        let max_ctx = 1024usize;

        for &(n_rot, kind) in &[(64usize, "full"), (128usize, "swa")] {
            let half = n_rot / 2;
            let cos = Tensor::from_vec(
                rand(0x8000 + n_rot as u64, max_ctx * half, -1.5, 1.5),
                (max_ctx, half),
                &device,
            )
            .unwrap();
            let sin = Tensor::from_vec(
                rand(0x8100 + n_rot as u64, max_ctx * half, -1.5, 1.5),
                (max_ctx, half),
                &device,
            )
            .unwrap();

            // k prefill [8, 17, 128], q prefill [48, 17, 128], k decode
            // [8, 1, 128], q decode [48, 1, 128] (the production f16-q shape).
            for &(heads, seq) in &[(8usize, 17usize), (48, 17), (8, 1), (48, 1)] {
                for &pos in &[3usize, 509] {
                    let v = rand(
                        0x8200 + heads as u64 * 5 + seq as u64 + pos as u64,
                        heads * seq * head_dim,
                        -6.0,
                        6.0,
                    );
                    let x = Tensor::from_vec(v, (heads, seq, head_dim), &device).unwrap();

                    let f16_store = rope_neox(&x, &cos, &sin, pos, n_rot, DType::F16).unwrap();
                    let want = rope_neox(&x, &cos, &sin, pos, n_rot, DType::F32)
                        .unwrap()
                        .to_dtype(DType::F16)
                        .unwrap();
                    assert_bits_eq_f16(
                        &f16_store,
                        &want,
                        &format!("rope f16-store {kind} heads={heads} seq={seq} pos={pos}"),
                    );
                }
            }
        }
    }

    /// UNIT 1b: the f16-attn-input gate variant must be BITWISE equal to the
    /// f16→f32 widening (exact) followed by the f32 gate — the fold that lets
    /// the decode path feed the raw f16 sdpa output straight to the gate,
    /// deleting the standalone post-sdpa cast. Decode shape [n_head, 1, 128]
    /// and a seq > 1 shape, both head counts, gate logits spanning both
    /// softplus branches. Never loosen to a tolerance.
    #[test]
    fn attn_gate_f16_input_matches_widened_bitwise() {
        let device = metal_device().unwrap();
        let head_dim = 128usize;

        for &seq in &[1usize, 8] {
            for &n_head in &[48usize, 72] {
                let attn_v = rand(0x9000 + seq as u64 * 7 + n_head as u64, n_head * seq * head_dim, -4.0, 4.0);
                let gate_v = rand(0x9100 + seq as u64 + n_head as u64, seq * n_head, -60.0, 60.0);

                let attn16 = Tensor::from_vec(attn_v, (n_head, seq, head_dim), &device)
                    .unwrap()
                    .to_dtype(DType::F16)
                    .unwrap();
                let gate = Tensor::from_vec(gate_v, (seq, n_head), &device).unwrap();

                let fused = attn_gate(&attn16, &gate).unwrap();
                let want = attn_gate(&attn16.to_dtype(DType::F32).unwrap(), &gate).unwrap();
                assert_bits_eq_f32(&fused, &want, &format!("gate f16-in seq={seq} n_head={n_head}"));
            }
        }
    }

    /// Every op resolves its operands via `start_offset * dtype_size` (like
    /// run_combine), but the other bitwise tests build inputs with
    /// `Tensor::from_vec` (offset 0). Feed each op a CONTIGUOUS view that
    /// starts mid-buffer (narrow along dim 0 keeps contiguity with a nonzero
    /// start_offset) and assert bitwise equality vs the candle chain on the
    /// SAME view — a dropped offset would read the buffer head and diverge.
    /// Covers f16 outputs (permute_01_f16, cast_f16) and an offset f16 input
    /// (cast_f32).
    #[test]
    fn glue_ops_handle_offset_views() {
        let device = metal_device().unwrap();
        let (n_head, seq, head_dim) = (48usize, 8usize, 128usize);

        // attn_gate: both operands are offset views.
        let attn_big = Tensor::from_vec(
            rand(0x7000, (n_head + 3) * seq * head_dim, -4.0, 4.0),
            (n_head + 3, seq, head_dim),
            &device,
        )
        .unwrap();
        let attn = attn_big.narrow(0, 3, n_head).unwrap();
        assert!(attn.is_contiguous(), "narrowed attn view must stay contiguous");
        let gate_big = Tensor::from_vec(
            rand(0x7100, (seq + 5) * n_head, -60.0, 60.0),
            (seq + 5, n_head),
            &device,
        )
        .unwrap();
        let gate = gate_big.narrow(0, 5, seq).unwrap();
        assert!(gate.is_contiguous(), "narrowed gate view must stay contiguous");
        let fused = attn_gate(&attn, &gate).unwrap();
        let want = candle_gate_chain(&attn, &gate, n_head, seq);
        assert_bits_eq_f32(&fused, &want, "offset attn_gate");

        // permute_cast (f32->f32 and f32->f16) plus the shape-preserving casts,
        // all on one offset view.
        let (d0, d1, d2) = (8usize, 48usize, 128usize);
        let x_big = Tensor::from_vec(
            rand(0x7200, (d0 + 2) * d1 * d2, -8.0, 8.0),
            (d0 + 2, d1, d2),
            &device,
        )
        .unwrap();
        let x = x_big.narrow(0, 2, d0).unwrap();
        assert!(x.is_contiguous(), "narrowed permute view must stay contiguous");
        let want = x.transpose(0, 1).unwrap().contiguous().unwrap();
        assert_bits_eq_f32(&permute_01(&x).unwrap(), &want, "offset permute_01");
        assert_bits_eq_f16(
            &permute_01_f16(&x).unwrap(),
            &want.to_dtype(DType::F16).unwrap(),
            "offset permute_01_f16",
        );
        assert_bits_eq_f16(
            &cast_f16(&x).unwrap(),
            &x.to_dtype(DType::F16).unwrap(),
            "offset cast_f16",
        );
        // Offset f16 INPUT for the widening cast: narrow an f16 buffer.
        let x16_big = x_big.to_dtype(DType::F16).unwrap();
        let x16 = x16_big.narrow(0, 2, d0).unwrap();
        assert!(x16.is_contiguous(), "narrowed f16 view must stay contiguous");
        assert_bits_eq_f32(
            &cast_f32(&x16).unwrap(),
            &x16.to_dtype(DType::F32).unwrap(),
            "offset cast_f32",
        );

        // rope_neox: x AND the cos/sin tables are offset views.
        let (heads, n_rot, max_ctx, pos) = (48usize, 64usize, 64usize, 7usize);
        let half = n_rot / 2;
        let x_big = Tensor::from_vec(
            rand(0x7300, (heads + 2) * seq * head_dim, -6.0, 6.0),
            (heads + 2, seq, head_dim),
            &device,
        )
        .unwrap();
        let x = x_big.narrow(0, 2, heads).unwrap();
        assert!(x.is_contiguous(), "narrowed rope view must stay contiguous");
        let table = |seed: u64| {
            let big = Tensor::from_vec(
                rand(seed, (max_ctx + 4) * half, -1.5, 1.5),
                (max_ctx + 4, half),
                &device,
            )
            .unwrap();
            let t = big.narrow(0, 4, max_ctx).unwrap();
            assert!(t.is_contiguous(), "narrowed table view must stay contiguous");
            t
        };
        let (cos, sin) = (table(0x7400), table(0x7500));
        let fused = rope_neox(&x, &cos, &sin, pos, n_rot, DType::F32).unwrap();
        let want = candle_rope_chain(&x, &cos, &sin, pos, n_rot);
        assert_bits_eq_f32(&fused, &want, "offset rope_neox");
    }

    #[test]
    fn shape_and_dtype_errors() {
        let device = metal_device().unwrap();
        // attn_gate: gate shape mismatch.
        let attn = Tensor::zeros((4, 2, 8), DType::F32, &device).unwrap();
        let bad_gate = Tensor::zeros((2, 5), DType::F32, &device).unwrap();
        assert!(attn_gate(&attn, &bad_gate).is_err());
        // attn_gate: wrong gate dtype (attn itself may be f32 OR f16, but the
        // gate logits are always f32).
        let gate16 = Tensor::zeros((2, 4), DType::F16, &device).unwrap();
        assert!(attn_gate(&attn, &gate16).is_err());
        // permute: unsupported dtype pair (f16 -> f16 has no kernel).
        let x16 = Tensor::zeros((2, 3, 4), DType::F16, &device).unwrap();
        assert!(dispatch::run_permute_cast(&x16, DType::F16).is_err());
        // rope: table-width mismatch, odd n_rot, oversized n_rot, table too
        // short, unsupported output dtype.
        let x = Tensor::zeros((2, 3, 8), DType::F32, &device).unwrap();
        let cs = Tensor::zeros((16, 3), DType::F32, &device).unwrap();
        assert!(rope_neox(&x, &cs, &cs, 0, 4, DType::F32).is_err()); // cols 3 != n_rot/2 = 2
        let cs4 = Tensor::zeros((16, 4), DType::F32, &device).unwrap();
        assert!(rope_neox(&x, &cs4, &cs4, 0, 7, DType::F32).is_err()); // odd
        assert!(rope_neox(&x, &cs4, &cs4, 0, 16, DType::F32).is_err()); // > head_dim
        assert!(rope_neox(&x, &cs4, &cs4, 15, 8, DType::F32).is_err()); // pos + seq > rows
        assert!(rope_neox(&x, &cs4, &cs4, 0, 8, DType::BF16).is_err()); // f32/f16 only
    }
}
