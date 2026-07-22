use std::f64::consts::PI;

use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::config::RopeKind;

/// Precomputed rotary tables for one layer type. Full-attention layers use YaRN
/// with partial rotary (n_rot = 64 of head_dim 128); SWA layers plain rope over
/// all 128 dims. NEOX pairing: dim i pairs with i + n_rot/2, dims >= n_rot pass
/// through untouched.
pub struct Rope {
    /// cos(theta) * mscale, shape [max_ctx, n_rot/2], f32.
    cos: Tensor,
    /// sin(theta) * mscale, shape [max_ctx, n_rot/2], f32.
    sin: Tensor,
    n_rot: usize,
}

impl Rope {
    pub fn new(kind: &RopeKind, max_ctx: usize, device: &Device) -> Result<Self> {
        // Per-pair inverse frequencies (one entry per rotated dimension pair) and
        // the magnitude scaling applied uniformly to cos/sin. `mscale` is 1 for
        // plain rope; for YaRN it is the net attention factor (see below).
        let (n_rot, inv_freq, mscale) = match kind {
            RopeKind::Plain { freq_base, n_rot } => {
                let base = *freq_base as f64;
                let n = *n_rot as f64;
                let inv_freq: Vec<f64> = (0..n_rot / 2)
                    .map(|j| base.powf(-(2.0 * j as f64) / n))
                    .collect();
                (*n_rot, inv_freq, 1.0)
            }
            RopeKind::Yarn {
                freq_base,
                factor,
                original_ctx,
                beta_fast,
                beta_slow,
                attn_factor,
                n_rot,
            } => {
                let base = *freq_base as f64;
                let n = *n_rot as f64;
                let factor = *factor as f64;

                // Correction range (in dim-pair units): high-frequency pairs
                // below `low` are left extrapolated (unscaled), low-frequency
                // pairs above `high` are fully interpolated (scaled by 1/factor),
                // and a linear ramp blends the two in between.
                let corr = |num_rot: f32| {
                    n * ((*original_ctx as f64) / (num_rot as f64 * 2.0 * PI)).ln()
                        / (2.0 * base.ln())
                };
                let low = corr(*beta_fast).floor().max(0.0);
                let mut high = corr(*beta_slow).ceil().min(n - 1.0);
                if low == high {
                    high += 0.001;
                }

                let inv_freq: Vec<f64> = (0..n_rot / 2)
                    .map(|j| {
                        let extrap = base.powf(-(2.0 * j as f64) / n);
                        let interp = extrap / factor;
                        let ramp = (((j as f64) - low) / (high - low)).clamp(0.0, 1.0);
                        interp * ramp + extrap * (1.0 - ramp)
                    })
                    .collect();

                // The fork applies YaRN's magnitude scaling once: ggml's rope
                // multiplies cos/sin by attn_factor * (1 + 0.1*ln(1/freq_scale)),
                // and llama_context pre-divides attn_factor by that same term, so
                // the net factor reaching the tables is exactly the config
                // attention_factor. Replicate that net effect directly.
                (*n_rot, inv_freq, *attn_factor as f64)
            }
        };

        let half = inv_freq.len();
        let mut cos = vec![0f32; max_ctx * half];
        let mut sin = vec![0f32; max_ctx * half];
        for p in 0..max_ctx {
            for j in 0..half {
                let theta = p as f64 * inv_freq[j];
                cos[p * half + j] = (theta.cos() * mscale) as f32;
                sin[p * half + j] = (theta.sin() * mscale) as f32;
            }
        }

        let cos = Tensor::from_vec(cos, (max_ctx, half), device)?;
        let sin = Tensor::from_vec(sin, (max_ctx, half), device)?;
        Ok(Self { cos, sin, n_rot })
    }

    /// q, k: [n_head, seq, head_dim]; positions pos..pos+seq.
    pub fn apply(&self, q: &Tensor, k: &Tensor, pos: usize) -> Result<(Tensor, Tensor)> {
        Ok((self.rotate(q, pos)?, self.rotate(k, pos)?))
    }

    /// Rotate the first `n_rot` dims of x (f32) with NEOX pairing (dim i with
    /// i + n_rot/2); any trailing dims pass through untouched.
    fn rotate(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let (_, seq, head_dim) = x.dims3()?;
        let cos = self.cos.narrow(0, pos, seq)?;
        let sin = self.sin.narrow(0, pos, seq)?;

        // candle's rope kernel wants a batch dim and pairs the two contiguous
        // halves of its input, so feed it exactly the rotated block.
        let x = x.unsqueeze(0)?;
        let rotated = candle_nn::rotary_emb::rope(&x.narrow(3, 0, self.n_rot)?.contiguous()?, &cos, &sin)?;
        let out = if self.n_rot < head_dim {
            let pass = x.narrow(3, self.n_rot, head_dim - self.n_rot)?;
            Tensor::cat(&[&rotated, &pass], 3)?
        } else {
            rotated
        };
        Ok(out.squeeze(0)?.contiguous()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cos_row(rope: &Rope, pos: usize) -> Vec<f32> {
        rope.cos.narrow(0, pos, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn sin_row(rope: &Rope, pos: usize) -> Vec<f32> {
        rope.sin.narrow(0, pos, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// Laguna's YaRN attention_factor = 1 + 0.1*ln(128), applied to cos/sin once.
    const LAGUNA_MSCALE: f64 = 1.4852030263919618;

    fn laguna_yarn() -> RopeKind {
        RopeKind::Yarn {
            freq_base: 500_000.0,
            factor: 128.0,
            original_ctx: 8192,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: LAGUNA_MSCALE as f32,
            n_rot: 64,
        }
    }

    #[test]
    fn rope_plain_hand_computed() {
        // theta=10000, n_rot=4, head_dim=4: pairs (0,2) with inv_freq 1.0 and
        // (1,3) with inv_freq 10000^(-1/2) = 0.01.
        let dev = Device::Cpu;
        let rope = Rope::new(&RopeKind::Plain { freq_base: 10_000.0, n_rot: 4 }, 8, &dev).unwrap();

        let pos = 3usize;
        let (f0, f1) = (1.0f64, 0.01f64);
        let (c0, s0) = ((pos as f64 * f0).cos() as f32, (pos as f64 * f0).sin() as f32);
        let (c1, s1) = ((pos as f64 * f1).cos() as f32, (pos as f64 * f1).sin() as f32);

        let cos = cos_row(&rope, pos);
        let sin = sin_row(&rope, pos);
        assert!((cos[0] - c0).abs() < 1e-6 && (sin[0] - s0).abs() < 1e-6);
        assert!((cos[1] - c1).abs() < 1e-6 && (sin[1] - s1).abs() < 1e-6);

        // A single query vector rotated at position 3.
        let x = vec![0.5f32, -1.0, 2.0, 0.25];
        let q = Tensor::from_vec(x.clone(), (1, 1, 4), &dev).unwrap();
        let (out, _) = rope.apply(&q, &q, pos).unwrap();
        let out = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let expect = vec![
            x[0] * c0 - x[2] * s0,
            x[1] * c1 - x[3] * s1,
            x[0] * s0 + x[2] * c0,
            x[1] * s1 + x[3] * c1,
        ];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-6, "got {out:?} expected {expect:?}");
        }
    }

    #[test]
    fn yarn_pass_through_dims_untouched() {
        // Dims n_rot..head_dim (64..128) must be bit-identical after apply.
        let dev = Device::Cpu;
        let rope = Rope::new(&laguna_yarn(), 32, &dev).unwrap();

        let n_head = 3;
        let seq = 5;
        let head_dim = 128;
        let data: Vec<f32> = (0..n_head * seq * head_dim).map(|i| (i as f32) * 0.001 - 0.5).collect();
        let q = Tensor::from_vec(data, (n_head, seq, head_dim), &dev).unwrap();
        let k = q.clone();

        let (out_q, _) = rope.apply(&q, &k, 7).unwrap();

        let before = q.narrow(2, 64, 64).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let after = out_q.narrow(2, 64, 64).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(before, after, "pass-through dims must be bit-identical");
    }

    #[test]
    fn pos_shift_consistency() {
        // apply at pos=5 on one token equals row 5 of apply at pos=0 on six tokens.
        let dev = Device::Cpu;
        let rope = Rope::new(&laguna_yarn(), 16, &dev).unwrap();

        let head_dim = 128;
        let n_head = 2;
        let single: Vec<f32> = (0..n_head * head_dim).map(|i| ((i * 7 % 13) as f32) * 0.1 - 0.6).collect();

        // One token placed at position 5.
        let q1 = Tensor::from_vec(single.clone(), (n_head, 1, head_dim), &dev).unwrap();
        let (out1, _) = rope.apply(&q1, &q1, 5).unwrap();
        let out1 = out1.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Six tokens at positions 0..6, with the same vector in slot 5.
        let mut six = vec![0f32; n_head * 6 * head_dim];
        for h in 0..n_head {
            for d in 0..head_dim {
                six[h * 6 * head_dim + 5 * head_dim + d] = single[h * head_dim + d];
            }
        }
        let q6 = Tensor::from_vec(six, (n_head, 6, head_dim), &dev).unwrap();
        let (out6, _) = rope.apply(&q6, &q6, 0).unwrap();
        let row5 = out6.narrow(1, 5, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for (a, b) in out1.iter().zip(row5.iter()) {
            assert!((a - b).abs() < 1e-6, "pos-shift mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn yarn_sanity() {
        let dev = Device::Cpu;
        let rope = Rope::new(&laguna_yarn(), 8192, &dev).unwrap();

        // Position 0: rotation is identity scaled by mscale (cos = mscale, sin = 0).
        let cos0 = cos_row(&rope, 0);
        let sin0 = sin_row(&rope, 0);
        for &c in &cos0 {
            assert!((c as f64 - LAGUNA_MSCALE).abs() < 1e-6, "cos row 0 should equal mscale, got {c}");
        }
        for &s in &sin0 {
            assert!(s.abs() < 1e-6, "sin row 0 should be zero, got {s}");
        }

        // The lowest-frequency pair (j = 31) is fully interpolated for the Laguna
        // config (its ramp saturates to 1), so its angle is scaled down by exactly
        // `factor` relative to unscaled rope over the same geometry. Compare angles
        // via atan2, which cancels the shared mscale, at a position small enough to
        // avoid wraparound.
        let plain = Rope::new(&RopeKind::Plain { freq_base: 500_000.0, n_rot: 64 }, 8192, &dev).unwrap();
        let p = 4096usize;
        let (cy, sy) = (cos_row(&rope, p)[31], sin_row(&rope, p)[31]);
        let (cp, sp) = (cos_row(&plain, p)[31], sin_row(&plain, p)[31]);
        let theta_yarn = (sy as f64).atan2(cy as f64);
        let theta_plain = (sp as f64).atan2(cp as f64);

        assert!(theta_plain > 0.0, "plain angle should be a small positive value");
        assert!(
            (theta_yarn - theta_plain / 128.0).abs() < 1e-6,
            "yarn low-freq angle should be plain/factor: {theta_yarn} vs {}",
            theta_plain / 128.0
        );
        // And the raw rotation differs from plain (scaling actually engaged).
        assert!((theta_yarn - theta_plain).abs() > 1e-3, "yarn scaling must change the angle");
    }
}
