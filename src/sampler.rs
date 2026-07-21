use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};

/// Sampling per generation_config.json defaults: temp 1.0, top_k 20, top_p 1.0.
/// Wraps candle's LogitsProcessor (TopKThenTopP) + dual-EOG stop detection.
pub struct Sampler {
    processor: LogitsProcessor,
    eog: Vec<u32>,
}

pub struct SamplerOptions {
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub seed: u64,
}

impl Default for SamplerOptions {
    fn default() -> Self {
        Self { temperature: 1.0, top_k: 20, top_p: 1.0, seed: 42 }
    }
}

impl Sampler {
    pub fn new(opts: SamplerOptions, eog_tokens: Vec<u32>) -> Self {
        // A zero (or negative) temperature, or a top-k that keeps at most one
        // candidate, collapses to greedy decoding; otherwise apply top-k then
        // top-p filtering at the configured temperature.
        let sampling = if opts.temperature <= 0.0 || opts.top_k <= 1 {
            Sampling::ArgMax
        } else {
            Sampling::TopKThenTopP { k: opts.top_k, p: opts.top_p, temperature: opts.temperature }
        };
        Self { processor: LogitsProcessor::from_sampling(opts.seed, sampling), eog: eog_tokens }
    }

    /// logits: [vocab] f32 on any device; reads back and samples on CPU.
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let logits = logits.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        Ok(self.processor.sample(&logits)?)
    }

    pub fn is_eog(&self, token: u32) -> bool {
        self.eog.contains(&token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(values: &[f32]) -> Tensor {
        Tensor::new(values, &Device::Cpu).unwrap()
    }

    // A fixed seed and identical logits must produce the same token every time,
    // so generation is reproducible run-to-run.
    #[test]
    fn seeded_sampling_is_deterministic() {
        let opts = || SamplerOptions { temperature: 1.0, top_k: 20, top_p: 1.0, seed: 1234 };
        let l = logits(&[0.5, 1.5, 0.2, 2.5, 1.0]);
        let a = Sampler::new(opts(), vec![]).sample(&l).unwrap();
        let b = Sampler::new(opts(), vec![]).sample(&l).unwrap();
        assert_eq!(a, b);
    }

    // Temperature 0 is greedy: the highest-logit id is always chosen regardless
    // of seed.
    #[test]
    fn temperature_zero_is_argmax() {
        let l = logits(&[0.1, 0.2, 5.0, 0.3]);
        for seed in [0u64, 1, 99] {
            let mut s = Sampler::new(
                SamplerOptions { temperature: 0.0, top_k: 20, top_p: 1.0, seed },
                vec![],
            );
            assert_eq!(s.sample(&l).unwrap(), 2);
        }
    }

    // top_k of 0 or 1 also collapses to greedy.
    #[test]
    fn top_k_one_is_argmax() {
        let l = logits(&[0.1, 4.0, 0.2, 0.3]);
        let mut s =
            Sampler::new(SamplerOptions { temperature: 1.0, top_k: 1, top_p: 1.0, seed: 7 }, vec![]);
        assert_eq!(s.sample(&l).unwrap(), 1);
    }

    // With top_k = 2, only the two highest-logit ids may ever be drawn, no matter
    // how the mass is split across the rest of the vocabulary.
    #[test]
    fn top_k_restricts_to_top_two() {
        // Highest two logits are at indices 0 (3.0) and 2 (2.0).
        let l = logits(&[3.0, 0.1, 2.0, 0.2]);
        let mut s = Sampler::new(
            SamplerOptions { temperature: 1.0, top_k: 2, top_p: 1.0, seed: 2024 },
            vec![],
        );
        for _ in 0..64 {
            let id = s.sample(&l).unwrap();
            assert!(id == 0 || id == 2, "drew id {id}, outside the top-2 set {{0, 2}}");
        }
    }

    #[test]
    fn is_eog_matches_configured_tokens() {
        let s = Sampler::new(SamplerOptions::default(), vec![2, 24]);
        assert!(s.is_eog(2));
        assert!(s.is_eog(24));
        assert!(!s.is_eog(23));
    }
}
