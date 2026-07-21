use std::time::Instant;

use anyhow::{Result, ensure};
use candle_core::Tensor;

use crate::model::LagunaModel;
use crate::sampler::Sampler;
use crate::tokenizer::LagunaTokenizer;

/// Prompt tokens are fed to the model in chunks of this size. Keeps prefill
/// attention/MoE tensors bounded while still amortizing per-forward overhead.
const PREFILL_CHUNK: usize = 512;

/// Chunked prefill (512-token chunks) + single-token decode loop with a
/// streaming callback. One GPU->CPU sync per decoded token (the logits
/// readback that feeds the CPU sampler).
pub struct Generator {
    model: LagunaModel,
    tokenizer: LagunaTokenizer,
    sampler: Sampler,
}

#[derive(Debug, Default, Clone)]
pub struct GenStats {
    pub prefill_tokens: usize,
    pub prefill_secs: f64,
    pub decode_tokens: usize,
    pub decode_secs: f64,
}

impl GenStats {
    pub fn prefill_tps(&self) -> f64 {
        if self.prefill_secs > 0.0 { self.prefill_tokens as f64 / self.prefill_secs } else { 0.0 }
    }

    pub fn decode_tps(&self) -> f64 {
        if self.decode_secs > 0.0 { self.decode_tokens as f64 / self.decode_secs } else { 0.0 }
    }
}

impl Generator {
    pub fn new(model: LagunaModel, tokenizer: LagunaTokenizer, sampler: Sampler) -> Self {
        Self { model, tokenizer, sampler }
    }

    /// Generate up to max_tokens from a rendered prompt, streaming text chunks
    /// to `on_text`. Stops on any EOG token.
    ///
    /// The prompt string is encoded as-is: BOS ownership lives with the caller
    /// (the chat template writes BOS as literal text; the raw-prompt CLI path
    /// prepends it). This method never injects a BOS of its own.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.model.reset_cache()?;
        let device = self.model.device().clone();

        let tokens = self.tokenizer.encode(prompt)?;
        ensure!(!tokens.is_empty(), "prompt encoded to zero tokens");

        // The Full KV cache errors if appended past max_ctx; validate the whole
        // budget (prompt + generation) up front with a clear message.
        let max_ctx = self.model.max_ctx();
        ensure!(
            tokens.len() + max_tokens <= max_ctx,
            "prompt ({} tokens) + max_tokens ({max_tokens}) exceeds max_ctx ({max_ctx}); \
             raise --max-ctx or lower -n",
            tokens.len()
        );

        // Benchmark mode (LAGUNA_BENCH): run a full prefill once to page the
        // touched weights into the GPU residency set and compile every pipeline,
        // then reset so the timed run below measures steady state rather than
        // one-time upload/compile costs. Off by default; never affects output.
        if std::env::var_os("LAGUNA_BENCH").is_some() {
            let mut warm_pos = 0usize;
            for chunk in tokens.chunks(PREFILL_CHUNK) {
                let input = Tensor::new(chunk, &device)?;
                let _ = self.model.forward(&input, warm_pos)?;
                warm_pos += chunk.len();
            }
            device.synchronize()?;
            self.model.reset_cache()?;
        }

        // ---- Prefill: feed the prompt in chunks, advancing the absolute pos.
        let prefill_start = Instant::now();
        let mut pos = 0usize;
        let mut logits = None;
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            let input = Tensor::new(chunk, &device)?;
            logits = Some(self.model.forward(&input, pos)?);
            pos += chunk.len();
        }
        // Force the queued prefill GPU work to complete before timing it (the
        // decode loop's per-token readback would otherwise absorb it).
        device.synchronize()?;
        let prefill_secs = prefill_start.elapsed().as_secs_f64();
        let mut logits = logits.expect("non-empty prompt produced logits");

        // ---- Decode: sample, stream, feed back, one token at a time.
        let decode_start = Instant::now();
        let mut stream = self.tokenizer.decode_stream();
        let mut decoded = 0usize;
        while decoded < max_tokens {
            let token = self.sampler.sample(&logits)?;
            if self.sampler.is_eog(token) {
                break;
            }
            if let Some(text) = stream.step(token)? {
                on_text(&text);
            }
            decoded += 1;
            if decoded == max_tokens {
                break; // hit the cap: no need to run another forward
            }
            let input = Tensor::new(&[token], &device)?;
            logits = self.model.forward(&input, pos)?;
            pos += 1;
        }
        device.synchronize()?;
        let decode_secs = decode_start.elapsed().as_secs_f64();

        Ok(GenStats {
            prefill_tokens: tokens.len(),
            prefill_secs,
            decode_tokens: decoded,
            decode_secs,
        })
    }
}
