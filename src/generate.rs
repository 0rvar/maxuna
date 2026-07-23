use std::time::Instant;

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};

use crate::dflash::DflashDrafter;
use crate::model::LagunaModel;
use crate::sampler::Sampler;
use crate::tokenizer::LagunaTokenizer;

/// Prompt tokens are fed to the model in chunks of this size. Keeps prefill
/// attention/MoE tensors bounded while still amortizing per-forward overhead.
const PREFILL_CHUNK: usize = 512;

/// Chunked prefill (512-token chunks) + single-token decode loop with a
/// streaming callback. One GPU->CPU sync per decoded token (the logits
/// readback that feeds the CPU sampler).
///
/// When a DFlash drafter is attached (`attach_drafter`), `generate` runs the
/// speculative decode loop instead; with the drafter absent it is byte-identical
/// to the pre-DFlash single-token loop.
pub struct Generator {
    model: LagunaModel,
    tokenizer: LagunaTokenizer,
    sampler: Sampler,
    /// The speculative drafter, or `None` for the plain single-token loop.
    drafter: Option<DflashDrafter>,
    spec_params: SpecParams,
}

/// Speculative-decode knobs (mirrors the fork's `common_params_speculative`).
#[derive(Debug, Clone)]
pub struct SpecParams {
    /// Max draft tokens per round (`--draft-max`); also clamped to `block_size-1`.
    pub draft_max: usize,
    /// Discard a round's whole draft if fewer than this many were collected
    /// (`--draft-min`).
    pub draft_min: usize,
    /// Stop collecting drafts at the first drafter token whose full-vocab softmax
    /// probability falls below this (`--draft-p-min`).
    pub draft_p_min: f32,
}

impl Default for SpecParams {
    fn default() -> Self {
        // p_min 0.5 measured best on this machine (2026-07-23): adaptive draft
        // length beats any fixed draft_max — code-gen 25.5 vs 18.4 tok/s base at
        // temp 1.0 (80% acceptance); p_min 0 with full blocks was a net LOSS
        // (9.1 tok/s, 13% acceptance — rejected tail verifies are pure waste).
        Self { draft_max: 15, draft_min: 0, draft_p_min: 0.5 }
    }
}

/// Per-run speculative-decode accounting.
#[derive(Debug, Default, Clone)]
pub struct SpecStats {
    /// Verify rounds executed (each commits >= 1 token).
    pub rounds: usize,
    /// Total draft tokens proposed across all rounds.
    pub drafted: usize,
    /// Total draft tokens accepted (matched the target's sample).
    pub accepted: usize,
}

impl SpecStats {
    /// Fraction of proposed drafts the target accepted (0.0 if none proposed).
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted > 0 { self.accepted as f64 / self.drafted as f64 } else { 0.0 }
    }
}

#[derive(Debug, Default, Clone)]
pub struct GenStats {
    pub prefill_tokens: usize,
    pub prefill_secs: f64,
    pub decode_tokens: usize,
    pub decode_secs: f64,
    /// True when generation stopped because the cancel poll returned true
    /// (rather than hitting an EOG token or the max_tokens cap).
    pub cancelled: bool,
    /// Speculative-decode stats, present only when a drafter was attached.
    pub spec: Option<SpecStats>,
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
        Self { model, tokenizer, sampler, drafter: None, spec_params: SpecParams::default() }
    }

    /// Attach a DFlash drafter so subsequent `generate` calls run speculative
    /// decoding. The target's spec taps come from the drafter's config via
    /// `DflashConfig::spec_tap_layers` (the enforced `target_layers -> l_out`
    /// translation), kept in `target_layers` order — the order the drafter's
    /// encoder concatenates them.
    pub fn attach_drafter(&mut self, drafter: DflashDrafter, params: SpecParams) -> Result<()> {
        let tap_layers = drafter.config().spec_tap_layers()?;
        self.model.set_spec_taps(Some(tap_layers));
        self.drafter = Some(drafter);
        self.spec_params = params;
        Ok(())
    }

    /// Longest prompt + generation budget the KV cache can hold. Callers can
    /// use this to report context headroom or validate input sizes up front.
    pub fn max_ctx(&self) -> usize {
        self.model.max_ctx()
    }

    /// Generate up to max_tokens from a rendered prompt, streaming text chunks
    /// to `on_text`. Stops on any EOG token, or early when `should_stop`
    /// returns true (polled once per prefill chunk and once per decoded token;
    /// an early stop is reported via `GenStats::cancelled`).
    ///
    /// The prompt string is encoded as-is: BOS ownership lives with the caller
    /// (the chat template writes BOS as literal text; the raw-prompt CLI path
    /// prepends it). This method never injects a BOS of its own.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        on_text: &mut dyn FnMut(&str),
        should_stop: &mut dyn FnMut() -> bool,
    ) -> Result<GenStats> {
        if self.drafter.is_some() {
            return self.generate_spec(prompt, max_tokens, on_text, should_stop);
        }
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
        let mut cancelled = false;
        let mut pos = 0usize;
        let mut logits = None;
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            if should_stop() {
                cancelled = true;
                break;
            }
            let input = Tensor::new(chunk, &device)?;
            logits = Some(self.model.forward(&input, pos)?);
            pos += chunk.len();
        }
        // Force the queued prefill GPU work to complete before timing it (the
        // decode loop's per-token readback would otherwise absorb it).
        device.synchronize()?;
        let prefill_secs = prefill_start.elapsed().as_secs_f64();
        let mut logits = match logits {
            Some(logits) if !cancelled => logits,
            // Cancelled during (or before) prefill: no logits to decode from.
            _ => {
                return Ok(GenStats {
                    prefill_tokens: pos,
                    prefill_secs,
                    cancelled: true,
                    ..GenStats::default()
                });
            }
        };

        // ---- Decode: sample, stream, feed back, one token at a time.
        let decode_start = Instant::now();
        let mut stream = self.tokenizer.decode_stream();
        let mut decoded = 0usize;
        while decoded < max_tokens {
            if should_stop() {
                cancelled = true;
                break;
            }
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
            cancelled,
            spec: None,
        })
    }

    /// Speculative decode with the attached DFlash drafter. A mirror of the
    /// fork's `common_speculative_impl_draft_dflash` (speculative.cpp:902-1219)
    /// wired into this engine's chunked prefill + streaming decode.
    ///
    /// Correctness property: with the same seed this consumes the main sampler's
    /// RNG exactly once per committed token, in the same sequence order plain
    /// decode would, so `--draft` output matches spec-off output token-for-token
    /// except where the batched-verify forward's numerics (QMatMul lm_head over a
    /// span, vs the decode mat-vec bypass at seq==1) flip a near-tie.
    fn generate_spec(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        on_text: &mut dyn FnMut(&str),
        should_stop: &mut dyn FnMut() -> bool,
    ) -> Result<GenStats> {
        // Destructure into per-field borrows so the drafter can stay borrowed
        // across target-model / sampler / tokenizer calls (distinct fields).
        let Self { model, tokenizer, sampler, drafter, spec_params } = self;
        let drafter = drafter.as_mut().expect("generate_spec requires a drafter");
        model.reset_cache()?;
        let device = model.device().clone();
        drafter.reset();
        let params = spec_params.clone();

        let tokens = tokenizer.encode(prompt)?;
        ensure!(!tokens.is_empty(), "prompt encoded to zero tokens");

        let max_ctx = model.max_ctx();
        ensure!(
            tokens.len() + max_tokens <= max_ctx,
            "prompt ({} tokens) + max_tokens ({max_tokens}) exceeds max_ctx ({max_ctx}); \
             raise --max-ctx or lower -n",
            tokens.len()
        );

        // Draft-block width is bounded by the trained block size (id_last plus
        // block_size-1 masks) and the drafter's mask token.
        let block_size = drafter.config().block_size;
        let mask_token_id = drafter.config().mask_token_id;
        let draft_max = params.draft_max.min(block_size.saturating_sub(1));

        // Benchmark warm-up (LAGUNA_BENCH): page weights + compile pipelines on a
        // throwaway prefill (target and drafter both), then reset. Never affects
        // output. Mirrors the plain-decode path so spec timings are steady-state.
        if std::env::var_os("LAGUNA_BENCH").is_some() {
            let mut warm_pos = 0usize;
            for chunk in tokens.chunks(PREFILL_CHUNK) {
                let input = Tensor::new(chunk, &device)?;
                let _ = model.forward(&input, warm_pos)?;
                let taps = model.take_spec_taps();
                let fused = drafter.encode(&taps)?;
                drafter.inject(&fused, warm_pos)?;
                warm_pos += chunk.len();
            }
            device.synchronize()?;
            model.reset_cache()?;
            drafter.reset();
        }

        // ---- Prefill: feed the prompt in chunks. After each chunk, fuse the
        // target taps through the drafter's encoder and inject them into the
        // drafter's KV cache at the same absolute positions.
        let prefill_start = Instant::now();
        let mut cancelled = false;
        let mut pos = 0usize;
        let mut logits = None;
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            if should_stop() {
                cancelled = true;
                break;
            }
            let input = Tensor::new(chunk, &device)?;
            logits = Some(model.forward(&input, pos)?);
            let taps = model.take_spec_taps();
            let fused = drafter.encode(&taps)?;
            drafter.inject(&fused, pos)?; // pos == drafter.committed_len()
            pos += chunk.len();
        }
        device.synchronize()?;
        let prefill_secs = prefill_start.elapsed().as_secs_f64();
        let logits = match logits {
            Some(logits) if !cancelled => logits,
            _ => {
                return Ok(GenStats {
                    prefill_tokens: pos,
                    prefill_secs,
                    cancelled: true,
                    spec: Some(SpecStats::default()),
                    ..GenStats::default()
                });
            }
        };
        debug_assert_eq!(drafter.committed_len(), tokens.len());

        // ---- Decode. `id_last` is the last sampled-but-not-yet-verified token,
        // living at absolute position `n_past`; the drafter's committed length
        // tracks `n_past` at every round boundary.
        let decode_start = Instant::now();
        let mut stream = tokenizer.decode_stream();
        let mut stats = SpecStats::default();
        let mut n_past = tokens.len();
        let mut decoded = 0usize;

        // A zero budget must not draw from the sampler at all (plain decode's
        // first sample lives inside its `while decoded < max_tokens` loop).
        if max_tokens == 0 {
            return Ok(GenStats {
                prefill_tokens: pos,
                prefill_secs,
                spec: Some(stats),
                ..GenStats::default()
            });
        }

        // First token: sampled from the last prefill logits, exactly as plain
        // decode does. It is emitted here; every later token is emitted as a
        // committed token of a verify round.
        let mut id_last = sampler.sample(&logits)?;
        let first_eog = sampler.is_eog(id_last);
        if !first_eog {
            if let Some(text) = stream.step(id_last)? {
                on_text(&text);
            }
            decoded = 1;
        }

        if !first_eog {
            'rounds: while decoded < max_tokens {
                if should_stop() {
                    cancelled = true;
                    break;
                }

                // ---- Draft: run the drafter over [id_last, MASK * n_draft] and
                // greedily read one token per masked position.
                let n_draft = draft_max.min(max_ctx - n_past - 1);
                let mut drafts: Vec<u32> = Vec::new();
                if n_draft > 0 {
                    let mut noise_ids = Vec::with_capacity(n_draft + 1);
                    noise_ids.push(id_last);
                    noise_ids.extend(std::iter::repeat(mask_token_id).take(n_draft));
                    let noise_embd = model.embed_ids(&noise_ids)?; // [n_draft+1, n_embd]
                    let hidden = drafter.draft_forward(&noise_embd, n_past)?; // [n_draft+1, n_embd]
                    // Rows 1..=n_draft are the masked positions' drafts (row 0 is
                    // id_last's own position, which the target verifies). Narrowing
                    // to those rows before lm_head is a row-OFFSET view; QLinear's
                    // forward now materializes such views before the quantized
                    // matmul (gguf.rs), so lm_head reads the intended rows rather
                    // than the base offset — the Metal QMatMul offset trap.
                    let hidden_masked = hidden.narrow(0, 1, n_draft)?;
                    let dlogits = model.lm_head(&hidden_masked)?; // [n_draft, vocab]
                    let (rows, vocab) = dlogits.dims2()?;
                    let flat = dlogits.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
                    for r in 0..rows {
                        let (tok, p) = argmax_softmax(&flat[r * vocab..(r + 1) * vocab]);
                        if p < params.draft_p_min {
                            break;
                        }
                        drafts.push(tok);
                    }
                    if drafts.len() < params.draft_min {
                        drafts.clear();
                    }
                }

                // Committed tokens for this round and the last sampled token.
                let committed: Vec<u32>;
                if drafts.is_empty() {
                    // No usable draft: a plain single-token decode step. This is
                    // the same target forward + sampler draw plain decode runs, so
                    // it stays byte-identical when drafting yields nothing.
                    let input = Tensor::new(&[id_last], &device)?;
                    let step_logits = model.forward(&input, n_past)?;
                    let taps = model.take_spec_taps(); // each [1, n_embd]
                    let fused = drafter.encode(&taps)?;
                    drafter.inject(&fused, n_past)?;
                    let s = sampler.sample(&step_logits)?;
                    n_past += 1;
                    stats.rounds += 1;
                    committed = vec![s];
                } else {
                    // ---- Verify: forward [id_last, drafts...] in one batch, snapshot
                    // the KV first so a partial accept can roll the span back.
                    let span = 1 + drafts.len();
                    let ckpt = model.kv_checkpoint(span)?;
                    let mut verify_ids = Vec::with_capacity(span);
                    verify_ids.push(id_last);
                    verify_ids.extend_from_slice(&drafts);
                    let vinput = Tensor::new(verify_ids.as_slice(), &device)?;
                    let logits_all = model.forward_all_logits(&vinput, n_past)?; // [span, vocab]
                    let taps = model.take_spec_taps(); // each [span, n_embd]
                    let logits_cpu = logits_all.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;

                    // Sample the target token per verify row in order (RNG advances
                    // once per committed token, matching plain decode), accepting
                    // matching drafts until the first mismatch, the bonus token, an
                    // EOG, or the max_tokens budget — never drawing past any of them.
                    let (m, accepted) = accept_drafts(&drafts, max_tokens - decoded, |i| {
                        let row = logits_cpu.narrow(0, i, 1)?;
                        let s = sampler.sample(&row)?;
                        Ok((s, sampler.is_eog(s)))
                    })?;

                    // Keep exactly the verify positions backing the committed
                    // tokens: every emitted token except the last is forwarded,
                    // the exit state plain decode leaves (an EOG or budget-capped
                    // final token is never retained in the caches).
                    let keep = accepted.len();
                    model.kv_rollback(&ckpt, keep)?;
                    let prefix: Vec<Tensor> =
                        taps.iter().map(|t| t.narrow(0, 0, keep)).collect::<candle_core::Result<_>>()?;
                    let fused = drafter.encode(&prefix)?;
                    drafter.inject(&fused, n_past)?;

                    n_past += keep;
                    stats.rounds += 1;
                    stats.drafted += drafts.len();
                    stats.accepted += m;
                    committed = accepted;
                }

                id_last = *committed.last().expect("a round always commits >= 1 token");

                // Emit the committed tokens, stopping at the first EOG or the cap.
                for &tok in &committed {
                    if sampler.is_eog(tok) {
                        break 'rounds;
                    }
                    if let Some(text) = stream.step(tok)? {
                        on_text(&text);
                    }
                    decoded += 1;
                    if decoded >= max_tokens {
                        break 'rounds;
                    }
                }
            }
        }

        device.synchronize()?;
        let decode_secs = decode_start.elapsed().as_secs_f64();

        Ok(GenStats {
            prefill_tokens: tokens.len(),
            prefill_secs,
            decode_tokens: decoded,
            decode_secs,
            cancelled,
            spec: Some(stats),
        })
    }
}

/// Walk a verify block's rows in order, sampling the target's token for each and
/// accepting matching drafts until the first mismatch (or after the bonus token
/// when every draft matched). The walk also stops WITHOUT drawing when `budget`
/// committed tokens have been collected, and right after committing an EOG —
/// plain decode would never draw past either point, and the RNG must advance
/// exactly once per committed token in sequence order (the property that keeps
/// spec-on output equal to spec-off under the same seed).
///
/// Returns `(m, committed)` where `m` counts the committed tokens that were
/// accepted drafts (for stats) and `committed` is every sampled token, in
/// order. `committed.len()` is also the number of verify-span positions the
/// caller must keep in the KV caches: uniformly, every emitted token except
/// the last has been forwarded — the same exit state plain decode leaves.
///
/// `sample_row(i)` draws the target token for verify row `i` with the main
/// sampler and reports whether it is an EOG; it is called for `i = 0, 1, ...`
/// up to and INCLUDING the stopping row and never beyond. `budget` must
/// be >= 1.
fn accept_drafts(
    drafts: &[u32],
    budget: usize,
    mut sample_row: impl FnMut(usize) -> Result<(u32, bool)>,
) -> Result<(usize, Vec<u32>)> {
    debug_assert!(budget >= 1, "accept_drafts requires a nonzero budget");
    let span = drafts.len() + 1;
    let mut committed = Vec::with_capacity(span.min(budget));
    let mut m = 0usize;
    for i in 0..span {
        if committed.len() >= budget {
            break;
        }
        let (s, eog) = sample_row(i)?;
        let matched = i < drafts.len() && s == drafts[i];
        committed.push(s);
        if matched {
            m += 1;
        }
        // A matched non-EOG draft continues the walk; anything else (mismatch,
        // the bonus row past the last draft, or an EOG) commits and stops.
        if !matched || eog {
            break;
        }
    }
    Ok((m, committed))
}

/// The argmax token of a full logit row and its full-vocab softmax probability.
/// The probability of the max-logit token is `1 / Σ_j exp(logit_j - max)`.
fn argmax_softmax(logits: &[f32]) -> (u32, f32) {
    let mut max = f32::NEG_INFINITY;
    let mut arg = 0usize;
    for (i, &l) in logits.iter().enumerate() {
        if l > max {
            max = l;
            arg = i;
        }
    }
    let sum: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
    (arg as u32, (1.0 / sum) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// EOG marker used by the tests: any token >= 900 "ends generation".
    fn test_eog(t: u32) -> bool {
        t >= 900
    }

    /// Drive `accept_drafts` with a fixed list of "target samples", tracking how
    /// many rows it read so we can assert it never samples past the stop point
    /// (the invariant that keeps RNG consumption aligned with plain decode).
    fn run_budget(drafts: &[u32], samples: &[u32], budget: usize) -> (usize, Vec<u32>, usize) {
        let calls = Cell::new(0usize);
        let (m, committed) = accept_drafts(drafts, budget, |i| {
            calls.set(calls.get() + 1);
            Ok((samples[i], test_eog(samples[i])))
        })
        .unwrap();
        (m, committed, calls.get())
    }

    fn run(drafts: &[u32], samples: &[u32]) -> (usize, Vec<u32>, usize) {
        run_budget(drafts, samples, usize::MAX)
    }

    // Every draft matches, so the target's bonus token past the last draft is a
    // free extra commit: m == drafts.len() and one more token than drafts.
    #[test]
    fn all_drafts_match_plus_bonus() {
        let (m, committed, calls) = run(&[10, 20, 30], &[10, 20, 30, 40]);
        assert_eq!(m, 3);
        assert_eq!(committed, vec![10, 20, 30, 40]);
        // Sampled row 0 (id_last's follow), rows 1..3 (drafts), row 3 (bonus).
        assert_eq!(calls, 4);
    }

    // The very first sampled token disagrees with draft 0: nothing is accepted,
    // and only that single correction is committed and sampled.
    #[test]
    fn first_token_mismatch() {
        let (m, committed, calls) = run(&[10, 20, 30], &[99, 20, 30, 40]);
        assert_eq!(m, 0);
        assert_eq!(committed, vec![99]);
        assert_eq!(calls, 1);
    }

    // A mismatch in the middle: drafts before it are accepted, the correction is
    // committed, and no rows past the mismatch are sampled.
    #[test]
    fn mid_block_mismatch() {
        let (m, committed, calls) = run(&[10, 20, 30], &[10, 20, 77, 40]);
        assert_eq!(m, 2);
        assert_eq!(committed, vec![10, 20, 77]);
        assert_eq!(calls, 3);
    }

    // Zero drafts (span 1): exactly one token is sampled and committed — this is
    // the plain single-token decode step expressed through the same walk.
    #[test]
    fn zero_drafts_single_commit() {
        let (m, committed, calls) = run(&[], &[42]);
        assert_eq!(m, 0);
        assert_eq!(committed, vec![42]);
        assert_eq!(calls, 1);
    }

    // A single draft that matches yields the draft plus its bonus token.
    #[test]
    fn single_draft_match() {
        let (m, committed, calls) = run(&[10], &[10, 55]);
        assert_eq!(m, 1);
        assert_eq!(committed, vec![10, 55]);
        assert_eq!(calls, 2);
    }

    // An accepted draft that is an EOG stops the walk immediately: no bonus row
    // is sampled (plain decode never draws past an EOG), and the EOG is the
    // last committed token so the caller's keep-count excludes it from the KV.
    #[test]
    fn accepted_eog_draft_stops_without_bonus() {
        let (m, committed, calls) = run(&[10, 900, 30], &[10, 900, 30, 40]);
        assert_eq!(m, 2);
        assert_eq!(committed, vec![10, 900]);
        assert_eq!(calls, 2);
    }

    // An EOG arriving as the correction (mismatch) also stops the walk — same
    // stopping row a mismatch alone would produce, no extra draws.
    #[test]
    fn eog_correction_stops() {
        let (m, committed, calls) = run(&[10, 20], &[10, 901, 30]);
        assert_eq!(m, 1);
        assert_eq!(committed, vec![10, 901]);
        assert_eq!(calls, 2);
    }

    // The budget caps commits BEFORE the draw: with 2 tokens of budget and every
    // draft matching, only rows 0 and 1 are ever sampled.
    #[test]
    fn budget_stops_before_sampling() {
        let (m, committed, calls) = run_budget(&[10, 20, 30], &[10, 20, 30, 40], 2);
        assert_eq!(m, 2);
        assert_eq!(committed, vec![10, 20]);
        assert_eq!(calls, 2);
    }

    // Budget of 1 commits exactly one token regardless of how many drafts match.
    #[test]
    fn budget_one_single_commit() {
        let (m, committed, calls) = run_budget(&[10, 20, 30], &[10, 20, 30, 40], 1);
        assert_eq!(m, 1);
        assert_eq!(committed, vec![10]);
        assert_eq!(calls, 1);
    }

    // argmax_softmax picks the highest logit; its probability rises as that logit
    // dominates the rest of the row.
    #[test]
    fn argmax_softmax_basics() {
        let (tok, p) = argmax_softmax(&[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(tok, 0);
        assert!((p - 0.25).abs() < 1e-6, "uniform row prob {p}");

        let (tok, p) = argmax_softmax(&[1.0, 2.0, 100.0, 3.0]);
        assert_eq!(tok, 2);
        assert!(p > 0.999, "dominant-logit prob {p} should be ~1");
    }
}
