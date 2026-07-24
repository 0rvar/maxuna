use std::time::Instant;

use anyhow::{Result, ensure};
use candle_core::{D, DType, Device, Tensor};

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
    /// Forced-reasoning floor: while fewer than this many tokens have been
    /// decoded in a call, `</think>` and the EOG ids are banned from sampling,
    /// holding the model inside the `<think>` block the chat template opened.
    /// 0 (the default) leaves the model free to close the block immediately —
    /// which the hybrid reasoner does on conversational prompts. Only
    /// meaningful when the prompt ends with `<assistant><think>`.
    min_think: usize,
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
    /// Auto-pause margin (`--draft-pause-margin`). Speculation pauses when its
    /// measured wall-clock cost per committed token exceeds a plain decode
    /// step's cost times this factor. `0.0` disables auto-pause entirely
    /// (always draft — the pre-auto-pause behavior).
    pub pause_margin: f32,
}

impl Default for SpecParams {
    fn default() -> Self {
        // p_min 0.5 measured best on this machine (2026-07-23): adaptive draft
        // length beats any fixed draft_max — code-gen 25.5 vs 18.4 tok/s base at
        // temp 1.0 (80% acceptance); p_min 0 with full blocks was a net LOSS
        // (9.1 tok/s, 13% acceptance — rejected tail verifies are pure waste).
        Self { draft_max: 15, draft_min: 0, draft_p_min: 0.5, pause_margin: 1.0 }
    }
}

/// Per-run speculative-decode accounting. `GenStats::decode_tokens` counts only
/// EMITTED tokens (accepted drafts plus each round's final committed token); the
/// draft positions a round proposes and the target rejects cost wall time but
/// are never emitted, so `decode_tps` is the true end-to-end throughput. These
/// fields expose where that wall time goes.
#[derive(Debug, Default, Clone)]
pub struct SpecStats {
    /// Verify rounds executed (each commits >= 1 token).
    pub rounds: usize,
    /// Total draft tokens proposed across all rounds.
    pub drafted: usize,
    /// Total draft tokens accepted (matched the target's sample).
    pub accepted: usize,
    /// Rounds that ran a plain single-token step because auto-pause was active:
    /// paused-plain rounds, plus probe rounds whose draft came up empty (they
    /// too ran plain). Real drafting probes are excluded — those still
    /// speculate. Zero when auto-pause never engaged (good economics, or
    /// `pause_margin == 0`).
    pub paused_rounds: usize,
    /// Total target-forward positions across all rounds: the anchor row (the
    /// last committed token, re-forwarded) plus each round's drafts. A plain
    /// round contributes 1. Equals `rounds + drafted` when every round drafts.
    pub verify_positions: usize,
    /// Total wall ms spent in the draft phase (drafter forward + lm_head +
    /// argmax collection). A round that drafts nothing but still ran the drafter
    /// (all tokens below `draft_p_min`) charges its wasted draft time here.
    pub draft_ms: f64,
    /// Total wall ms spent in the commit phase (the target forward — batched
    /// verify or the plain single-token step — plus acceptance sampling).
    pub verify_ms: f64,
}

impl SpecStats {
    /// Fraction of proposed drafts the target accepted (0.0 if none proposed).
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted > 0 { self.accepted as f64 / self.drafted as f64 } else { 0.0 }
    }

    /// Draft positions the target rejected (proposed but not accepted). Pure
    /// wall-time cost — these are never emitted.
    pub fn rejected(&self) -> usize {
        self.drafted - self.accepted
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

/// What kind of round the [`PauseController`] wants next. `Spec`/`Probe` draft
/// and verify; `ForcedPlain`/`PausedPlain` run a plain single-token step. The
/// distinction between the two plain kinds is only for accounting: a forced
/// plain is an Active round sacrificed to keep the plain-cost EMA fresh, while
/// a paused plain is a round where speculation is switched off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoundKind {
    /// Active speculative round.
    Spec,
    /// Active round forced to plain decode to refresh `ema_plain_ms`.
    ForcedPlain,
    /// Paused round run plain because speculation is not paying off.
    PausedPlain,
    /// Paused round run speculative to re-measure whether it pays off now.
    Probe,
}

/// Wall-clock auto-pause for speculative decoding: a pure, GPU-free state
/// machine that decides, per round, whether to speculate or fall back to plain
/// decode so `--draft` never loses to plain decode on low-acceptance text.
///
/// It compares the speculative cost per committed token against the cost of one
/// plain decode step (`ema_plain_ms`), all EMAs at `α = 0.25`. The plain
/// comparator is the empty-draft fallback path, which carries the same tap
/// encode/inject overhead a paused round pays, so the two are like-for-like.
///
/// The speculative cost is a RATE, not a mean of per-round ratios: two EMAs
/// (`ema_spec_round_ms`, `ema_spec_committed`) are tracked separately and the
/// decision uses their quotient. `EMA(ms) / EMA(tok)` estimates the
/// exponentially-windowed aggregate `Σms / Σtok`, which is the quantity that
/// actually equals throughput cost. A single EMA of `round_ms / committed`
/// would be a mean-of-ratios that over-weights low-commit rounds: on a measured
/// code-gen greedy run rounds are bimodal (~25 ms/tok for 11-commit rounds vs
/// ~350 ms/tok for 1-commit mismatch rounds), and the mean-of-ratios crossed
/// plain's ~58 ms even though the aggregate spec cost was ~42 ms/tok — pausing
/// 76 of 99 rounds on a workload where speculation wins by 1.4x. The quotient
/// form does not: a 1-commit round contributes 1 to the token EMA, so its cost
/// is amortized the same way throughput amortizes it.
///
/// Acceptance-count thresholds were rejected deliberately: the break-even
/// committed-tokens-per-round varies with draft span (~2.5 at span 2 to ~3.8 at
/// span 16), so a fixed count misfires. Round economics must be judged on the
/// clock, and the EMA absorbs single-round noise (one probe is not trusted on
/// its own — its cost is folded in and the decision made on the smoothed value).
///
/// Recovery from a stale pause is deliberately EAGER, because the cost
/// asymmetry is inverted: a wrongly-held pause loses 30%+ throughput for the
/// rest of the run, while a wrongly-lifted pause loses only a few rounds before
/// the (still-warm) pause logic re-pauses. A pause can go stale when the samples
/// that triggered it came from a transient world — GPU contention from a second
/// process, a hard text stretch — that has since passed. Three mechanisms fight
/// that: (1) a probe EVENT is a PAIR of consecutive speculative rounds, not one,
/// so the resume decision rests on up to two live samples (an empty-draft probe
/// round contributes none, and a pair that gathers zero real samples decides
/// nothing and simply retries; a pair with one real sample decides on it);
/// (2) probe samples fold at `α = 0.5` (vs 0.25) because while paused `ema_spec`
/// is stale by construction and fresh probe evidence must dominate memory;
/// (3) resume triggers on the pair's OWN aggregate rate (`Σround_ms / Σcommitted`
/// over its real rounds) crossing back under the threshold — regardless of the
/// smoothed EMA still being poisoned — and on such a resume the spec EMAs are
/// SNAPPED to the pair's values, dropping the poisoned history so they rebuild
/// from live active rounds. A pair that fails still doubles the backoff (a failed
/// pair is decent evidence spec still loses).
///
/// Warm-up itself is accelerated: since only plain rounds refresh `ema_plain`,
/// an all-drafting run would otherwise not collect its second plain sample until
/// the `FORCE_PLAIN_EVERY` (32) cadence fires twice — so a short net-negative
/// generation would never reach the `warm()` gate and never pause. Until the
/// plain warm-up is met the forced-plain cadence tightens to
/// `WARMUP_FORCE_PLAIN_EVERY` (every 4th Active round); a forced plain costs the
/// same as plain decode, so early sampling is free.
struct PauseController {
    /// Pause when the speculative cost per token exceeds `ema_plain_ms * margin`.
    /// `0.0` disables auto-pause: every round speculates.
    margin: f64,
    /// EMA of speculative round wall-ms. Paired with `ema_spec_committed`; the
    /// per-token cost is their quotient (see the type doc for why not a single
    /// per-round-ratio EMA).
    ema_spec_round_ms: Option<f64>,
    /// EMA of committed tokens per speculative round.
    ema_spec_committed: Option<f64>,
    /// EMA of a plain single-token round's wall-ms.
    ema_plain_ms: Option<f64>,
    /// Speculative samples folded in (warm-up gate).
    spec_samples: usize,
    /// Plain samples folded in (warm-up gate).
    plain_samples: usize,
    state: PauseState,
    /// Active-state rounds since the last plain sample; at the current forced-
    /// plain cadence (accelerated to `WARMUP_FORCE_PLAIN_EVERY` until the plain
    /// warm-up is met, then `FORCE_PLAIN_EVERY`) the next round is forced plain
    /// so `ema_plain_ms` never goes stale on a run where speculation always wins.
    rounds_since_plain: usize,
    /// Paused-state committed tokens since the last probe event; at `backoff` the
    /// next round starts a probe pair.
    tokens_since_probe: usize,
    /// The in-flight probe pair's accumulated real samples, or `None` between
    /// events. Drives the eager-resume decision once the pair completes.
    probe_event: Option<ProbeEvent>,
}

/// A probe pair in flight: up to [`PauseController::PROBE_PAIR`] consecutive
/// probe rounds, accumulating only the REAL (non-empty-draft) rounds' costs. An
/// empty-draft probe round advances `rounds` but adds no sample, so a pair that
/// never drafts decides nothing.
#[derive(Debug, Clone, Copy, Default)]
struct ProbeEvent {
    /// Probe rounds run in this event so far (real or empty).
    rounds: usize,
    /// Σ round_ms over the event's real rounds.
    real_ms: f64,
    /// Σ committed over the event's real rounds.
    real_committed: usize,
    /// Count of real (non-empty-draft) rounds.
    real: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PauseState {
    Active,
    /// Probe every `backoff` committed tokens; the backoff doubles on each
    /// failed probe up to [`PauseController::BACKOFF_CAP`].
    Paused { backoff: usize },
}

impl PauseController {
    const ALPHA: f64 = 0.25;
    /// Fold weight for probe-pair samples: heavier than `ALPHA` so fresh probe
    /// evidence dominates the stale, paused `ema_spec` (see the type doc).
    const PROBE_ALPHA: f64 = 0.5;
    /// Minimum speculative + plain samples before the first pause decision.
    const WARMUP_SPEC: usize = 8;
    const WARMUP_PLAIN: usize = 2;
    /// Active rounds between forced plain rounds (keeps `ema_plain_ms` fresh).
    const FORCE_PLAIN_EVERY: usize = 32;
    /// Tighter forced-plain cadence until the plain warm-up is met, so a short
    /// all-drafting run reaches `warm()` fast enough to pause.
    const WARMUP_FORCE_PLAIN_EVERY: usize = 4;
    /// Probe rounds per probe event (a probe pair).
    const PROBE_PAIR: usize = 2;
    /// Committed tokens between probe events while paused; doubles per failed
    /// probe pair.
    const BACKOFF_START: usize = 32;
    const BACKOFF_CAP: usize = 256;

    fn new(margin: f64) -> Self {
        Self {
            margin,
            ema_spec_round_ms: None,
            ema_spec_committed: None,
            ema_plain_ms: None,
            spec_samples: 0,
            plain_samples: 0,
            state: PauseState::Active,
            rounds_since_plain: 0,
            tokens_since_probe: 0,
            probe_event: None,
        }
    }

    /// Forced-plain cadence for Active rounds: tighter until the plain warm-up
    /// is met so a short run collects plain samples in time to pause.
    fn force_plain_every(&self) -> usize {
        if self.plain_samples < Self::WARMUP_PLAIN {
            Self::WARMUP_FORCE_PLAIN_EVERY
        } else {
            Self::FORCE_PLAIN_EVERY
        }
    }

    /// The kind of round to run next. Pure read of the current state; the round
    /// is executed by the caller, which then feeds the outcome back via
    /// [`record`](Self::record).
    fn next_kind(&self) -> RoundKind {
        if self.margin == 0.0 {
            return RoundKind::Spec;
        }
        match self.state {
            PauseState::Active => {
                if self.rounds_since_plain >= self.force_plain_every() {
                    RoundKind::ForcedPlain
                } else {
                    RoundKind::Spec
                }
            }
            PauseState::Paused { backoff } => {
                // Mid-pair: the pair's remaining probe rounds always follow the
                // first. Otherwise probe once the backoff's worth of tokens have
                // elapsed.
                if self.probe_event.is_some() || self.tokens_since_probe >= backoff {
                    RoundKind::Probe
                } else {
                    RoundKind::PausedPlain
                }
            }
        }
    }

    /// Fold a completed round's cost into the EMAs and update the pause state.
    /// `round_ms` is the whole round's wall time; `draft_ms` is the drafter
    /// portion of it. `ran_spec` is whether the round actually speculated (a
    /// `Spec`/`Probe` round with an empty draft falls back to a plain step, so
    /// it folds into the plain EMA, not the speculative one). `committed` is the
    /// number of tokens the round committed (always `>= 1`).
    fn record(&mut self, kind: RoundKind, round_ms: f64, draft_ms: f64, committed: usize, ran_spec: bool) {
        if ran_spec {
            debug_assert!(committed >= 1, "a speculative round commits >= 1 token");
            // Track the ms and token EMAs separately; the per-token cost is
            // their quotient (a rate, not a mean of ratios — see the type doc).
            // Probe samples fold heavier so a stale pause's poisoned `ema_spec`
            // yields to fresh evidence.
            let alpha = if kind == RoundKind::Probe { Self::PROBE_ALPHA } else { Self::ALPHA };
            self.ema_spec_round_ms = Some(Self::ema(self.ema_spec_round_ms, round_ms, alpha));
            self.ema_spec_committed = Some(Self::ema(self.ema_spec_committed, committed as f64, alpha));
            self.spec_samples += 1;
        } else {
            // An empty-draft round still ran the drafter, but that cost is not
            // part of plain decode — fold only the target-forward portion
            // (`round_ms - draft_ms`) so the plain comparator is not inflated by
            // wasted drafting (which would delay pausing exactly on the
            // low-acceptance text auto-pause exists to catch). A genuine plain
            // round has `draft_ms == 0`, so this is a no-op there. Plain samples
            // always fold at the base rate; only `ema_spec` goes stale while
            // paused, so only probe SPEC samples get the heavier weight.
            self.ema_plain_ms = Some(Self::ema(self.ema_plain_ms, (round_ms - draft_ms).max(0.0), Self::ALPHA));
            self.plain_samples += 1;
        }

        match self.state {
            PauseState::Active => {
                // A plain round (forced or an empty-draft fallback) just
                // refreshed the plain EMA; otherwise count down to the next
                // forced plain.
                if ran_spec {
                    self.rounds_since_plain += 1;
                } else {
                    self.rounds_since_plain = 0;
                }
                if self.should_pause() {
                    self.state = PauseState::Paused { backoff: Self::BACKOFF_START };
                    self.tokens_since_probe = 0;
                }
            }
            PauseState::Paused { backoff } => {
                self.tokens_since_probe += committed;
                if kind == RoundKind::Probe {
                    // Accumulate this probe round into the in-flight pair, starting
                    // one if needed. An empty-draft probe round (ran_spec == false)
                    // advances the round count but contributes NO sample and, on
                    // its own, decides nothing — it is accounting-wise a paused
                    // plain round that retries. Only real rounds move the decision.
                    let mut ev = self.probe_event.take().unwrap_or_default();
                    ev.rounds += 1;
                    if ran_spec {
                        ev.real_ms += round_ms;
                        ev.real_committed += committed;
                        ev.real += 1;
                    }
                    if ev.rounds < Self::PROBE_PAIR {
                        // Pair still in flight; wait for the remaining round.
                        self.probe_event = Some(ev);
                    } else if ev.real > 0 {
                        // Pair complete with live evidence: decide on the pair's
                        // OWN aggregate rate over its real rounds, not the smoothed
                        // EMA, which may still be poisoned.
                        self.tokens_since_probe = 0;
                        let pair_rate = ev.real_ms / ev.real_committed as f64;
                        let resume = self
                            .ema_plain_ms
                            .is_some_and(|plain| pair_rate <= plain * self.margin);
                        if resume {
                            // Drop the poisoned history: reseed the spec EMAs to
                            // the pair's per-real-round means (their quotient is
                            // the pair rate) so they rebuild from live rounds.
                            self.ema_spec_round_ms = Some(ev.real_ms / ev.real as f64);
                            self.ema_spec_committed = Some(ev.real_committed as f64 / ev.real as f64);
                            self.state = PauseState::Active;
                            self.rounds_since_plain = 0;
                        } else {
                            self.state =
                                PauseState::Paused { backoff: (backoff * 2).min(Self::BACKOFF_CAP) };
                        }
                    }
                    // else: the pair completed with zero real samples (the drafter
                    // came up empty both rounds) — decide nothing, leave the
                    // backoff and tokens_since_probe untouched, and retry a fresh
                    // pair on the next round (probe_event stays None).
                }
            }
        }
    }

    fn ema(prev: Option<f64>, sample: f64, alpha: f64) -> f64 {
        match prev {
            Some(p) => alpha * sample + (1.0 - alpha) * p,
            None => sample,
        }
    }

    fn warm(&self) -> bool {
        self.spec_samples >= Self::WARMUP_SPEC && self.plain_samples >= Self::WARMUP_PLAIN
    }

    /// Speculative wall-ms per committed token: `EMA(round_ms) / EMA(committed)`.
    /// `None` until at least one speculative round has been recorded.
    fn spec_cost(&self) -> Option<f64> {
        match (self.ema_spec_round_ms, self.ema_spec_committed) {
            (Some(ms), Some(tok)) if tok > 0.0 => Some(ms / tok),
            _ => None,
        }
    }

    fn should_pause(&self) -> bool {
        match (self.spec_cost(), self.ema_plain_ms) {
            (Some(spec), Some(plain)) if self.margin > 0.0 && self.warm() => spec > plain * self.margin,
            _ => false,
        }
    }
}

impl Generator {
    pub fn new(model: LagunaModel, tokenizer: LagunaTokenizer, sampler: Sampler) -> Self {
        Self {
            model,
            tokenizer,
            sampler,
            drafter: None,
            spec_params: SpecParams::default(),
            min_think: 0,
        }
    }

    /// Set the forced-reasoning floor (see the `min_think` field). Applies to
    /// every subsequent `generate` call, on both the plain and speculative
    /// decode paths.
    pub fn set_min_think(&mut self, n: usize) {
        self.min_think = n;
    }

    /// The ids banned from sampling while the forced-reasoning floor is in
    /// effect: `</think>` plus the EOG ids (banning EOG too keeps the forcing
    /// window from ending the reply outright). Empty when the floor is 0, so
    /// the default path stays on the unmasked sampler.
    fn think_exit_ban(&self) -> Vec<u32> {
        if self.min_think == 0 {
            return Vec::new();
        }
        let mut ban = vec![LagunaTokenizer::THINK_CLOSE];
        ban.extend_from_slice(self.sampler.eog_ids());
        ban
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
        // A floor at or past the token budget would ban `</think>`/EOG for the
        // whole generation: the reply is all reasoning, and the chat REPL
        // (which splits on the `</think>` literal) would show nothing and file
        // the raw reasoning into history as content. Reject it up front.
        ensure!(
            self.min_think == 0 || self.min_think < max_tokens,
            "min_think ({}) must be below max_tokens ({max_tokens}) or the think block \
             can never close",
            self.min_think
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
        let think_exit_ban = self.think_exit_ban();
        let mut stream = self.tokenizer.decode_stream();
        let mut decoded = 0usize;
        while decoded < max_tokens {
            if should_stop() {
                cancelled = true;
                break;
            }
            let token = if decoded < self.min_think {
                self.sampler.sample_masked(&logits, &think_exit_ban)?
            } else {
                self.sampler.sample(&logits)?
            };
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
        let think_exit_ban = self.think_exit_ban();
        let min_think = self.min_think;
        // Destructure into per-field borrows so the drafter can stay borrowed
        // across target-model / sampler / tokenizer calls (distinct fields).
        let Self { model, tokenizer, sampler, drafter, spec_params, .. } = self;
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
        // Same floor-vs-budget guard as the plain path (see generate()).
        ensure!(
            min_think == 0 || min_think < max_tokens,
            "min_think ({min_think}) must be below max_tokens ({max_tokens}) or the think \
             block can never close",
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

        // Wall-clock auto-pause: never let speculation lose to plain decode on
        // low-acceptance text. Fed the measured cost of each round below; the
        // LAGUNA_BENCH warm-up prefill happened before this and is not a sample.
        let mut pause = PauseController::new(params.pause_margin as f64);

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
        // committed token of a verify round. Emitted index 0, so the forced-
        // reasoning ban applies whenever the floor is nonzero.
        let mut id_last = if min_think > 0 {
            sampler.sample_masked(&logits, &think_exit_ban)?
        } else {
            sampler.sample(&logits)?
        };
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

                // The controller decides this round's kind before any GPU work;
                // paused rounds skip drafting and run the plain fallback below.
                let round_kind = pause.next_kind();
                let want_draft = matches!(round_kind, RoundKind::Spec | RoundKind::Probe);
                let round_start = Instant::now();

                // ---- Draft: run the drafter over [id_last, MASK * n_draft] and
                // greedily read one token per masked position.
                let n_draft = draft_max.min(max_ctx - n_past - 1);
                let mut drafts: Vec<u32> = Vec::new();
                if want_draft && n_draft > 0 {
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
                    // Per-row argmax id + softmax prob of that argmax, reduced
                    // on-GPU so only the tiny [n_draft] id/prob arrays cross to
                    // the CPU — not the full [n_draft, vocab] logits. The p_min
                    // walk below is then a cheap CPU scan, identical in semantics
                    // to the old per-row full-vocab argmax_softmax.
                    let (ids, probs) = draft_argmax_probs(&dlogits)?;
                    for (&tok, &p) in ids.iter().zip(&probs) {
                        if p < params.draft_p_min {
                            break;
                        }
                        drafts.push(tok);
                    }
                    if drafts.len() < params.draft_min {
                        drafts.clear();
                    }
                }

                // Split the round timer at the draft/commit boundary. The draft
                // phase just ended at the `draft_argmax_probs` readback (a CPU
                // sync); the commit phase below ends at its own sampler readback,
                // so neither split adds a device sync. A round that ran the
                // drafter but kept no drafts still charges its draft time here.
                let draft_ms = round_start.elapsed().as_secs_f64() * 1000.0;

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
                    let s = if decoded < min_think {
                        sampler.sample_masked(&step_logits, &think_exit_ban)?
                    } else {
                        sampler.sample(&step_logits)?
                    };
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
                    // Row i commits emitted index `decoded + i`, so the forced-
                    // reasoning ban applies per row; a draft proposing a banned id
                    // simply mismatches the masked sample and is rejected.
                    let (m, accepted) = accept_drafts(&drafts, max_tokens - decoded, |i| {
                        let row = logits_cpu.narrow(0, i, 1)?;
                        let s = if decoded + i < min_think {
                            sampler.sample_masked(&row, &think_exit_ban)?
                        } else {
                            sampler.sample(&row)?
                        };
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

                // Feed the round's wall-clock cost to the pause controller. A
                // round's kv_rollback + encode + inject are queued but not yet
                // synced at this point, so without a barrier ~5-15ms of GPU tail
                // would bleed into the NEXT round's window and skew whichever EMA
                // that round feeds. One synchronize per round trims that CPU
                // run-ahead; the GPU queue is in-order so total GPU time is
                // unchanged, and the cost is negligible at the ~200ms round
                // cadence. A round that requested a draft but produced none ran
                // the plain fallback, so it counts as a plain sample (`ran_spec
                // == false`), with the wasted drafter time excluded via draft_ms.
                let ran_spec = !drafts.is_empty();
                device.synchronize()?;
                let round_ms = round_start.elapsed().as_secs_f64() * 1000.0;
                stats.draft_ms += draft_ms;
                stats.verify_ms += round_ms - draft_ms;
                stats.verify_positions += 1 + drafts.len();
                pause.record(round_kind, round_ms, draft_ms, committed.len(), ran_spec);
                // Count every round that ran plain because of the pause: the
                // paused-plain rounds, and any probe round whose draft came up
                // empty (it ran the plain fallback). Real drafting probes are
                // excluded — they speculated.
                if round_kind == RoundKind::PausedPlain
                    || (round_kind == RoundKind::Probe && !ran_spec)
                {
                    stats.paused_rounds += 1;
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

/// For each row of `logits` `[n, vocab]`, the argmax token id and the full-vocab
/// softmax probability of that argmax (`1 / Σ_j exp(logit_j - max)`). The reduce
/// runs on whatever device `logits` is on (the GPU in production) and only the
/// tiny `[n]` id and prob vectors are read back — not the `[n, vocab]` logits.
/// On an exact logit tie candle's argmax returns the FIRST maximal index; the
/// probability is derived from the max VALUE, so it is unaffected by which index
/// wins.
fn draft_argmax_probs(logits: &Tensor) -> Result<(Vec<u32>, Vec<f32>)> {
    let logits = logits.to_dtype(DType::F32)?;
    let max = logits.max_keepdim(D::Minus1)?; // [n, 1]
    // prob(argmax) = exp(max - logsumexp) = 1 / Σ_j exp(logit_j - max).
    let probs = logits.broadcast_sub(&max)?.exp()?.sum_keepdim(D::Minus1)?.recip()?; // [n, 1]
    let ids = logits.argmax_keepdim(D::Minus1)?; // [n, 1]
    let ids = ids.flatten_all()?.to_dtype(DType::U32)?.to_device(&Device::Cpu)?.to_vec1::<u32>()?;
    let probs = probs.flatten_all()?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    Ok((ids, probs))
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

    // ---- PauseController: a pure state machine, tested without any GPU by
    // feeding synthetic per-round costs. Bad economics = a speculative round
    // that costs more per committed token than a plain round; good economics =
    // the reverse.

    /// Drive the controller for `rounds`, faithfully following `next_kind` and
    /// feeding each round a synthetic cost: `spec_ms` over `committed` tokens
    /// for speculative rounds, `plain_ms` for plain rounds.
    fn drive(
        margin: f64,
        rounds: usize,
        spec_ms: f64,
        committed: usize,
        plain_ms: f64,
    ) -> PauseController {
        let mut c = PauseController::new(margin);
        for _ in 0..rounds {
            let kind = c.next_kind();
            match kind {
                RoundKind::Spec | RoundKind::Probe => c.record(kind, spec_ms, 0.0, committed, true),
                RoundKind::ForcedPlain | RoundKind::PausedPlain => c.record(kind, plain_ms, 0.0, 1, false),
            }
        }
        c
    }

    // Sustained bad economics (100 ms/spec-tok vs 40 ms/plain): the controller
    // pauses once warm-up is met (the forced-plain cadence supplies the plain
    // samples on an all-speculative run).
    #[test]
    fn pauses_under_sustained_bad_economics() {
        let c = drive(1.0, 100, 100.0, 1, 40.0);
        assert!(matches!(c.state, PauseState::Paused { .. }), "state {:?}", c.state);
    }

    // Sustained good economics (10 ms/spec-tok — 30 ms over 3 committed — vs
    // 50 ms/plain): speculation always pays, so the controller never pauses.
    #[test]
    fn stays_active_under_good_economics() {
        let c = drive(1.0, 100, 30.0, 3, 50.0);
        assert_eq!(c.state, PauseState::Active);
    }

    // Each failed probe PAIR doubles the backoff, capped at BACKOFF_CAP. A pair
    // is two consecutive probe rounds; the first stashes, the second decides.
    #[test]
    fn failed_probe_pairs_back_off_to_cap() {
        let mut c = PauseController::new(1.0);
        // Warm and paused with speculation clearly losing (100 > 40): 100 ms
        // per round over 1 committed token = 100 ms/tok.
        c.ema_spec_round_ms = Some(100.0);
        c.ema_spec_committed = Some(1.0);
        c.ema_plain_ms = Some(40.0);
        c.spec_samples = PauseController::WARMUP_SPEC;
        c.plain_samples = PauseController::WARMUP_PLAIN;
        c.state = PauseState::Paused { backoff: 32 };

        let mut backoff = 32;
        for expected in [64, 128, 256, 256, 256] {
            // The pair's aggregate (200 ms / 2 committed = 100 ms/tok) stays
            // above the margin: no resume, backoff doubles. The first probe only
            // stashes — the backoff must not change until the pair completes.
            c.record(RoundKind::Probe, 100.0, 0.0, 1, true);
            assert_eq!(c.state, PauseState::Paused { backoff }, "first probe must not decide");
            c.record(RoundKind::Probe, 100.0, 0.0, 1, true);
            assert_eq!(c.state, PauseState::Paused { backoff: expected });
            backoff = expected;
        }
    }

    // A probe PAIR whose own aggregate rate falls under the margin resumes to
    // Active and clears the backoff (the next pause starts at BACKOFF_START).
    #[test]
    fn resume_resets_backoff() {
        let mut c = PauseController::new(1.0);
        // Warm and paused with a stale, inflated spec cost (200 ms/tok) that a
        // live probe pair will contradict.
        c.ema_spec_round_ms = Some(200.0);
        c.ema_spec_committed = Some(1.0);
        c.ema_plain_ms = Some(40.0);
        c.spec_samples = PauseController::WARMUP_SPEC;
        c.plain_samples = PauseController::WARMUP_PLAIN;
        c.state = PauseState::Paused { backoff: 128 };

        // A cheap probe pair (2 x 30 ms / 1 = 30 ms/tok <= 40) resumes on its
        // own aggregate despite the stale EMA still reading 200.
        c.record(RoundKind::Probe, 30.0, 0.0, 1, true);
        assert_eq!(c.state, PauseState::Paused { backoff: 128 }, "first probe must not resume");
        c.record(RoundKind::Probe, 30.0, 0.0, 1, true);
        assert_eq!(c.state, PauseState::Active);

        // Economics sour again: the fresh pause starts from BACKOFF_START, not
        // the old 128 — the backoff was reset on resume.
        c.record(RoundKind::Spec, 400.0, 0.0, 1, true);
        assert_eq!(c.state, PauseState::Paused { backoff: PauseController::BACKOFF_START });
    }

    // margin 0 disables auto-pause: every round speculates and the state never
    // leaves Active however bad the economics get.
    #[test]
    fn margin_zero_never_pauses() {
        let mut c = PauseController::new(0.0);
        for _ in 0..100 {
            assert_eq!(c.next_kind(), RoundKind::Spec);
            c.record(RoundKind::Spec, 1000.0, 0.0, 1, true);
        }
        assert_eq!(c.state, PauseState::Active);
    }

    // Warm-up: no pause until both sample counts are met, even under clearly
    // bad economics. The speculative count is the last to cross here.
    #[test]
    fn warmup_spec_threshold_respected() {
        let mut c = PauseController::new(1.0);
        for _ in 0..PauseController::WARMUP_SPEC - 1 {
            c.record(RoundKind::Spec, 100.0, 0.0, 1, true);
        }
        for _ in 0..PauseController::WARMUP_PLAIN {
            c.record(RoundKind::ForcedPlain, 40.0, 0.0, 1, false);
        }
        assert_eq!(c.state, PauseState::Active, "must not pause below the spec warm-up");
        // The final speculative sample crosses the threshold and triggers pause.
        c.record(RoundKind::Spec, 100.0, 0.0, 1, true);
        assert!(matches!(c.state, PauseState::Paused { .. }));
    }

    // The plain warm-up count gates pausing symmetrically: enough spec samples
    // but too few plain samples stays Active until the last plain sample lands.
    #[test]
    fn warmup_plain_threshold_respected() {
        let mut c = PauseController::new(1.0);
        for _ in 0..PauseController::WARMUP_SPEC {
            c.record(RoundKind::Spec, 100.0, 0.0, 1, true);
        }
        for _ in 0..PauseController::WARMUP_PLAIN - 1 {
            c.record(RoundKind::ForcedPlain, 40.0, 0.0, 1, false);
        }
        assert_eq!(c.state, PauseState::Active, "must not pause below the plain warm-up");
        c.record(RoundKind::ForcedPlain, 40.0, 0.0, 1, false);
        assert!(matches!(c.state, PauseState::Paused { .. }));
    }

    // Regression for the mean-of-ratios misfire. Bimodal spec rounds — cheap
    // high-commit (200 ms / 11 tokens) alternating with expensive 1-commit
    // mismatch rounds (360 ms / 1 token) — aggregate to 560 ms / 12 = 46.7
    // ms/tok, under the plain comparator, so speculation wins and the controller
    // must never pause. The old single EMA of per-round ratios folded the
    // 360/1 = 360 ms/tok spikes and paused this exact shape (measured: 76 of 99
    // rounds on a 1.4x-winning code-gen run). The rate form (EMA(ms)/EMA(tok))
    // amortizes a 1-commit round the way throughput does and tracks the true
    // aggregate.
    //
    // The comparator is 60 ms rather than the aggregate-hugging 55 ms on
    // purpose: the quotient is phase-dependent and peaks near 55 right after a
    // 1-commit sample (that round transiently dominates both EMAs), so 60 leaves
    // the assertion unambiguous headroom while still proving speculation wins.
    #[test]
    fn rate_ema_stays_active_when_aggregate_wins() {
        let mut c = PauseController::new(1.0);
        c.record(RoundKind::ForcedPlain, 60.0, 0.0, 1, false);
        c.record(RoundKind::ForcedPlain, 60.0, 0.0, 1, false);
        for i in 0..200 {
            let (ms, committed) = if i % 2 == 0 { (200.0, 11) } else { (360.0, 1) };
            c.record(RoundKind::Spec, ms, 0.0, committed, true);
            assert_eq!(c.state, PauseState::Active, "paused at round {i} on a winning workload");
        }
    }

    // Mirror: when the aggregate genuinely loses (every round 360 ms / 3
    // committed = 120 ms/tok vs 55 ms plain), the controller must pause once
    // warm — the rate form does not mask a real regression.
    #[test]
    fn rate_ema_pauses_when_aggregate_loses() {
        let mut c = PauseController::new(1.0);
        c.record(RoundKind::ForcedPlain, 55.0, 0.0, 1, false);
        c.record(RoundKind::ForcedPlain, 55.0, 0.0, 1, false);
        for _ in 0..PauseController::WARMUP_SPEC {
            c.record(RoundKind::Spec, 360.0, 0.0, 3, true);
        }
        assert!(matches!(c.state, PauseState::Paused { .. }));
    }

    // An empty-draft round ran the drafter (draft_ms) then a plain step; only
    // the plain-step portion (round_ms - draft_ms) feeds the plain comparator,
    // so the wasted drafter time never inflates it.
    #[test]
    fn empty_draft_round_excludes_drafter_cost_from_plain() {
        let mut c = PauseController::new(1.0);
        c.record(RoundKind::Spec, 90.0, 40.0, 1, false);
        assert_eq!(c.ema_plain_ms, Some(50.0));
    }

    // Poisoned-pause recovery: a transient bad patch (e.g. GPU contention from a
    // second process) leaves the spec EMAs grossly inflated and the controller
    // paused. When the transient passes, the FIRST good probe pair must resume
    // immediately on its own aggregate — not crawl back through a 256-token
    // backoff and an α=0.25 EMA that would take dozens of probes to recover — and
    // the spec cost must be reseeded to the pair's rate, dropping the poison.
    #[test]
    fn poisoned_pause_recovers_on_first_good_probe_pair() {
        let mut c = PauseController::new(1.0);
        c.ema_spec_round_ms = Some(4000.0); // 4000 ms/tok — grossly poisoned
        c.ema_spec_committed = Some(1.0);
        c.ema_plain_ms = Some(55.0);
        c.spec_samples = PauseController::WARMUP_SPEC;
        c.plain_samples = PauseController::WARMUP_PLAIN;
        c.state = PauseState::Paused { backoff: PauseController::BACKOFF_START };

        // Plain rounds run until the backoff's worth of tokens elapse.
        while c.next_kind() == RoundKind::PausedPlain {
            c.record(RoundKind::PausedPlain, 55.0, 0.0, 1, false);
        }
        assert_eq!(c.next_kind(), RoundKind::Probe);

        // A good probe pair (2 x 220 ms / 8 committed = 27.5 ms/tok). The first
        // round only stashes; the second resumes despite the smoothed EMA still
        // reading thousands, and reseeds the quotient to the pair's rate.
        c.record(RoundKind::Probe, 220.0, 0.0, 8, true);
        assert_eq!(c.next_kind(), RoundKind::Probe, "still mid-pair");
        assert!(matches!(c.state, PauseState::Paused { .. }), "first probe must not resume");
        c.record(RoundKind::Probe, 220.0, 0.0, 8, true);
        assert_eq!(c.state, PauseState::Active);
        assert!(
            (c.spec_cost().unwrap() - 27.5).abs() < 1e-9,
            "spec cost {} must be reseeded to the pair rate 27.5",
            c.spec_cost().unwrap(),
        );
    }

    // Mirror: a bad probe pair (2 x 360 ms / 1 committed = 360 ms/tok vs 55 ms
    // plain) must NOT resume and must double the backoff — eager recovery does
    // not mean gullible recovery.
    #[test]
    fn bad_probe_pair_stays_paused_and_doubles_backoff() {
        let mut c = PauseController::new(1.0);
        c.ema_spec_round_ms = Some(4000.0);
        c.ema_spec_committed = Some(1.0);
        c.ema_plain_ms = Some(55.0);
        c.spec_samples = PauseController::WARMUP_SPEC;
        c.plain_samples = PauseController::WARMUP_PLAIN;
        c.state = PauseState::Paused { backoff: PauseController::BACKOFF_START };

        while c.next_kind() == RoundKind::PausedPlain {
            c.record(RoundKind::PausedPlain, 55.0, 0.0, 1, false);
        }
        c.record(RoundKind::Probe, 360.0, 0.0, 1, true);
        c.record(RoundKind::Probe, 360.0, 0.0, 1, true);
        assert_eq!(c.state, PauseState::Paused { backoff: PauseController::BACKOFF_START * 2 });
    }

    // Accelerated warm-up: on an all-drafting run the normal 32-round forced-
    // plain cadence would delay the second plain sample (and thus the first
    // possible pause) to ~round 64, so a short net-negative generation would
    // never pause. The tightened warm-up cadence (a forced plain every 4th
    // Active round until the plain warm-up is met) collects the plain samples in
    // time, so a uniformly-losing workload (360 ms/tok spec vs 55 ms plain)
    // pauses within ~15 rounds.
    #[test]
    fn accelerated_warmup_pauses_short_generation() {
        let mut c = PauseController::new(1.0);
        let mut paused_round = None;
        for round in 1..=15 {
            let kind = c.next_kind();
            let ran_spec = matches!(kind, RoundKind::Spec | RoundKind::Probe);
            let (ms, committed) = if ran_spec { (360.0, 1) } else { (55.0, 1) };
            c.record(kind, ms, 0.0, committed, ran_spec);
            if matches!(c.state, PauseState::Paused { .. }) {
                paused_round = Some(round);
                break;
            }
        }
        assert!(paused_round.is_some(), "must pause within 15 rounds via accelerated warm-up");
    }

    // An empty-draft probe pair (both rounds ran the drafter but kept no draft)
    // collects no real sample: it must not reset the backoff, resume, or refresh
    // the still-poisoned spec EMA — it just retries. Only real evidence decides.
    #[test]
    fn empty_draft_probe_pair_decides_nothing() {
        let mut c = PauseController::new(1.0);
        c.ema_spec_round_ms = Some(4000.0);
        c.ema_spec_committed = Some(1.0);
        c.ema_plain_ms = Some(55.0);
        c.spec_samples = PauseController::WARMUP_SPEC;
        c.plain_samples = PauseController::WARMUP_PLAIN;
        c.state = PauseState::Paused { backoff: PauseController::BACKOFF_START };
        while c.next_kind() == RoundKind::PausedPlain {
            c.record(RoundKind::PausedPlain, 55.0, 0.0, 1, false);
        }
        // Both probe rounds draft empty (ran_spec = false): the drafter ran
        // (30 ms) then a plain step.
        c.record(RoundKind::Probe, 85.0, 30.0, 1, false);
        c.record(RoundKind::Probe, 85.0, 30.0, 1, false);
        assert_eq!(
            c.state,
            PauseState::Paused { backoff: PauseController::BACKOFF_START },
            "backoff untouched by a draftless probe pair",
        );
        assert_eq!(c.spec_cost(), Some(4000.0), "spec EMA not refreshed by empty probes");
    }

    // A probe pair with one empty round and one real round decides on the real
    // sample alone — the empty round contributes nothing but does not block
    // recovery; the real evidence resumes and reseeds the spec cost to itself.
    #[test]
    fn probe_pair_decides_on_single_real_round() {
        let mut c = PauseController::new(1.0);
        c.ema_spec_round_ms = Some(4000.0);
        c.ema_spec_committed = Some(1.0);
        c.ema_plain_ms = Some(55.0);
        c.spec_samples = PauseController::WARMUP_SPEC;
        c.plain_samples = PauseController::WARMUP_PLAIN;
        c.state = PauseState::Paused { backoff: PauseController::BACKOFF_START };
        while c.next_kind() == RoundKind::PausedPlain {
            c.record(RoundKind::PausedPlain, 55.0, 0.0, 1, false);
        }
        // First probe drafts empty (no sample); the second is a good real round
        // (220 ms / 8 committed = 27.5 ms/tok <= 55 -> resume).
        c.record(RoundKind::Probe, 85.0, 30.0, 1, false);
        assert!(matches!(c.state, PauseState::Paused { .. }), "one round in: no decision yet");
        c.record(RoundKind::Probe, 220.0, 0.0, 8, true);
        assert_eq!(c.state, PauseState::Active);
        assert!(
            (c.spec_cost().unwrap() - 27.5).abs() < 1e-9,
            "spec cost {} must be reseeded to the single real sample",
            c.spec_cost().unwrap(),
        );
    }

    // draft_argmax_probs reduces per-row argmax + softmax-of-argmax on-device;
    // it must match an independent scalar reference row-for-row. Runs on the GPU
    // when available (else CPU) over distinct random logits (no exact ties, so
    // the first-maximal argmax convention is unambiguous).
    #[test]
    fn draft_argmax_probs_matches_scalar() {
        let dev = crate::gguf::metal_device().unwrap_or(Device::Cpu);
        let (n, vocab) = (5usize, 257usize);
        // Distinct pseudo-random logits, deterministic across runs.
        let data: Vec<f32> = (0..n * vocab).map(|i| (i as f32 * 1.2345).sin() * 7.0).collect();
        let logits = Tensor::from_vec(data.clone(), (n, vocab), &dev).unwrap();

        let (ids, probs) = draft_argmax_probs(&logits).unwrap();
        assert_eq!(ids.len(), n);
        assert_eq!(probs.len(), n);

        for r in 0..n {
            let row = &data[r * vocab..(r + 1) * vocab];
            let mut max = f32::NEG_INFINITY;
            let mut arg = 0usize;
            for (i, &l) in row.iter().enumerate() {
                if l > max {
                    max = l;
                    arg = i;
                }
            }
            let sum: f64 = row.iter().map(|&l| ((l - max) as f64).exp()).sum();
            let want_p = (1.0 / sum) as f32;
            assert_eq!(ids[r] as usize, arg, "row {r} argmax");
            assert!((probs[r] - want_p).abs() < 1e-5, "row {r} prob {} vs {want_p}", probs[r]);
        }
    }
}
