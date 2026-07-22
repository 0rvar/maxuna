# Engineering log

What was tried, what worked, what didn't, and why. Append new entries AT THE
TOP (reverse-chronological). TODO.md is the forward ledger; this is the history.
Dates marked `~` are reconstructed from git/TODO records, not contemporaneous.

## 2026-07-22 — f16 divergence hunt: accumulator hypothesis DEAD; operand convention is the gap

**Context.** The hybrid restructure (entry below) re-gated: ppl best-ever
(Δnll 0.000524), decode text-mixed/long-swa PASS under the widened contender
rule — but code-short got WORSE: mm cosine 0.9929 (< 0.995 bar; all-f16 scored
0.9961) and greedy step 2 flipped a 2.6-margin decision. The fork-calibration
data showed the asymmetry is ours: at that same step the fork WIDENS the margin
2.60 → 4.38 while our all-f16 nearly flipped it (0.118 left) and hybrid flips
it. Perf: hybrid is the fastest path yet, 17.2 tok/s sustained (LPM).

**Experiments.**
- Isolation split: hybrid + mv_id prefill (`LAGUNA_NO_MM_ID=1`) scores 0.9957
  vs the oracle → the 0.9929 is roughly additive: hybrid attention ~3.4e-3
  cosine deficit + mm_id prefill's usual ~2.8e-3.
- Non-monotonicity warning: hybrid's rounding sources are a strict SUBSET of
  all-f16's, yet it scores lower on single-position cosine (0.9929 vs 0.9961)
  while beating it on 4385-position ppl. Single-fixture last-position cosine is
  a chaos-dominated instrument (router near-tie roulette) — do not rank
  variants by it alone.
- HYPOTHESIS: candle's f16 matmul accumulates in f16 (ggml accumulates f32) →
  systematic per-matmul gap. **DEAD.** Kernel-source audit at the pinned rev:
  candle dispatches MLX gemv (m==1) / steel gemm_nt (prefill), and BOTH
  accumulate f32 (gemv.metal:28-31,112 `AccT=float`; mlx_gemm.metal:1410
  explicit `float` AccumType, simdgroup_float8x8). ggml's
  kernel_mul_mv_f16_f32 / kernel_mul_mm_f16_f32 likewise f32. Our measured
  1.8e-4 per-block rel error is consistent with f32 accumulation (f16 accum
  would be ~2.7e-2 for K=3072 dots).
- The REAL structural difference: candle's f16 path is f16×f16→f16 — the
  activation is pre-cast to f16 and the matmul result is rounded to f16 at the
  store — while ggml runs f16 weights × f32 activations → f32 out. The fork's
  ONLY f16 rounding is the stored weights; we add two ~2.4e-4 rounding
  boundaries per projection that the fork never sees.

**Verdict — RESOLVED, shipped.** Vendored ggml's mixed-dtype kernels
(`src/ops/f16.metal`: `kernel_mul_mv_f16_f32_v` decode gemv from the fork's
`_4` vectorized template, `kernel_mul_mm_f16_f32_v` prefill gemm with FLOAT
staging tiles — strictly tighter than the fork, which stages activations half;
host `ops::matmul_f16`, ggml's ne11 ≥ 8 mv/mm split). Proj is cast-free, f32
in/out. Per-block rel error 8.2e-6 (was 1.8e-4 cast-path). Gate: ALL SIX PASS
at the f32-era anchors — mm 0.996987, ppl Δnll 0.001937, decode 0 non-excused
— because the numerics genuinely collapse onto the legacy path: A/B verified
the prefill gemm is BIT-IDENTICAL to candle's f32 steel gemm (both are
simdgroup_float8x8 MMA walking K monotonically over identical operand values;
dequantized-f16 f32 weights ≡ f16 weights exactly), and decode gemv differs
only at the 4th decimal (geometry ulps), same tokens. Decode 18.2 tok/s
sustained LPM (+55% over the 11.7 f32 baseline; hybrid was 17.2, all-f16
16.2); prefill neutral. Lesson trio: (1) verify a plausible kernel-level
hypothesis in the kernel SOURCE before building on it — the accumulator
hypothesis died in one read-only investigation; (2) when a per-op-better
change scores worse on a chaotic metric, trust the aggregate (ppl) and hunt
the structural difference instead of reverting; (3) when gate numbers
reproduce another era's to 6+ decimals, distinguish "stale binary" from
"genuine numerical equivalence" EMPIRICALLY (strings + provenance + an A/B
with a step that must differ) — both happened today, once each.

## 2026-07-22 — f16 attention: all-f16 chain, then restructured to hybrid (superseded — see the divergence-hunt entry above for the shipped vendored-kernel resolution)

**Context.** Phase-0 (below) showed decode attention cost is ~80% f32 weight
streaming: candle dequantizes the GGUF's F16 attention weights to dense f32,
11.2 GB/token. Plan: keep the weights f16 and run the attention block in f16.

**Change.** All-f16 chain implemented first (f16 weights + f16 activations, f16
QK-norm weights and rope tables; casts deleted), with `LAGUNA_ATTN_F32` as the
kill-switch back to the legacy f32 path. Bench-validated before wiring: 15.8 vs
25.8 ms boost-clock isolation for the chain.

**Result.** Decode 11.7 → 16.2 tok/s sustained (+38%); prefill +6% @931-tok
chunk, neutral @4k. Gates: strict PASS (kill-switch path bit-identical), mm
PASS 0.9961 (from 0.9970), ppl PASS Δnll 0.00175 (f32-era 0.00194). BUT decode
greedy FAILED 3 steps of 192 (2 code-short, 1 text-mixed).

Fork calibration (teacher-forced llama-server replay, 192 steps, raw `n_probs`
logprobs): the fork itself flips 4/192 vs the f32 oracle — always to the
reference's top-2, max reference margin 0.348, never outside top-2. Two
findings: (a) our text-mixed "failure" was a GATE bug — the fork ranks our pick
#2, 0.13 behind; the reference's stored top-2 is not the true contender set at
a 3-way tie, so the contender-set rule was widened; (b) our code-short
pair-drift tail (2.24 logits) exceeds the fork's envelope (0.67) — all-f16
INTERMEDIATES are numerically heavier than what the fork does.

**Verdict.** Restructured to the fork's structure: f16 weights + f32
activations, casts inside Proj only. Keeps the weight-streaming win (the whole
point); expected to collapse the drift to fork-class. RESOLUTION: the cast
hybrid still failed mm/code-short (see the entry above); the shipped endpoint
is the vendored mixed-dtype kernels, which pass everything.

## 2026-07-22 — Oracle policy: reference dumps pin LAGUNA_ATTN_F32

**Context.** With attention dtype now runtime-switchable, "which path is the
oracle" became ambiguous.

**Change.** Reference dumps pin `LAGUNA_ATTN_F32=1` — the oracle is the
maximally precise path. `scripts/parity-gate.ts` `referenceEnv()` enforces it.
The committed ppl fixture (`tests/fixtures/reference-ppl.json`) was regenerated
under the pin plus a provenance field.

**Result.** Regenerated fixture mean_nll bit-identical: 2.020392.

**Verdict.** Standing policy: precision-reducing changes must never leak into
the reference side of any gate.

## 2026-07-22 — Measurement traps: stale binaries, a "greedy" oracle that samples, boost clocks

**Context.** Two near-misses and one systematic bench error, all caught during
the f16 attention gating.

**Findings.**
- Stale-binary vacuous gate pass: an agent's HEAD-clone comparison build
  clobbered `target/release` with pre-change binaries; cargo saw them as fresh,
  and the full gate "passed" — against old code. Caught only because the
  results were identical to six decimals to the previous run. Fix: `attn_dtype`
  in dump provenance, hard-fail when the field is missing, and an
  `isReferenceDump` reuse check in parity-gate.ts.
- Fork llama-server `temperature: 0` dist-samples on this build — ~25% of
  emitted tokens differ from its own top logprob. Greedy oracling against the
  fork must use `top_k: 1` or take the argmax of raw `n_probs` logprobs. Now
  warned in docs/parity.md.
- Means-vs-mins in GPU benches: boost-clock decay (see Phase-0) poisons
  cross-variant ablation deltas — real improvements measured as negative
  deltas. Compare plateau means only.

**Lesson.** A gate that passes to six decimals is itself a finding. Provenance
fields exist so "what actually ran" is checkable, not assumed.

## 2026-07-22 — Phase-0 decode-budget re-measurement: death-by-dispatch refuted

**Context.** The old decode budget (TODO "Decode kernel work") attributed
~48.6 ms/token to per-layer dispatch overhead in attention and priced a fusion
prize accordingly. Re-measured before spending on it.

**Experiment.** Ignored bench harness: `decode_bench` modules in attention.rs
and moe.rs (9 benches — `attn_decode_chain_bench`, `attn_decode_ablation_bench`,
`attn_proj_f16_bench`, `attn_decode_f16_chain_bench`, `dispatch_overhead_bench`,
`moe_decode_ffn_bench`, `sampler_decode_bench`, `token_tail_bench`,
`full_stack_decode_bench`; synthetic weights, no GGUF), plus a real-model sweep.

**Result.**
- Sustained vs boost clocks: identical GPU work runs ~1.7x slower after ~1 s of
  load. full_stack time series 41 → 76 ms plateau, matching real decode
  78.7 ms/token within 3%. Isolation mins are boost-clock fiction.
  CONTEXT (learned 2026-07-22, after the entry was written): the machine runs in
  macOS **Low Power Mode** during these sessions (deliberate — high-performance
  mode brings coil whine, fans, hot keyboard), so the "decay" is likely the
  low-power governor clamping after a ~1 s burst (implied bandwidth ~540 GB/s
  burst → ~315 GB/s plateau). ALL absolute ms/tok-s numbers in this log are
  low-power-mode numbers unless marked otherwise; ratios and budget shares
  should transfer across modes (bandwidth-bound throughout), but comparisons
  against the fork's historical llama-bench figures (power mode unknown) need a
  one-time same-mode calibration pair before being treated as like-for-like.
  CPU-side numbers (command-buffer encode = the 2.4 µs/dispatch, sampler top-k,
  model load 15.3 s) were also measured under low-power scheduling (E-core bias,
  capped clocks) — i.e. dispatch overhead was refuted at its WORST case; the
  warm-load figure is the number most likely to improve in high-perf mode
  (re-check before sizing the mmap/no-copy lever).
- Sum-checked sustained budget of the 78.7 ms token: attention ~49 (projections
  ~40 = f32 weight streaming at bandwidth; sdpa ~4; glue ~6), MoE ~24 (mv_id
  gather ~14, routing ~6, shared ~3), tail+sampler ~3.
- Dispatch overhead: 2.4 µs/dispatch, and decode tok/s is FLAT across
  `CANDLE_METAL_COMPUTE_PER_BUFFER` 10 → 1000.

**Verdict.** Death-by-dispatch REFUTED as the main story; the 48.6 ms fusion
prize did not exist; old lm_head "6.5 ms" was ~1.3 sustained. The old budget's
section totals were roughly right at sustained clocks; its intra-section
attribution was wrong. Attack replanned around f16 attention (entry above).
Lesson: use plateau means or trust ratios only; never compare a variant's min
against another's mean.

## 2026-07-22 — Vendored ggml mv geometry: perf-flat, kept for insulation

**Context.** The (pre-Phase-0) budget claimed candle's mv "runs ~15x under
bandwidth" with lm_head at 6.5 ms; porting ggml's current mv geometry looked
like a decode win.

**Change.** Ported ggml's current `kernel_mul_mv_{id_,}q{4,6}_K_f32_impl`
geometry (N_R0=2, N_SG=2, register-row f32 accumulate) into `src/ops/mv.metal`
(separate library, no Metal-4 dep), host dispatch in `src/ops/dispatch.rs`;
lm_head bypass at seq==1 over a retained shared buffer
(`gguf::qlinear_with_buffer`, same zero-copy trick as ExpertStack). Default for
q4_K/q6_K; `LAGUNA_MV_CLASSIC` reverts to candle's baked kernels.

**Result.** End-to-end decode FLAT: 13.1 (vendored) vs 13.0 (classic) tok/s
@512ctx, 256-tok warm bench. Microbench (`plain_mv_lmhead_bench`): the
[100352x3072] q6_K matvec at seq==1 is 0.685 ms vendored vs 0.738 ms QMatMul —
both near the ~0.62 ms/250MB bandwidth floor. Correctness solid: greedy gate
passes all three fixtures (62/2 excused, 64/0, 59/5 excused, 0 non-excused);
decode-tier diagnostic cosine 0.99789.

**Verdict.** LESSON: the mv compute was never the bottleneck. Both hot mv paths
were already ~bandwidth-optimal in candle; the old 6.5/18.4 ms line items were
per-dispatch latency inside the full pipeline, which geometry can't recover.
DECIDED (Orvar, 2026-07-22): vendored stays the default anyway — not slower,
more fork-faithful, and it insulates decode from upstream candle kernel changes.
`LAGUNA_MV_CLASSIC` remains the escape hatch.

## 2026-07-22 — Parity gate goes three-tier; decode graded by greedy replay + perplexity, not cosine

**Context.** The strict full-logit gate passes at cosine 0.999057 on code-short
— essentially zero headroom. The vendored mv kernels (correct, but reordered
f32 accumulation) land at 0.997887. Every remaining decode lever reorders
accumulation the same way, so a 0.999 cosine can never accept a correct
decode-kernel change.

**Change** (docs/parity.md §3b, `LAGUNA_PARITY_TIER`):
- strict — classic mv fallback only (`LAGUNA_NO_MM_ID=1` +
  `LAGUNA_MV_CLASSIC=1`): cos ≥ 0.999, top-1, top-5 ≥ 4/5.
- mm — mm_id prefill default: fork-equivalence (cos ≥ 0.995, top-5 ≥ 4/5, top-1
  match or reference near-tie < 0.5 logit).
- decode — shipped decode path and all future decode-kernel work: greedy
  agreement vs the Reference oracle under teacher-forced replay (mismatch
  excused only at reference near-ties < 0.5 logit) plus a perplexity-delta
  bound; cosine printed as a diagnostic only.

Teacher-forcing because free-run greedy comparisons cascade at the first
near-tie (WP8: long-swa agreed 9 post-prompt tokens, then split on a
0.079-logit tie and was incomparable after). Scale-sensitive hard checks in
every tier (finiteness, L2-norm ratio bound 1.18) backstop the scale-invariant
metrics.

**Ppl gate calibration** (docs/parity.md "Perplexity gate"): wikitext-2 raw
test head, 4386 tokens (4385 scored); Reference pass ~15 min, Fused ~46 s.
Mean NLL 2.020392 (Reference) vs 2.018455 (Fused), delta 0.001937 nats →
`PPL_NLL_DELTA_MAX` frozen at 0.006 (max(3×delta, 0.002), rounded up keeping
the ≥3x margin).

**Verdict.** Decode work is graded on behavior (argmax agreement) and
distribution (NLL delta). Full-logit cosine remains the gate only for the paths
whose accumulation order matches the oracle's.

## 2026-07-22 — Rescale glue removed from the default path (+8% decode)

**Context.** The ~6-op L2 rescale glue existed to guard the f16 activation cast
in the mm_id f16-tile kernel. Audit: the default down paths never cast the
activation to f16 (mv_id reads f32 and accumulates f32; mm_id-hp stages src1 as
float).

**Change.** Glue skipped by default; kept only under `LAGUNA_MM_ID_F16`.

**Result.** Decode 12.5 → 13.5 tok/s (+8%), prefill ~149 → 157. No inf/nan on
code/mixed/long-swa prefill or greedy decode; strict tier 0.99906, mm tier
0.99687 (the code fixture's 350/268 top-1 is a genuine reference near-tie,
margin 0.319).

**Verdict.** Kept. The cheapest "fusion" was deleting work that guarded a case
the default path never hits — no kernel needed.

## 2026-07-22 — Fused-activation kernels RETIRED: the MoE router is a chaos amplifier

**Context.** The prefill gap is surrounding-dispatch overhead, so fusing the
silu/mul/rescale glue into one kernel looked like an easy win.

**Change.** Two vendored kernels built: (a) fused silu/mul/L2/rescale, whose
f32 L2 reduction order differed from candle's by ~1e-6; (b) a plain elementwise
silu*mul differing by ~1e-7 (division vs candle's multiply-by-reciprocal).

**Result.** Both per-op correct (end-to-end 1.6e-7/layer vs candle), yet final
logits diverged 1.3e-3–1.5e-3 under the strict gate: a ~1e-6 activation nudge
flips near-tie expert selections in later layers and the error compounds.

**Verdict.** Both removed. CONSTRAINT for all future kernel work: do not
reimplement any op upstream of the router (activation, norm, router logits)
unless it is BIT-IDENTICAL to candle; post-router ops (down output, lm_head)
are safe to fuse — no cascade.

## ~2026-07-21→22 — Prefill mm_id arc: 60 → 151 → 188 tok/s

**Context.** Prefill sat at ~60 tok/s on mv_id. Candle's baked
`kernel_mul_mm_id_*` is unusable at 256 experts (its `top_k × tokens`
threadgroup-memory row map caps out), so the fix had to be vendored.

**Change/Result**, in sequence:
- Classic simdgroup port of ggml's two-pass row-map kernels
  (`src/ops/mm_id.metal`, runtime-compiled via `src/ops/pipelines.rs`; map0
  builds per-expert compacted token-slot lists in device scratch appended to
  the dst buffer, so no smem cap): 60 → ~151 tok/s.
- Tile precision: f32 tiles (`_hp`) over f16 — per-matmul oracle rel error
  2.6e-4 → 1.94e-7 (~1330x) at ~0 throughput cost. The residual model-level
  drift (~0.9969 cosine) is tiled f32 K-accumulation ORDER, not precision — the
  fork's own tensor-path prefill scores ~0.9962 and fails the strict 0.999 gate
  identically. This is what forced the fork-equivalence mm tier into being.
- Cooperative-tensor `matmul2d` port (Metal-4, `kernel_mul_mm_id_t`; probe
  `tensor_matmul2d_probe` guards toolchain support): 151 → 188 tok/s @925-tok
  chunk (+24%), 142 → 157 @4230 (+10%); agrees with the classic-f16 path to
  4.4e-7 per matmul. Now the default.
- `_t_hp` (float-operand `matmul2d`): compiles and is f32-precise (1.94e-7) but
  SLOWER at model scale (163 tok/s — double-width tiles cost more than the
  rescale they save), so opt-in only. Its speculative instantiations were split
  into `src/ops/mm_id_t_hp.metal`, lazily compiled, so a future toolchain that
  rejects float `matmul2d` operands breaks only the opt-in path
  (`instantiation_matrix_matches_metal` enforces the partition).

**Verdict.** 3.1x prefill; fork is still ~361 pp512 — the remaining gap is the
~50 surrounding candle dispatches per MoE+attention layer, NOT the matmul
(tensor-hp has no rescale glue yet is slower, killing the glue-as-bottleneck
theory). Gotcha for posterity: the variant env toggles (`LAGUNA_MM_ID_F16`
etc.) are presence-based — `=0` still enables them.

## 2026-07-21 — WP8: what a workable parity gate looks like

**Context.** First end-to-end parity campaign against the fork (initial
implementation commit e7ff50b). Shaped docs/parity.md.

**Findings.**
- Track B (full-logit dump-vs-dump) is the real gate. Track A
  (llama-eval-callback bisection) only exposes per-node sums plus
  first-3/last-3 samples — good for LOCATING a divergence, not for gating.
  Judge Track A by divergence cliff, not absolute thresholds: smooth drift to
  ~0.2 sampled rel-L2 by layer 47 is normal candle-Metal vs ggml-Metal noise on
  identical Q4_K_M weights.
- Free-run greedy comparisons cascade at the first near-tie: code-short agreed
  107 tokens then split on a 0.015-logit gap; text-mixed 16 tokens / 0.0053;
  long-swa 9 tokens / 0.079. Divergences acceptable only at gaps < 0.15 logit
  (empirical Q4 noise floor ~0.1); the original 1e-3 demand fails correct
  engines.
- `llama-cli -st -no-cnv` applies the chat template, so it is useless as a raw
  greedy oracle. Use llama-server `/completion` with a token-id array prompt.
- Known benign divergence sources: candle's Metal arg_sort is unstable on exact
  routing ties (ggml's is stable); our softplus differs from ggml only at
  overflow magnitudes.

## 2026-07-21 — Zero-copy expert_stack: one MTLBuffer, ~70GB saved

**Context.** The stacked expert tensors are most of the 75GB file; naive
QTensor construction would double-copy them on device.

**Change.** `gguf.rs::expert_stack` uploads once via `QStorage::from_data` and
clones the Buffer handle (objc retain) BEFORE the storage moves into
`QTensor::new`, so `ExpertStack.buffer` and the QTensor share one MTLBuffer.
Candle exposes no accessor for a QTensor's Metal buffer, so the construction
order IS the invariant.

**Verdict.** Break the order and you reintroduce a ~70GB VRAM double-copy. This
(plus the baked mm_id/mv_id kernels and the `new_library_with_source` surface)
is why the candle rev is pinned at `27f20fea…` — do not bump casually.

## 2026-07-21 — The official GGUF is a 256k conversion, not 1M

**Context.** The HF checkpoint config claims 1M context via YaRN factor 128.

**Finding.** The official GGUF says otherwise: `laguna.context_length =
262144`, YaRN factor 32. The GGUF metadata (`laguna inspect`) outranks
`config.json` as ground truth. Related: the net YaRN cos/sin magnitude is
COMPUTED (`(1 + 0.1·ln(factor)) × rope.scaling.attn_factor`, config.rs) — the
GGUF's `yarn_attn_factor` key is a saver artifact the fork never reads for
laguna.

**Verdict.** Going past 256k needs a rope-scaling override at load (TODO
"1M-context tuning"); v1 caps max_ctx at 32768. Lesson generalized into
CLAUDE.md's authority order: fork source > GGUF metadata > HF config files.

## ~2026-07-21 — Operational hazards, each learned the hard way

One-time incidents now codified in CLAUDE.md "Operational hazards"; logged here
so the ledger shows they were earned, not invented:

- Two concurrent 75GB model loads → GPU OOM
  (kIOGPUCommandBufferCallbackErrorOutOfMemory). `pgrep -fl "laguna|llama"`
  before every model run.
- An EOF-spinning llama-cli piped through glance fed it 88GB of RAM. Model
  output goes to a file, never a pager.
- Scripted llama-cli needs `-st -no-cnv </dev/null` or it spins in the
  interactive loop.
- The first forward folds in the one-time Metal weight upload — never report
  first-forward prefill as steady-state (`LAGUNA_BENCH` adds a warm-up).
- Homebrew here is nix-managed; cmake comes from nix and skips Apple SDK
  detection, so `scripts/build-llamacpp.sh` passes the sysroot explicitly.
