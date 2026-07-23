# Parity verification

How we prove the Rust/candle engine reproduces the vendored llama.cpp fork
(`reference/llama.cpp-laguna-branch`, poolside's branch with native Laguna
support) on Laguna S 2.1. Two independent tracks, because the two oracles expose
different things.

## Two tracks

**Track A — first-divergence bisection (`scripts/parity.ts` vs `llama-eval-callback`).**
The fork's `llama-eval-callback` runs the real graph and prints, for every node,
a full-tensor `sum` plus a handful of sampled values. It does *not* print full
tensors. That is exactly enough to walk the graph in execution order and find
the *first* layer where our intermediate math drifts from the fork — which is
what you want when something is wrong. It is not enough for a real full-vector
cosine on the logits.

**Track B — full-logit gate (`tests/parity.rs`, dump vs dump).**
Two `logits-dump` JSON files in the same schema are compared on the full
last-position logit vector: cosine, top-1, top-5. The reference dump is a
blessed run (initially of our own engine as a regression guard; in WP8, one
derived from the fork). This is the track that enforces `cos >= 0.999`.

Neither track needs the other; run Track A to *locate* a bug, Track B to *gate*
a release.

## The dump format (`logits-dump`)

`src/bin/logits-dump.rs` feeds raw token ids through one forward pass and writes
JSON. The format is final as of WP6; the tool compiles now but only produces
output once WP7 implements `LagunaModel::{load,forward,set_tap_capture,take_taps}`.

```jsonc
{
  "model": "models/laguna-s-2.1-Q4_K_M.gguf",
  "prompt": "def fib(n):",       // provenance only; may be null. The tool never tokenizes.
  "moe_impl": "reference",        // reference (oracle) or fused
  "tokens": [2, 1288, ...],       // input token ids (u32)
  "n_tokens": 12,
  "vocab": 100352,
  "logits": [ ...vocab f32... ],  // FULL last-position logits
  "top1": 1288,
  "top5": [[1288, 21.5], ...],    // (token_id, logit), descending
  "taps": [
    {
      "name": "attn_norm-0",      // fork cb() name + "-{layer}"; layer -1 nodes are bare
      "shape": [12, 3072],        // candle dims, outer..inner (last dim = feature)
      "sum": 12.34,               // whole-tensor sum — directly comparable to eval-callback `sum`
      "mean": 0.001, "std": 0.98, "l2": 34.2,
      "first8": [ ... ],          // first 8 of the last-position row
      "last_row": [ ...feature f32... ] | null  // full last-position row; null if > 16384 elems
    }
  ]
}
```

Why "last-position row": eval-callback truncates every tensor to first-3/last-3
along each dim, and its last printed row is the last token. So the last-position
feature vector is the one row we can compare in detail on both sides. We store
it in full; the fork gives its first-3/last-3.

## Tap names (`cb()` in `reference/.../src/models/laguna.cpp`)

The fork names graph nodes with `cb(tensor, name, il)`. The printed node name is
`"{name}-{il}"` for `il >= 0` and bare `"{name}"` for `il == -1`
(`src/llama-context.cpp:2437`). `logits-dump` uses the identical names so
`parity.ts` can match by name.

Per-layer taps, in graph order (`il` = 0..n_layer-1; S 2.1 has 48 layers):

| tap name             | what it is                                              | layers        |
|----------------------|---------------------------------------------------------|---------------|
| `attn_norm`          | pre-attention RMSNorm output                            | all           |
| `attn_gate_proj`     | `g_proj` on the pre-attention hidden state (pre-softplus)| all          |
| `Qcur_normed`        | Q after head-dim QK-RMSNorm, pre-RoPE                   | all           |
| `Kcur_normed`        | K after head-dim QK-RMSNorm, pre-RoPE                   | all           |
| `Qcur_rope`          | Q after RoPE (YaRN on full layers, plain on SWA)        | all           |
| `Kcur_rope`          | K after RoPE                                            | all           |
| `attn_out`           | attention output before the softplus gate              | all           |
| `attn_gate_softplus` | the gate after `softplus(g_proj)`                       | all           |
| `attn_gated`         | attention output after multiplying by the gate         | all           |
| `attn_o_proj`        | after the output projection `wo`                        | all           |
| `ffn_inp`            | residual sum feeding the FFN block                      | all           |
| `ffn_norm`           | pre-FFN RMSNorm output                                  | all           |
| `ffn_moe_out`        | routed mixture-of-experts output (scaled, summed)       | MoE only (il>=1)|
| `ffn_shexp`          | shared-expert SwiGLU output                             | MoE only (il>=1)|
| `ffn_out`            | MoE: `moe_out + shexp`; dense layer 0: dense SwiGLU     | all           |
| `l_out`              | layer output (post-FFN residual, after control vector)  | all           |

Global taps (`il == -1`, bare names), in order:

| tap name        | what it is                                                     |
|-----------------|----------------------------------------------------------------|
| `h_nextn`       | pre-final-norm residual stream (DFlash drafter capture point)  |
| `result_norm`   | final RMSNorm output (the embedding fed to the lm head)        |
| `result_output` | final logits = `output @ result_norm` (last position)          |

S 2.1 has one leading dense layer (`leading_dense_block_count = 1`), so layer 0
emits `ffn_out` from a dense SwiGLU and *no* `ffn_moe_out`/`ffn_shexp`; layers
1..47 are MoE. `llama-eval-callback` additionally prints the generic helper
nodes that `build_attn`/`build_moe_ffn`/`build_ffn`/`build_norm` emit internally
(e.g. `kq`, `kqv`, `ffn_moe_logits`, `ffn_moe_weights`); those are not in our
explicit tap set, and `parity.ts` simply ignores nodes with no matching tap.

## eval-callback output format

`common_debug_cb_eval` (`common/debug.cpp`) prints, per node:

```
ggml_debug:                 inp_embd = (f32)   GET_ROWS(token_embd.weight{2560, 51200, 1, 1}, inp_tokens{1, 1, 1, 1}}) = {2560, 1, 1, 1}
    [
     [
      [ -0.0181,   0.0272,   0.0272,   ...,   0.19,   0.02,  -0.11  ],
      ...
     ],
    ]
    sum = -3.214000
```

Findings that shaped the tooling:

- **Header:** `ggml_debug: <name> = (<dtype>) <OP>(<src0>{ne}, <src1>{ne}}) = {ne}`.
  `ne` is ggml order — `ne[0]` is the innermost/feature dim, `ne[1]` the token
  dim (the transpose of our candle `shape`).
- **`sum`** is over the *entire* tensor (computed before the truncated print), so
  it is a real full-tensor signal even though the values are truncated. This is
  the backbone of the divergence walk.
- **Values are truncated** to first-3/last-3 along each dimension (`n = 3`,
  hardcoded in `common_debug_print_tensor`). You cannot recover a full tensor —
  hence Track B for full-vector cosine. The final printed innermost row is the
  last token = last position.
- **Token ids are echoed.** `run()` in `examples/eval-callback/eval-callback.cpp`
  prints `number of input tokens = N` followed by N lines of one id each. There
  is no raw-token input to the fork, so these echoed ids are authoritative;
  `ref-dump.sh` extracts them so our dump runs on the identical sequence.
- Everything goes to stdout+stderr via the common logger, so capture with `2>&1`.

Optional full-vocab top-k oracle: `llama-server`'s `/completion` accepts
`"n_probs": 5` and returns per-position top-5 tokens with logprobs
(`tools/server/server-task.cpp`). Useful in WP8 if you want top-5 agreement
straight from the fork rather than dump-vs-dump.

## Runbook

Prerequisites: the fork is built at `reference/llama.cpp-laguna-branch/build/bin`
and the model is at `models/laguna-s-2.1-Q4_K_M.gguf`.

**The one-command path: `scripts/parity-gate.ts`.** For the full Track-B cycle —
produce the Reference dumps, produce the Fused candidate dumps, and run every
tiered gate — run:

```bash
bun scripts/parity-gate.ts        # all tiers; ~25-30 min of model time on a cold set
bun scripts/parity-gate.ts --tiers strict,mm            # just the full-logit gate
bun scripts/parity-gate.ts --tiers decode --fixtures long-swa   # one decode fixture
bun scripts/parity-gate.ts --regen-ref                  # rebuild the Reference dumps too
```

Flags: `--tiers strict,mm,decode,ppl` (default all), `--fixtures
code-short,text-mixed,long-swa` (default all; strict/mm always grade code-short,
decode grades all three, ppl has no fixture axis), `--regen-ref` (force-rebuild
the Reference oracle dumps — otherwise a dump carrying `provenance.moe_impl ==
"reference"` is reused), `--regen-ppl-ref` (regenerate the *committed*
`tests/fixtures/reference-ppl.json` — review and stage it yourself),
`--parity-dir DIR` (default `$LAGUNA_PARITY_DIR` or `/tmp/laguna-parity`).

The script enforces the model-run hazards for you: it `pgrep`s for a running
`laguna|llama` process before every model invocation, runs strictly serial (one
75GB process at a time), streams all model output to log files under the parity
dir, and never pipes through a pager. Candidate dumps are always regenerated (the
thing under test); Reference dumps are reused when valid. It prints a per-tier
PASS/FAIL summary with the key metric and exits nonzero on any failure. The
manual per-step commands below are what the script automates — they remain the
reference and the fallback.

**1. Produce the reference side** (per prompt). `ref-dump.sh` runs
eval-callback, extracts the authoritative token ids, cross-checks against
`llama-tokenize`, and optionally greedy-decodes for a top-1 oracle:

```bash
scripts/ref-dump.sh -m models/laguna-s-2.1-Q4_K_M.gguf --fixture code-short -o /tmp/ref-code --gen 24
# or with an ad-hoc prompt:
scripts/ref-dump.sh -m models/laguna-s-2.1-Q4_K_M.gguf -p "def fib(n):" -o /tmp/ref-code
```

Outputs in the out dir: `eval-callback.txt` (the trace), `tokens.txt`
(authoritative ids, comma-separated), `llama-cli.txt` (greedy continuation, with
`--gen`), `ref-cmd.txt` (the exact next commands).

**2. Produce our side** on the identical ids (needs WP7):

```bash
cargo run --release --bin logits-dump -- \
  --model models/laguna-s-2.1-Q4_K_M.gguf \
  --tokens "$(cat /tmp/ref-code/tokens.txt)" \
  --taps \
  --output /tmp/ref-code/ours.json
```

**3a. Track A — bisection:**

```bash
bun scripts/parity.ts \
  --ours /tmp/ref-code/ours.json \
  --ref  /tmp/ref-code/eval-callback.txt \
  --report /tmp/ref-code/parity-report.json
```

Prints a pass/fail summary: first divergent node (with `sumRelErr`/`rowRelL2`),
final-logits sum + sampled cosine. Divergence threshold defaults to rel error
`1e-2`; override with `--threshold`. Exit code 0 = pass.

**3b. Track B — full-logit gate:** put two dumps in a directory as
`candidate.json` and `reference.json`, then run the tier matching how the
candidate was produced (`LAGUNA_PARITY_TIER`, default `strict`):

```bash
# strict tier — candidate is the CLASSIC mv fallback path with legacy f32
# attention (dump with LAGUNA_NO_MM_ID=1 AND LAGUNA_MV_CLASSIC=1 AND
# LAGUNA_ATTN_F32=1):
LAGUNA_PARITY_DIR=/tmp/ref-code LAGUNA_PARITY_TIER=strict \
  cargo test --test parity -- --ignored --nocapture
# mm tier — candidate is the default mm_id prefill path:
LAGUNA_PARITY_DIR=/tmp/ref-code LAGUNA_PARITY_TIER=mm \
  cargo test --test parity -- --ignored --nocapture
```

The gate is **two-tier**, because the fused path uses two different expert
kernels and they have different (both correct) numerical envelopes vs the f32
`Reference` oracle:

- **classic mv fallback — prefill under `LAGUNA_NO_MM_ID=1` AND
  `LAGUNA_MV_CLASSIC=1` AND `LAGUNA_ATTN_F32=1` (the legacy f32 attention
  path; the shipped default computes attention in f16)** — strict gate: cosine
  `>= 0.999`, top-1 match, top-5
  overlap `>= 4/5`. This gates the legacy/kill-switch mat-vec path — candle's
  classic mv kernels, whose per-(token,slot) matvec accumulates each output as
  one f32 dot product in the same order as the oracle's per-row matmul, so it
  tracks the oracle tightly (0.999057 on code-short, essentially zero headroom).
  **The shipped DEFAULT decode path is the *vendored* ggml-geometry mv kernel**
  (`LAGUNA_NO_MM_ID=1` alone, without `LAGUNA_MV_CLASSIC`), which reorders that
  f32 accumulation and lands at ~0.997887 vs the oracle — below the 0.999 gate
  by ~0.0012, the same class of correct-but-reordered change as a decode-kernel
  edit. The vendored default is therefore gated by the decode greedy tier +
  the perplexity tier, with its full-logit cosine reported as a diagnostic only
  (`LAGUNA_PARITY_TIER=decode`); strict validates only the classic fallback.
- **mm_id path — prefill at seq >= 32 (the shipped default, tensor `_t` f16
  tiles; the `_hp`/`_t_hp` f32-tile variants gate identically)** —
  fork-equivalence gate: raw cosine `>= 0.995`, top-5 overlap `>= 4/5`, and top-1
  matches `Reference` **OR the candidate's top-1 is the `Reference`'s top-1 or
  top-2 AND the `Reference`'s own top-1/top-2 margin is < 0.5 logit** (a genuine
  near-tie — the candidate still has to pick one of the two contenders, not an
  arbitrary token). The near-tie test keys off the REFERENCE's margin, not the
  candidate's gap: cross-kernel drift here is ~1.5% rel-L2, which at this logit
  span (~24) is ~0.3 logit, so any sub-0.5 reference margin is inside ~2x the
  noise floor and the top-1 is not meaningfully determined. mm_id sums over K in
  8x8 simdgroup tiles — a different (equally valid) f32 accumulation ORDER than
  the per-row oracle — so it drifts a little further; this is not tile precision
  (the f16 tiles land within ~2e-4 of the f32 default at model scale, the residual
  is the tiling itself), and the fork's own tensor-path prefill fails the strict
  0.999 gate the same way.
- **decode-kernel changes (`LAGUNA_PARITY_TIER=decode`)** — full-logit cosine is
  a *reported diagnostic, not a gate*. Every remaining decode lever (fusing the
  attention chain, vendoring ggml's mv geometry) reorders f32 accumulation, and
  the strict tier passes at cosine 0.999057 with essentially zero headroom, so no
  such change can ever clear a 0.999 cosine even when it is correct. The gate for
  decode-kernel work is instead **greedy agreement vs the frozen Reference oracle
  under teacher-forced replay** (`greedy_parity`, below), plus a
  perplexity-delta bound (see "Perplexity gate"). Under the
  decode tier `logit_parity` still hard-fails on input-token / logit-length
  mismatch (a diagnostic on mismatched inputs is meaningless) but only *prints*
  cosine / top-1 / top-5.

  **Scale-sensitive hard checks (all tiers).** Cosine, top-1, and top-5 overlap
  are all scale-INVARIANT — a candidate that is a uniform rescale of the reference
  (`candidate = c · reference`) sails through them at cosine 1.0, and a NaN slips
  past (`NaN < cos_min` is false). The decode tier's greedy gate is likewise
  scale-invariant (it compares argmax). So `compare` adds two hard failures in
  EVERY tier: (1) any non-finite candidate logit (or non-finite computed metric)
  fails; (2) the candidate/reference L2-norm ratio must lie in
  `[1/NORM_RATIO_MAX, NORM_RATIO_MAX]`. `NORM_RATIO_MAX` is `1.18`: measured across
  every same-prompt dump pair in the parity dir (2026-07-22 — the code-short
  mv_id/mm_id/strict configs), the worst norm ratio vs `reference.json` was
  `1.0178` (a 1.78% drift; `ew_strict` 0.9907 on the low side), so ~10× headroom
  over the 1.78% drift gives 1.18 (floor 1.05). This is a coarse guard against a
  gross scale/NaN bug, not a precision gate — cosine/top-1 handle precision. The
  perplexity gate is the other scale-sensitive layer for decode.

  **Dump provenance + per-tier enforcement.** The tier is caller-selected
  (`LAGUNA_PARITY_TIER`) with no cross-check against how the dump was produced, so a
  decode/mv_id candidate graded under the looser mm tier would mask a regression the
  strict tier would catch. Every dump `logits-dump` writes now carries a `provenance`
  object (moe_impl, prefill `seq_len`, active `mm_variant`, `no_mm_id`, `mm_min_seq` =
  the seq threshold, and `attn_dtype` = the attention compute dtype, all from a single
  source in `src/ops`). Enforcement by tier:
  - `attn_dtype` is pinned per side in EVERY tier: the reference must always be
    `"f32"` (the oracle runs under `LAGUNA_ATTN_F32=1`); the strict candidate must be
    `"f32"` (strict gates the legacy f32 attention path); mm/decode/ppl candidates
    must be `"f16"` (the shipped default). A dump missing the field came from a stale
    `logits-dump` binary and hard-fails (regenerate).
  - `attn_mm` (the attention prefill gemm path) is pinned per side in EVERY tier:
    reference and strict candidates must be `"f32-bypass"`
    (`LAGUNA_ATTN_F32` routes attention through the legacy dequant-f32 QMatMul, so
    the f16 library's mm branch never runs; the strict/reference envs additionally
    pin `LAGUNA_ATTN_MM_CLASSIC=1` as belt-and-suspenders), while the mm/decode/ppl
    candidate paths run the SHIPPED default — the Metal-4 cooperative-tensor
    prefill gemm (`f16_t.metal`) — and must be `"tensor"`
    (`LAGUNA_ATTN_MM_CLASSIC` reverts to the classic simdgroup gemm →
    `"classic"`). History: tensor was REJECTED as default on 2026-07-23 morning
    (its f16 activation staging flipped a 0.6-margin reference decision at
    code-short step 29 — a flip the fork does not make), then UNLOCKED the same
    day by the flash-attention kernel's f32 sdpa boundaries (the drift-attribution
    experiment proved f16 sdpa was AMPLIFYING the staging noise; with flash, the
    flash+tensor matrix passes all six gates) and flipped to default
    owner-approved. A dump missing the field is a stale binary and hard-fails
    (regenerate).
  - `attn_glue` (the attention-glue path: the fused softplus gate, the fused
    permute/cast copies, and the fused partial-rotary rope — all vendored kernels
    behind ONE kill-switch, `LAGUNA_ATTN_GLUE_CLASSIC`) is pinned per side in EVERY
    tier: reference dumps and strict candidates must be `"classic"` (the gate
    script pins the env var for both — unlike `combine`, the Reference oracle
    EXECUTES the attention glue, so its anchor is the env pin, not a separate code
    path), while mm/decode/ppl candidates grade the shipped fused kernels and must
    be `"fused"`. Each fused kernel is BIT-IDENTICAL to the candle chain it
    replaces (proven by `f32::to_bits` tests in `src/ops/attn_glue.rs` against the
    live chains), so the pin anchors provenance rather than a numerical tier. A
    dump missing the field is a stale binary and hard-fails (regenerate).
  - `sdpa` (the sdpa compute dtype: `"f16"` = the shipped kernel, `"f32"` = the
    `LAGUNA_SDPA_F32` experiment hook, which skips q's f16 cast, widens the
    cached f16 k/v exactly, and dispatches candle's float32 Metal sdpa kernels)
    is pinned per side: references and strict candidates must be `"f16"` (every
    blessed anchor ran the f16 kernel); mm/decode/ppl candidates expect `"f16"`
    by default. Introduced at provenance schema version 2 with grandfather
    `"f16"` (see the versioning note below), so pre-versioning dumps stay valid.
  - `flash` (the prefill attention path: `"fused"` = the vendored flash
    kernel in `src/ops/flash.metal` — f32 Q/output boundaries, f16 K/V
    cache reads, in-kernel causal+SWA-window masking with block skip, no
    mask tensor; `"classic"` = candle sdpa + materialized f16 mask, the
    `LAGUNA_FLASH_CLASSIC` kill-switch) is pinned per side: references and
    strict candidates must be `"classic"` (the gate script pins the env for
    both — every blessed strict/reference anchor ran the classic path);
    mm/decode/ppl candidates grade the shipped flash kernel and must be
    `"fused"`, overridable via `LAGUNA_PARITY_EXPECT_FLASH` (the
    `--flash-classic` gate flag sets env + override together for A/B runs).
    Introduced at provenance schema version 3 with grandfather `"classic"`.
    The flash path is value-identical to the `--sdpa-f32` experiment
    configuration by construction (bitwise unit tests in `src/ops/flash.rs`
    against the composed f32-sdpa reference), which the model-level gate
    confirmed: flash-default mm cos and Δnll match the sdpa-f32-only matrix
    column digit-for-digit.
  - **Experiment expectation overrides.** The mm/decode/ppl CANDIDATE
    expectations for `sdpa`, `attn_mm`, and `flash` are overridable per gate
    run via `LAGUNA_PARITY_EXPECT_SDPA` (e.g. `f32`),
    `LAGUNA_PARITY_EXPECT_ATTN_MM` (e.g. `classic`), and
    `LAGUNA_PARITY_EXPECT_FLASH` (e.g. `classic`), read by
    `tests/parity.rs`. This is how a non-default path (f32 sdpa, classic
    projections gemm, classic sdpa fallback) gets graded against the real
    oracle without editing the test; `parity-gate.ts --sdpa-f32` /
    `--attn-mm-classic` / `--flash-classic` set the candidate env and the
    matching override together and print a prominent EXPERIMENT banner.
    Reference-side and strict-tier pins are never overridable — they anchor
    the oracle.
  - Under `LAGUNA_PARITY_TIER=mm` the CANDIDATE must prove the mm_id path was
    actually active — `moe_impl == "fused"`, `seq_len >= mm_min_seq`,
    `no_mm_id == false` — else the gate hard-fails asking you to regenerate.
  - Under `LAGUNA_PARITY_TIER=strict` the CANDIDATE must be a Fused-runner dump
    produced under `LAGUNA_NO_MM_ID=1` — `moe_impl == "fused"` and
    `no_mm_id == true` — else a copied reference dump (moe_impl "reference",
    attn_dtype "f32": exactly what strict's attn pin expects) would clear the strict
    thresholds vacuously. (`LAGUNA_MV_CLASSIC` is not recorded in provenance and
    cannot yet be asserted from the dump — see TODO.md.)
  **The reference side requires provenance in every tier**: `reference.json`'s
  `provenance.moe_impl` must be `"reference"`, or `compare` hard-fails (a
  `reference.json` accidentally produced with `--moe-impl fused` would otherwise make
  every tier a fused-vs-fused self-comparison that hides a regression). This is
  fail-closed with no legacy exception, so **a long-lived dump that predates the
  provenance (or `attn_dtype`) field must be regenerated once with the current
  `logits-dump`** before the gate will run.

  **Provenance schema versioning.** Missing-field-equals-stale-binary is the only
  sound default, but before versioning it meant every provenance field ADDITION
  retroactively invalidated every cached/committed reference dump (~40 min of GPU
  regeneration each time — paid three times for `combine`/`attn_mm`/`attn_glue`).
  Each dump's provenance now carries `schema_version` (written by `logits-dump`;
  `src/parity_schema.rs` is the single source of truth for the current version and
  the field-introduction table, used by both `tests/parity.rs` and mirrored in
  `parity-gate.ts`'s reference-reuse check). A dump with no `schema_version` is
  version 1: the baseline field set (`moe_impl`, `seq_len`, `mm_variant`,
  `mm_min_seq`, `no_mm_id`, `attn_dtype`, `combine`, `attn_mm`, `attn_glue`), all
  REQUIRED with no grandfathering — the references regenerated at that era carry
  every one, so a v1 dump missing any of them is genuinely stale, not merely old.
  A field introduced at version N (e.g. `sdpa`: N=2, grandfather `"f16"`;
  `flash`: N=3, grandfather `"classic"`) resolves to its grandfather value when
  missing from a dump whose version predates N — adding a field therefore no
  longer invalidates existing references — while missing at/after N remains the
  stale-binary hard fail. A dump claiming a version NEWER than the gate binary
  knows is rejected (rebuild the test binary).

  **Why forced replay, not free-run.** Comparing two free-running greedy decodes
  cascades at the first near-tie: the moment the two engines pick different
  tokens their histories diverge and every subsequent position is incomparable
  (WP8's long-swa free-run agreed for 9 post-prompt tokens, then split on a
  0.079-logit near-tie and was uncomparable thereafter). Teacher-forcing keeps
  every step comparable: the Reference runner free-runs greedy N tokens
  (`--greedy N`, the reference dump), then the candidate (Fused runner) is forced
  along that exact token sequence (`--replay`), recording its OWN argmax at each
  step before being forced. A step passes when the candidate's argmax equals the
  reference token; a mismatch is excused only when the *Reference's own*
  top-1/top-2 margin at that step is `< 0.5` logit (same NEAR_TIE_MARGIN rule as
  the mm tier — which token wins a sub-0.5 oracle tie is noise) AND the two picks
  are mutual contenders: the candidate's pick is in the reference's top-2, or the
  reference token is in the candidate's top-2 with the candidate's own top-1/top-2
  margin also `< 0.5` (fork calibration showed the reference's stored top-2 can be
  the wrong contender set at 3-way ties, but only when the candidate is also flat
  there — a confident wrong pick is never fork-class). Total excused steps are
  capped at `max(2, n/8)` (fork calibration measured `<= 4/64` tie-flips per
  fixture; more than 1-in-8 is drift, not tie noise).

Calibration (code-short fixture, 2026-07-22, full-logit vs the f32 `Reference`):

| fixture | reference top1/top2 margin | verdict |
|---|---|---|
| code-short | 0.319 logit (350 over 268) | near-tie: sub-0.5, candidate top-1 of 268 or 350 passes |
| mixed-text | ~2.3 logit (350 decisive) | must match: candidate top-1 = 350 |
| long-text | ~2.3 logit (350 decisive) | must match: candidate top-1 = 350 |

Candidate cosines on code-short (all configs `>= 0.995`), each labelled by its
exact expert config AND attention era so the near-identical numbers are
distinguishable. The shipped default's mm cosine is identical across the
f32→f16 attention switch because the vendored f16-weight prefill gemm is
BIT-IDENTICAL to candle's f32 gemm (simdgroup MMA, same K order, identical
operand values; the decode gemv differs in ulps only):

| candidate config | raw cos vs Reference | top-1 |
|---|---|---|
| classic mv fallback (`LAGUNA_NO_MM_ID=1 LAGUNA_MV_CLASSIC=1 LAGUNA_ATTN_F32=1`) — the **strict** gate | 0.999057 | 350 = ref (top5 5/5; still holds post-`run_stack` refactor) |
| vendored mv (`LAGUNA_NO_MM_ID=1`, shipped decode default) — diagnostic only | 0.997887 | 350 = ref (top5 4/5) |
| mm_id tensor `_t` prefill + vendored f16 attention (**shipped default**) | 0.996987 | 268 |
| mm_id `_hp` f32 tiles, glue-off (f32-attention era) | 0.99687 | 268 |
| mm_id `_hp` f32 tiles, with the (removed) L2 rescale glue (f32-attention era) | 0.99694 | 350 |
| mm_id f16 tiles (`LAGUNA_MM_ID_F16=1`), glue-off (f32-attention era) | 0.99672 | 268 |
| fork tensor-path prefill (llama-server, same oracle) | ~0.9962 raw / 0.99091 centered | 350 |

On code-short the model genuinely can't separate 350 from 268 (0.319-logit
reference margin), so the mm-tier top-1 there is unconstrained (268 and 350 both
pass); mixed-text and long-text are decisive 350 and the default matches.

The mm_id variant env toggles (`LAGUNA_MM_ID_F16` → f16 classic tiles;
`LAGUNA_MM_ID_CLASSIC` → f32 classic `_hp`; `LAGUNA_MM_ID_TENSOR_HP` → f32 tensor
`_t_hp`; `LAGUNA_NO_MM_ID` → force mv_id everywhere) are all **presence-based**:
they are enabled by the variable merely being SET, whatever its value —
`LAGUNA_MM_ID_F16=0` still selects f16 tiles. To disable one, UNSET it (do not set
it to `0`). The candidate dump's `provenance.mm_variant` / `provenance.no_mm_id`
record which path actually ran, and the mm tier hard-fails a candidate whose
provenance shows the mm_id path was not active (see 3b provenance note below).

**3c. Decode gate — greedy agreement (forced replay).** For decode-kernel
changes (see §3b's decode tier). Two dumps per fixture, produced STRICTLY SERIAL
(one 75GB model process at a time — `pgrep -fl "laguna|llama"` first). Default
N = 64 decode steps per fixture; run all three fixtures. Use the `tokens` array
from `tests/fixtures/parity-prompts.json` for `--tokens`.

Note `--greedy N` records N steps but only invokes the decode kernel N−1 times:
step 0's logits come from the prefill forward, and each subsequent step runs one
decode forward. So **N must be ≥ 2 to exercise the decode kernel at all** (N = 1
grades only prefill). The gate now also enforces, per side, runner provenance
(the reference dump must be `moe_impl == "reference"`, the candidate replay must
be `moe_impl == "fused"` — a forgotten `--moe-impl fused` on the candidate would
otherwise self-compare the oracle and pass vacuously) and, per step, finiteness
(no non-finite logits) plus a candidate/reference L2-norm ratio bound (the same
`NORM_RATIO_MAX` = 1.18 as the full-logit gate — greedy argmax is scale-invariant,
so a uniform rescale would agree on every token yet be wrong). Both require the
`l2`/`nonfinite` per-step fields the current `logits-dump` writes; an older dump
missing them is a hard fail (regenerate).

```bash
DIR=/tmp/decode-code-short
mkdir -p "$DIR"
TOKENS="2,1172,36668,..."   # the fixture's `tokens` array

# reference side: Reference oracle, free-run greedy 64 tokens
cargo run --release --bin logits-dump -- \
  --model models/laguna-s-2.1-Q4_K_M.gguf \
  --moe-impl reference --tokens "$TOKENS" --greedy 64 \
  --output "$DIR/reference-greedy.json"

# candidate side: Fused runner, teacher-forced along the reference dump
cargo run --release --bin logits-dump -- \
  --model models/laguna-s-2.1-Q4_K_M.gguf \
  --moe-impl fused --replay "$DIR/reference-greedy.json" \
  --output "$DIR/candidate-greedy.json"

# run the gate (reads reference-greedy.json + candidate-greedy.json from $DIR)
LAGUNA_PARITY_DIR="$DIR" LAGUNA_PARITY_TIER=decode \
  cargo test --test parity greedy_parity -- --ignored --nocapture
```

`--replay` takes the prompt from the reference dump, so `--tokens` is not needed
on the candidate side. `greedy_parity` prints a summary line (total steps,
agreements, excused near-ties, non-excused mismatches) and fails on any
non-excused mismatch, listing the step index, both tokens, and the reference
margin. (`LAGUNA_PARITY_TIER=decode` is not read by `greedy_parity` itself; set
it when running `logit_parity` for the diagnostic-only cosine report on the same
dumps.)

## Pass criteria

- Track A: what disqualifies is a divergence *cliff* — one node whose deviation
  jumps orders of magnitude above its layer neighborhood. Smooth monotonic drift
  across layers (observed in practice: sampled-row rel-L2 growing ~0.001 at layer
  0 to ~0.2 at layer 47) is expected candle-Metal vs ggml-Metal kernel noise on
  identical Q4_K_M weights, not a bug. parity.ts's `--threshold` (default `1e-2`)
  flags candidates; judge them against the drift profile.
- Track B: two-tier by expert kernel (see §3b for the rationale and calibration).
  classic mv fallback (`LAGUNA_NO_MM_ID=1 LAGUNA_MV_CLASSIC=1`): cosine `>= 0.999`,
  top-1 agreement, top-5 `>= 4/5` — this is the **strict** tier, and it gates the
  legacy kill-switch path, NOT the shipped default (the shipped vendored mv path
  reorders f32 accumulation to ~0.997887 and is gated by the decode + ppl tiers,
  cosine diagnostic-only). mm_id (default tensor prefill): cosine `>= 0.995`, top-5 `>= 4/5`,
  top-1 matches Reference OR (candidate top-1 is the Reference's top-1/top-2 AND
  the Reference's top-1/top-2 margin is `< 0.5` logit — near-tie). Both tiers
  require identical input ids. The `logit_parity` test selects the tier via
  `LAGUNA_PARITY_TIER=strict|mm` (default `strict`).
- Decode-kernel changes: the gate is **greedy agreement vs the Reference oracle
  under forced replay** (`greedy_parity`, §3c), across all three fixtures. Each
  step passes when the candidate's argmax equals the reference token; a mismatch
  is excused only at a reference near-tie (the Reference's own top-1/top-2 margin
  at that step `< 0.5` logit — the same NEAR_TIE_MARGIN rule as the mm tier) with
  mutual-contender overlap (both-sides-flat for the reference-in-candidate-top-2
  direction; see §3b), and total excuses are capped at `max(2, n/8)`. Any
  non-excused mismatch fails. The full-logit cosine is reported (via
  `LAGUNA_PARITY_TIER=decode logit_parity`) but NOT gated — the strict 0.999
  cosine passes with zero headroom and can't accept accumulation-reordering
  decode changes. A perplexity-delta bound complements this (see "Perplexity
  gate" below).
- Greedy vs the FORK (Track A style, distinct from the Reference-oracle decode
  gate above): sequences must agree except at near-ties. A divergence is
  acceptable when the logit gap between the two candidate tokens at the
  divergence point is `< 0.15` — the empirical Q4_K_M cross-kernel noise floor is
  ~0.1 logit, so demanding tighter (the original 1e-3) fails correct engines.
  (WP8 baseline, 2026-07-21: code-short agreed 107 tokens then split on a 0.015
  gap; text-mixed 16 tokens / 0.0053; long-swa 9 post-609-prompt tokens / 0.079.)

## Raw greedy oracle

`llama-cli -st -no-cnv` APPLIES THE CHAT TEMPLATE — its `llama-cli.txt` is a
chat-wrapped continuation, not a raw-completion oracle, and will "diverge" at
token 1 against a raw run. For raw greedy parity use `llama-server /completion`
with a token-id ARRAY as the prompt (no template, no tokenizer):

```bash
curl -s localhost:8080/completion -d '{
  "prompt": [2, 750, 15243, ...], "n_predict": 200, "temperature": 0,
  "top_k": 1
}'
```

**Trap** (fork-calibration, 2026-07-22): on this fork build `temperature: 0`
(with or without `samplers: []`) still DIST-samples the emitted token — ~25% of
steps disagree with the server's own top logprob. A greedy oracle must either
force `top_k: 1` (as above) or ignore the emitted tokens entirely and read the
argmax from `n_probs` raw logprobs (`post_sampling_probs: false`).

## Perplexity gate

Greedy agreement (§3c) catches token flips but is blind to how the fused path
reshapes the *distribution* — a change that keeps every argmax but systematically
sharpens or flattens the tails would pass. The perplexity-delta bound over a
frozen held-out corpus is that scale-sensitive complement: it compares the mean
next-token NLL of the Fused runner against the Reference oracle on the identical
corpus and fails if they drift more than the frozen bound.

**Corpus** (`tests/fixtures/ppl-corpus.txt`, attribution in
`tests/fixtures/ppl-corpus-README.md`): the head of the WikiText-2 raw *test*
split (`Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `test`), the
standard LM-perplexity corpus — held out (not the parity prompts), mixed-register
prose, truncated at a paragraph boundary. 4386 tokens after the crate tokenizer
(`add_special_tokens=false`, one leading BOS prepended by hand), 4385 scored.
Sized so the one-time Reference pass stays under the ~30 min budget (see the
calibration record).

**Protocol** (both runners run it identically, so protocol quirks cancel in the
delta): a single continuous chunked-prefill pass over the whole corpus —
512-token chunks (`PREFILL_CHUNK`-sized), positions continuous, the KV cache
fresh at the start (just-loaded model) and never reset between chunks. At every
position `p` the all-position lm-head path (`LagunaModel::forward_all_logits`)
yields logits that predict `tokens[p+1]`; we gather
`log_softmax(logits[p])[tokens[p+1]]` in f64 (stable logsumexp). Every position
`0..T-1` is scored — each target is a real corpus token; the BOS (`tokens[0]`) is
never itself a target, and the final position has no successor.
`mean_NLL = -mean(logprob)` over all scored positions. `forward_all_logits` runs
the identical transformer stack as the generate path but keeps the lm head over
every position instead of narrowing to the last, so the default generate/decode
path (which still narrows) and every other parity gate are unaffected. The fused
side runs the mm_id prefill kernel (chunks of 512 >= `MM_ID_MIN_SEQ`), so the
delta bounds that kernel's distribution fidelity, not just its argmax.

**Gate** (`ppl_parity` in `tests/parity.rs`, `#[ignore]`): loads
`reference-ppl.json` + `candidate-ppl.json` from `LAGUNA_PARITY_DIR`, enforces
runner provenance (reference `moe_impl=="reference"`, candidate `=="fused"` —
same fail-closed rule as the greedy gate; a forgotten `--moe-impl fused` would
otherwise self-compare the oracle), zero non-finite logprobs on both sides,
identical scored token streams (count + FNV `token_hash` + full ids), then
asserts `|mean_NLL(fused) − mean_NLL(reference)| <= PPL_NLL_DELTA_MAX`. The bound
is calibrate-then-freeze: `max(3 × |measured delta|, 0.002)` nats, the 0.002
floor keeping a near-zero measured delta from setting an impossibly tight bound.
The rejection paths (wrong kind, wrong/missing provenance, non-finite, token
mismatch, over-bound delta) have non-ignored unit tests alongside the greedy
ones.

Commands (STRICTLY SERIAL — one 75GB model process at a time, `pgrep -fl
"laguna|llama"` first; the corpus + BOS exceeds the default 4096 ctx, so pass
`--max-ctx`):

```bash
DIR=/tmp/ppl
mkdir -p "$DIR"
# reference side — keepable: the frozen oracle NLL lives at
# tests/fixtures/reference-ppl.json; only regenerate on a corpus change:
cargo run --release --bin logits-dump -- \
  --model models/laguna-s-2.1-Q4_K_M.gguf \
  --moe-impl reference --max-ctx 5120 \
  --ppl tests/fixtures/ppl-corpus.txt --output "$DIR/reference-ppl.json"
# candidate side (fused):
cargo run --release --bin logits-dump -- \
  --model models/laguna-s-2.1-Q4_K_M.gguf \
  --moe-impl fused --max-ctx 5120 \
  --ppl tests/fixtures/ppl-corpus.txt --output "$DIR/candidate-ppl.json"
# gate — the frozen fixture can stand in for the reference side:
cp tests/fixtures/reference-ppl.json "$DIR/reference-ppl.json"
LAGUNA_PARITY_DIR="$DIR" cargo test --test parity ppl_parity -- --ignored --nocapture
```

The reference dump is a keepable artifact (frozen oracle → frozen NLL): the
blessed copy is `tests/fixtures/reference-ppl.json` (21 KB — mean NLL, per-chunk
means, and the full token stream for alignment re-verification), so a routine
fused check only regenerates the fused side and points the gate at a dir holding
the fixture (as `reference-ppl.json`) beside the fresh `candidate-ppl.json`.
Resizing or regenerating the corpus invalidates both the fixture and the frozen
bound — recalibrate.

**Calibration record (2026-07-22).** Corpus 4386 tokens (4385 scored). The
Reference all-position-prefill rate was measured first on a 642-token slice:
~3.1 tok/s (~205 s wall including the ~30 s weight upload), which projected the
full pass to well under the 30 min budget; the full Reference pass then took
~15 min, the Fused pass ~46 s. Measured mean NLLs on the full corpus (both
`nonfinite == 0`):

| runner | mean NLL (nats) |
|---|---|
| Reference (oracle) | 2.020392 |
| Fused (mm_id prefill) | 2.018455 |

Delta `|fused − reference|` = **0.001937** nats. `max(3 × 0.001937, 0.002)` =
0.005811, frozen (rounded up to a clean value that keeps the >=3× margin) at
**`PPL_NLL_DELTA_MAX = 0.006`**.

## Limitations

- eval-callback exposes only per-node sums + first-3/last-3 samples, so Track A's
  logit comparison is coarse (sum + a few sampled logits, plus a sampled cosine
  over those points). Full-vector cosine is Track B's job.
- eval-callback's samples are the last *innermost* row. For rank-2 tensors
  ({feature, token}) that is the whole last-position vector and lines up with our
  `last_row`. For rank>2 tensors (`Qcur_normed`/`Kcur_normed`, shaped {head_dim,
  head, token}) it is only one head of the last position, so `parity.ts` skips
  the sampled-row check there and relies on the full-tensor `sum`, which is exact
  at every rank.
- eval-callback computes logits for the last position only (`llama_batch_get_one`),
  so per-position logit parity is not available; both tracks compare the last
  position. This matches `logits-dump`, which also returns last-position logits.
- CPU vs Metal floating-point accumulation in the fork can differ slightly; run
  eval-callback with `-ngl 999` (the `ref-dump.sh` default) so the fork uses the
  Metal path, closest to our engine.
