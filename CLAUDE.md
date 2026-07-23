# CLAUDE.md — agent context for the laguna engine

Read `README.md` first for what this is. This file is the context that is NOT
obvious from the code: ground-truth sources, hard-won gotchas, and workflows.
When picking up an item from `TODO.md`, read this whole file — most items have a
trap documented here.

## Non-negotiables

- **Design target**: max tokens/sec on THIS machine (M5 Max, 128GB, Metal),
  batch=1. No portability hedging — do not add CPU fallbacks or feature-gate
  Metal (the only CPU paths that exist are for unit tests and the Reference
  oracle).
- **TODO.md is the deferred-work ledger.** Append deferrals as they come up;
  never silently drop scope.
- **ReferenceExperts (moe.rs) is the frozen correctness oracle.** Never
  "optimize" it; every fused/perf change is validated against it (Track B).
- Any change touching model math must re-run the parity gate before it's done:
  `docs/parity.md` §3b (reference-vs-fused full-logit) at minimum.

## Ground truth, in order of authority

1. `reference/llama.cpp-laguna-branch/src/models/laguna.cpp` — the forward pass
   (attention block order ~lines 200-265, per-layer rope selection ~184-196).
2. `reference/llama.cpp-laguna-branch/src/llama-graph.cpp` `build_moe_ffn`
   (~line 1799) — exact MoE routing math.
3. The GGUF metadata itself (`laguna inspect`) — NOT the HF `config.json`
   (see gotchas: the official GGUF is a 256k conversion; HF config says 1M).
4. `reference/` HF files (tokenizer.json, chat_template.jinja,
   generation_config.json).

## Architecture cheat sheet (Laguna S 2.1, Q4_K_M GGUF)

- 48 layers, hidden 3072, vocab 100352, RMSNorm eps 1e-6, no biases, untied
  embeddings. Layer 0 dense SwiGLU (ff 12288); layers 1–47 MoE.
- Attention: layer il is FULL-attention iff `il % 4 == 0` (12 full / 36 SWA,
  window 512). Per-layer Q heads: 48 (full) / 72 (SWA); always 8 KV heads,
  head_dim 128. Order: gate logits = g_proj(**pre-attention normed input**);
  QK-norm (RMSNorm over head_dim) BEFORE rope; sdpa scale 1/√128 in f16;
  `attn_out *= softplus(gate)` per-head BEFORE o_proj.
- RoPE: full layers YaRN θ=500k partial-rotary 64/128 dims; SWA layers plain
  θ=10k all 128 dims. Net YaRN cos/sin magnitude = `(1 + 0.1·ln(factor)) ×
  gguf rope.scaling.attn_factor` — COMPUTED, not a stored constant (config.rs
  does this; the GGUF's `yarn_attn_factor` key is a saver artifact the fork
  never reads for laguna).
- MoE routing (exact order): sigmoid(router(h)) → `+ exp_probs_b` (bias affects
  SELECTION ONLY) → top-10 → gather UNBIASED probs → `/ max(sum, 6.1e-5)` →
  `× 2.5` → Σ wᵢ·SwiGLUᵢ(h) → `+ shared_expert(h)` unscaled. All on-GPU in
  moe.rs::route (no CPU readback in the layer loop).
- Tokenizer: byte-level BPE from `reference/tokenizer.json` (`from_gguf`
  intentionally errors). BOS 2 = `〈|EOS|〉` (U+3008/U+3009, NOT U+2329!),
  pad 9, EOG {2, 24=`</assistant>`, non-special}. `encode` uses
  add_special_tokens=false or the post-processor DOUBLES the BOS the chat
  template already wrote. Chat template is hand-rolled in chat.rs, validated
  byte-exact vs the fork's renderer (llama.cpp strips BOS from /apply-template
  and re-adds it at tokenize time — token streams are identical).

## GGUF facts that differ from the HF checkpoint config

- `laguna.context_length = 262144` (256k), YaRN factor **32** — the HF config's
  1M/factor-128 does NOT apply to the official GGUF. Going past 256k needs a
  rope override at load (TODO item).
- Attention weights are **F16** in the Q4_K_M file; only FFN/expert tensors are
  K-quantized (mixed Q4K/Q6K by imatrix). `exp_probs_b` is `.bias`-suffixed.
- Integer metadata arrives in mixed widths/signedness (head_count is i32[]).
- GGUF carries `general.sampling.temp=0.7/top_p=0.9`; we follow
  generation_config.json (temp 1.0, top_k 20) instead.
- Official quants: Q4_K_M 75GB (fits), Q8_0 128GB (does NOT fit), F16 235GB
  (+ published imatrix — self-quantizing Q5/Q6 is a TODO item).

## The candle situation

- Pinned by git rev `27f20fea…` in Cargo.toml (same rev as ../offload). The pin
  freezes: the baked `kernel_mul_mv_id_*` Metal kernels (now only the
  `LAGUNA_MV_CLASSIC` decode fallback — present in
  candle's metallib with ZERO upstream Rust wiring; src/ops/ is our host
  dispatch; candle's own FusedMoeGGUF/indexed_moe_forward PANIC on non-CUDA),
  plus the public QStorage/MetalDevice/`new_library_with_source` surface both the
  host dispatch and our vendored prefill kernels rely on. (candle's baked
  `kernel_mul_mm_id_*` is NO LONGER used — prefill is our vendored two-pass
  kernel; see below.) Do not bump the rev casually; re-verify those kernels and
  APIs if you do.
- Zero-copy invariant (gguf.rs `expert_stack`): the stacked expert tensor is
  uploaded ONCE via `QStorage::from_data`; the Buffer handle is cloned (objc
  retain) BEFORE the storage moves into `QTensor::new`, so ExpertStack.buffer
  and the QTensor share one MTLBuffer. Candle exposes no accessor for a
  QTensor's Metal buffer — preserve this construction or you reintroduce a
  ~70GB VRAM double-copy.
- `candle_metal_kernels::metal::Buffer` is the nameable buffer type; MTLSize is
  built via candle's `get_block_dims` factory. objc2-metal was deliberately NOT
  added (candle pins 0.3.2; a mismatched dep is a trap).
- Prefill mm_id is now our VENDORED two-pass kernel (`src/ops/mm_id.metal`,
  runtime-compiled via `src/ops/pipelines.rs` with `new_library_with_source`),
  NOT candle's baked `kernel_mul_mm_id_*` (unusable at 256 experts — see Perf
  state). map0 builds per-expert compacted token-slot lists + counts; mm reads
  them so each expert's threadgroups cover only its rows. No `top_k × tokens`
  threadgroup cap (the row map lives in device scratch appended to the dst
  buffer, not smem). Runtime-selectable variants via `ops::MmVariant` (cached
  env): tensor f16 tiles (`_t`, DEFAULT — Metal-4 `matmul2d`, ~2e-4 rel, needs
  the L2 rescale guard); tensor f32 tiles (`_t_hp`, `LAGUNA_MM_ID_TENSOR_HP`,
  ~2e-7 but slower); classic simdgroup f32 (`_hp`, `LAGUNA_MM_ID_CLASSIC`);
  classic simdgroup f16 (`LAGUNA_MM_ID_F16`). `LAGUNA_NO_MM_ID` forces mv_id
  everywhere. mm_id.metal includes `<metal_tensor>` + MetalPerformancePrimitives
  for the default `_t` path, so the default library requires Metal-4 tensor
  support to compile (fine per the M5-only mandate; the `tensor_matmul2d_probe`
  test guards it) — but only with HALF cooperative-tensor operands. The
  speculative FLOAT-operand `_t_hp` instantiations are split into a separate
  source (`src/ops/mm_id_t_hp.metal`), concatenated onto mm_id.metal and compiled
  lazily by pipelines.rs only when `LAGUNA_MM_ID_TENSOR_HP` is selected, so a
  future toolchain that rejects float `matmul2d` operands fails only that opt-in
  path, not the default prefill library (the
  `instantiation_matrix_matches_metal` test enforces the partition). Decode mv
  (routed gather + seq==1 lm_head) is our VENDORED
  ggml-current geometry (`src/ops/mv.metal`, separate library, no Metal-4 dep;
  f32 accumulate) — DEFAULT for q4_K/q6_K; `LAGUNA_MV_CLASSIC` reverts to
  candle's baked `kernel_mul_mv_id_*`/QMatMul. Perf-identical to candle
  (bandwidth-bound); kept to insulate decode from upstream candle changes.
- Attention projections are our VENDORED mixed-dtype f16 kernels
  (`src/ops/f16.metal`, own runtime-compiled library, no Metal-4 dep):
  f16 weights × f32 activations → f32 out, f32 accum (ggml's
  `mul_mv_f16_f32`/`mul_mm_f16_f32` convention; ggml's ne11 ≥ 8 mv/mm split).
  candle CANNOT express this (its matmul requires same-dtype operands and
  would round activations + outputs to f16 — measured 22x worse per block).
  `LAGUNA_ATTN_F32` reverts to dequant-f32 QMatMul (the legacy path the strict
  tier gates; reference dumps pin it via parity-gate's referenceEnv()).

## Verification workflow

- Runbook: `docs/parity.md`. Fixtures with real token ids:
  `tests/fixtures/parity-prompts.json` (code-short 58 / text-mixed 82 /
  long-swa 609 — the last exercises SWA-ring wraparound).
- One-command full cycle (dumps + all tiered gates, hazard-safe serial):
  `bun scripts/parity-gate.ts` (`--tiers`/`--fixtures`/`--regen-ref`; see
  docs/parity.md §3).
- Pass criteria philosophy (learned in WP8): the Track B full-logit gate is the
  real gate, and it is now THREE-TIER by change kind (`LAGUNA_PARITY_TIER`, see
  docs/parity.md §3b): **strict** (the CLASSIC mv fallback path only —
  `LAGUNA_NO_MM_ID=1` + `LAGUNA_MV_CLASSIC=1`) holds cos ≥ 0.999 + top-1 +
  top-5 ≥ 4/5; **mm** (tiled mm_id prefill default) holds a
  fork-equivalence gate (cos ≥ 0.995, top-5 ≥ 4/5, top-1 matches or a reference
  near-tie < 0.5 logit) because its f32 tile accumulation order drifts from the
  per-row oracle just as the fork's does; **decode** (the SHIPPED default
  decode path — vendored mv kernels — and all future decode-kernel changes)
  can't use full-logit cosine at all — accumulation reorders drop it to ~0.9979
  while strict's 0.999 has zero headroom — so cosine is
  DIAGNOSTIC-ONLY and the gate is greedy agreement vs the Reference oracle under
  teacher-forced replay (`greedy_parity`: candidate argmax == reference token,
  mismatches excused only at reference near-ties < 0.5 logit), plus a live
  perplexity-delta bound over a frozen wikitext-2 corpus (`ppl_parity`:
  |mean_NLL(fused) − mean_NLL(reference)| ≤ PPL_NLL_DELTA_MAX; docs/parity.md
  "Perplexity gate"). Track A vs
  llama-eval-callback: judge by divergence CLIFF, not absolute thresholds —
  smooth drift to ~0.2 sampled rel-L2 by layer 47 is normal cross-kernel Q4
  noise. Greedy: divergences acceptable only at near-ties, gap < 0.15 logit
  (empirical noise floor ~0.1).
- **Raw greedy oracle**: `llama-cli -st -no-cnv` APPLIES THE CHAT TEMPLATE —
  useless as a raw oracle. Use llama-server `/completion` with a token-id
  ARRAY prompt (docs/parity.md).
- Unit tests: `cargo test --release` — the `ops` tests REQUIRE Metal (they
  validate the kernel dispatch; never feature-gate them off).
- Known benign divergence sources: candle Metal arg_sort is unstable on exact
  routing ties (llama.cpp is stable); our softplus differs from ggml only at
  overflow magnitudes.

## Operational hazards (each of these has already bitten once)

- **ONE 75GB model process at a time.** Two concurrent loads = GPU OOM
  (kIOGPUCommandBufferCallbackErrorOutOfMemory). `pgrep -fl "laguna|llama"`
  before every model run.
- **Never pipe model/llama output through glance or any pager** — an
  EOF-spinning llama-cli once fed glance 88GB of RAM. Stream to a file
  (`> /tmp/x.txt 2>&1`), read the file.
- Scripted llama-cli runs need `-st -no-cnv </dev/null` or they spin forever in
  the interactive loop.
- Every model invocation reloads 75GB (~30s warm cache). Batch runs.
- Homebrew on this machine is nix-managed (`.homebrew-is-managed-by-nix`) —
  never `brew install` or chown /opt/homebrew. cmake comes from
  `nix shell nixpkgs#cmake`; nix's cmake skips Apple SDK detection, so
  `scripts/build-llamacpp.sh` passes the sysroot/framework path explicitly.
- The first forward folds in the one-time Metal weight upload — never report
  first-forward prefill numbers as steady-state.

## Perf state (2026-07-22)

POWER MODE CAVEAT: this machine runs in macOS Low Power Mode during dev
sessions (owner's choice — full power = coil whine + fans). Same-mode 2×2
calibration (2026-07-22) measured the LPM clamp at ~2.1x on decode
(bandwidth-bound) and ~2.3-2.8x on prefill (compute-bound) — bigger than the
~1.7x phase-0 estimate. VERIFY the active mode before labeling any number:
`pmset -g | awk '/lowpowermode/{print $2}'` (1 = LPM) — the System Settings
Energy Mode toggle is per-power-source (Battery vs Power Adapter tabs) and has
already silently failed to apply once. Ratios/budget shares transfer across
modes; absolute cross-mode comparisons do not (docs/log.md power-calibration
entry). Bench long enough to hit the plateau — first-second numbers are burst
fiction (even llama-bench pp512 reps are short enough to ride the LPM burst
window: ±30 t/s rep noise). Full-power numbers swing ~±10% with chip temp
(cool-first runs read high); LPM run-to-run is ~±5%.

Warm steady-state, fused path, vs fork `llama-bench`, pmset-verified 2×2
(2026-07-22; the fork's older 361/328/18.1 figures were LPM — confirmed):

| | ours LPM | fork LPM | ours full | fork full |
|---|---|---|---|---|
| decode (256-tok sustained / tg128) | ~18.2 tok/s | 18.5 | ~38.6 | 39.2 |
| prefill short (630-925 tok / pp512) | ~228 tok/s | 354 | (stale: ~415) | 990 |
| prefill 4k (4007 tok / pp4096) | ~237 tok/s | 348 | (stale: ~345) | 793 |

Decode is at parity with the fork in BOTH modes (0.98x). Prefill "ours" LPM
figures are 2026-07-23 post combine-fusion + mask-hoisting (174 → 228 @925,
~155 → 237 @4k in one day); the full-power ours column predates both. The
remaining prefill gap (0.64-0.68x fork) is led by the attention projections
gemm (classic simdgroup f16 path — tensor-path port is the open TODO), the
transpose/cast/rope glue, and serial scheduling vs ggml's concurrent encoder.

Prefill: the vendored two-pass mm_id kernel (tensor `matmul2d` default) took
prefill ~60 → ~188 tok/s (3.1x; ~174 re-measured 2026-07-22 in LPM). Decode:
11.7 → 18.2 tok/s (2026-07-22) via **f16 attention weights + vendored
mixed-dtype matmul kernels** (`src/ops/f16.metal`: f16 weights × f32
activations → f32 out, f32 accum — ggml's convention; the only f16 rounding is
the stored weights, so numerics match the legacy f32 path: prefill gemm is
bit-identical by MMA determinism, decode gemv differs in ulps). `LAGUNA_ATTN_F32`
(presence-based, load-time) reverts to the legacy dequant-f32 QMatMul path —
the strict parity tier gates that path; reference-oracle dumps pin it. The
per-token budget and how it was measured (sustained-vs-boost clocks, LPM
caveat above) live in docs/log.md's phase-0 entry; death-by-dispatch was
REFUTED — attention was ~80% weight-streaming, now halved.
One CPU↔GPU sync/token, routing on-GPU, transpose-contiguous copies are
metadata reshapes at seq==1.

Known remaining gaps (see TODO.md priority list for the plan):
- **Prefill (0.53-0.60x fork after combine fusion)**: the measured budget says
  attention is ~46% of the forward (transpose/contiguous copies, mask
  materialization, f16 casts, rope narrow/cat on full layers — the fork runs
  flash-attention on permute VIEWS with none of that), plus serial scheduling
  vs ggml's concurrent encoder. Routing glue measured ~1% — do NOT "fuse
  routing" for perf. The MoE combine tail is DONE (src/ops/combine.metal,
  bit-identical to candle's chain by construction, bitwise-tested;
  LAGUNA_COMBINE_CLASSIC kill-switch; provenance-enforced per tier).
- **Decode (parity with fork in both modes — 18.2 vs 18.5 LPM, 38.6 vs 39.2
  full)**: remaining levers to PASS the fork are the MoE mv_id gather (~14 ms
  sustained vs ~7 ms bandwidth floor), attention glue fusion (~6 ms
  sustained), then DFlash.
- `LAGUNA_BENCH` env var enables a warm-up forward for steady-state timing.
  Bench ≥ 256 decode tokens — sub-second runs report boost-clock fiction.

## DFlash (deferred, designed-for)

The drafter (`models/laguna-s-2.1-DFlash-BF16.gguf`, already downloaded) reads
residual-stream taps from the target: model.rs's tap capture (`h_nextn`,
per-layer `l_out`) is the attachment point. Fork implementation:
`src/models/dflash.cpp` + `common/speculative.cpp` (laguna drafters use CAUSAL
attention, generic DFlash non-causal).
