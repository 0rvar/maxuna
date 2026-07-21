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
  freezes: the `kernel_mul_mv_id_*`/`kernel_mul_mm_id_*` Metal kernels (present
  in candle's metallib with ZERO upstream Rust wiring — src/ops/ is our host
  dispatch; candle's own FusedMoeGGUF/indexed_moe_forward PANIC on non-CUDA),
  plus the public QStorage/MetalDevice surface ops relies on. Do not bump the
  rev casually; re-verify those kernels and APIs if you do.
- Zero-copy invariant (gguf.rs `expert_stack`): the stacked expert tensor is
  uploaded ONCE via `QStorage::from_data`; the Buffer handle is cloned (objc
  retain) BEFORE the storage moves into `QTensor::new`, so ExpertStack.buffer
  and the QTensor share one MTLBuffer. Candle exposes no accessor for a
  QTensor's Metal buffer — preserve this construction or you reintroduce a
  ~70GB VRAM double-copy.
- `candle_metal_kernels::metal::Buffer` is the nameable buffer type; MTLSize is
  built via candle's `get_block_dims` factory. objc2-metal was deliberately NOT
  added (candle pins 0.3.2; a mismatched dep is a trap).
- mm_id (prefill matmul) has a threadgroup-memory ceiling: `top_k × tokens ≤
  6144` per dispatch (chunk 512 × top_k 10 = 5120 OK). mm_id accumulates in
  half-precision tiles (~2e-4 rel); mv_id in f32 (~2e-7).

## Verification workflow

- Runbook: `docs/parity.md`. Fixtures with real token ids:
  `tests/fixtures/parity-prompts.json` (code-short 58 / text-mixed 82 /
  long-swa 609 — the last exercises SWA-ring wraparound).
- Pass criteria philosophy (learned in WP8): Track B full-logit gate
  (cos ≥ 0.999, top-1, top-5 ≥ 4/5) is the real gate. Track A vs
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

## Perf state (v1 final, 2026-07-21)

Warm steady-state, fused path, vs fork `llama-bench` on this machine:

| | ours | fork | ratio |
|---|---|---|---|
| decode (512 ctx) | 13.0 tok/s | 18.1 (tg128) | 72% |
| decode (4096 ctx) | 12.5 tok/s | — | |
| prefill (512) | ~60 tok/s | 361 (pp512) | 17% |
| prefill (4096) | ~59 tok/s | 328 (pp4096) | 18% |

Decode audit is clean: exactly one CPU↔GPU sync/token, routing on-GPU, no
per-token contiguous copies, compute-per-buffer sweep flat (GPU-work bound).
Bandwidth ceiling ≈ 95 tok/s.

Known remaining gaps (all kernel-level; see TODO.md):
- **Prefill: candle's `kernel_mul_mm_id` is unusable at 256 experts** — every
  threadgroup re-scans the ids buffer and the grid is sized for the worst case;
  measured 2.2 tok/s naive, 29 tok/s with a grid fix (still < mv_id's 60, and
  the fix dropped outputs). Prefill therefore uses per-token mv_id. Proper fix
  is ggml's two-pass token→expert row-map kernel. Attempts were REVERTED —
  dispatch.rs is the clean mv_id-geometry version.
- Decode: mv_id occupancy at batch=1; gate/up dispatched as two kernels (fork
  fuses them); ~8 elementwise dispatches/layer of parity-critical f16-overflow
  rescale glue in the fused MoE path.
- `LAGUNA_BENCH` env var enables a warm-up forward for steady-state timing.

## DFlash (deferred, designed-for)

The drafter (`models/laguna-s-2.1-DFlash-BF16.gguf`, already downloaded) reads
residual-stream taps from the target: model.rs's tap capture (`h_nextn`,
per-layer `l_out`) is the attachment point. Fork implementation:
`src/models/dflash.cpp` + `common/speculative.cpp` (laguna drafters use CAUSAL
attention, generic DFlash non-causal).
