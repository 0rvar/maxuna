# Deferred work ledger

- [ ] **Prefill mm_id kernel (biggest perf item)** — prefill runs at ~17% of the
  fork because candle's `kernel_mul_mm_id` re-scans ids per threadgroup and
  grids for the worst case (unusable at 256 experts; measured, then reverted —
  see CLAUDE.md perf section). Port ggml's two-pass token→expert row-map
  approach (vendored kernel or upstream candle PR); target the fork's ~360 tok/s
  pp512.
- [ ] **Decode kernel work** — 13.0 vs fork 18.1 tok/s: fuse gate/up expert
  matvecs into one dispatch, revisit mv_id occupancy at batch=1, and audit the
  ~8 elementwise dispatches/layer of f16-overflow rescale glue (parity-critical
  — re-run Track B after any change).
- [ ] **Track B dumps for text-mixed / long-swa** — the full-logit reference-vs-
  fused gate ran only on code-short (greedy covers the other two fixtures);
  generate the remaining dumps if fused ever changes.
- [ ] **ref-dump.sh greedy oracle** — still calls `llama-cli -st -no-cnv`, which
  applies the chat template; swap to the llama-server /completion token-array
  method documented in docs/parity.md.
- [ ] **KV-cache reuse across chat turns** — the REPL re-prefills the whole
  conversation each turn (correct but O(n²) over a long chat); reuse the cache for
  the common prefix instead.
- [ ] **Steady-state prefill timing** — the first forward folds in the one-time
  Metal weight upload, so reported prefill tok/s is misleading; add a warm-up
  forward before timing (or report load-adjusted numbers).
- [ ] **Fine-grained parity taps** — model.rs captures layer-level residual taps
  only; AttnBlock/MoeBlock expose no sub-node intermediates (Qcur_rope,
  attn_gated, ffn_moe_out, …), limiting first-divergence bisection to layer
  granularity. Add hooks if a real divergence ever needs sub-layer localization.

Items deliberately out of v1 scope. Append as new deferrals come up during
implementation — never silently drop scope.

- [ ] **DFlash speculative decoding** — trained drafter at `poolside/Laguna-S-2.1-DFlash`
  (BF16 GGUF already in `models/`). Drafter consumes residual-stream taps from target
  layers (`t_layer_inp[il]`, `t_h_nextn` in the fork's laguna.cpp); `model.rs` keeps
  per-layer residual capture hooks feasible for this. Biggest post-v1 perf lever.
- [ ] **HTTP server** (OpenAI-compatible /v1/chat/completions) so coding agents can
  connect; v1 is CLI-only per scope decision.
- [ ] **Self-quantized Q5/Q6 tier** — official GGUF repo only ships Q4_K_M (75.2GB),
  Q8_0 (127.7GB, exceeds 128GB RAM) and F16 (235GB) + imatrix. A Q6_K (~97GB) built
  with the fork's `llama-quantize` from F16 + the published imatrix would be the true
  "largest quant that fits" (needs raised `iogpu.wired_limit_mb` and capped context).
- [ ] **min-p sampling** — generation_config defaults min_p=0 so v1 omits it;
  candle's LogitsProcessor lacks it (would be a custom sampler stage).
- [ ] **Batching** — v1 is deliberately batch=1 (single-user local inference).
- [ ] **Tool-call / reasoning stream parsing** — emitting structured `<tool_call>` /
  `<think>` blocks as parsed events instead of raw text (needed for the server).
  Also inbound: `chat::Message` has no assistant tool-call variant and no tools-list
  header rendering, so the template's tool branches are currently unreachable.
- [ ] **Tokenizer from GGUF metadata** — `LagunaTokenizer::from_gguf` intentionally
  errors ("pass tokenizer.json via --tokenizer"); reconstructing byte-level BPE +
  the 70-entry added vocab from `tokenizer.ggml.*` arrays wasn't worth it while
  tokenizer.json ships with the checkpoint. Revisit if we want single-file UX.
- [ ] **1M-context tuning** — the official GGUF is a 256k conversion (YaRN factor 32,
  `laguna.context_length=262144`); the HF checkpoint config claims 1M via factor 128
  (net mscale 1+0.1·ln(factor) either way). Going past 256k means overriding the rope
  scaling at load (and ~48GB f16 full-attn KV at 1M). v1 caps max_ctx at 32768.
- [ ] **Sampling-defaults discrepancy** — the GGUF metadata carries
  `general.sampling.temp=0.7, top_p=0.9` while generation_config.json says
  temp 1.0 / top_k 20 / top_p 1.0. v1 follows generation_config; revisit if outputs
  seem off-distribution.
