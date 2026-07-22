# Deferred work ledger

## Priority order (decided 2026-07-22, next session starts at P1)

1. **Attention decode cost — PHASE 0 DONE (2026-07-22), plan revised.**
   Re-measured with isolation benches (`decode_bench` modules in attention.rs /
   moe.rs, `#[ignore]`, synthetic weights, no GGUF) + real-model sweep. Findings:
   (i) SUSTAINED vs BOOST CLOCKS: identical GPU work runs ~1.7x slower after ~1 s
   of load (full_stack_decode_bench time series: 41→76 ms plateau matching real
   78.7 ms/token within 3%) — isolation mins are boost-clock; use plateau means,
   or trust ratios only. (ii) Sum-checked sustained budget of the 78.7 ms token:
   attention ~49 (projections ~40 = f32 weight streaming at bandwidth — candle
   dequantizes the GGUF's F16 attn weights to dense f32, 11.2 GB/token; sdpa ~4;
   glue+dispatch ~6), MoE ~24 (mv_id gather ~14, routing ~6, shared ~3),
   tail+sampler ~3. (iii) Dispatch overhead is 2.4 µs/dispatch ≈ 3 ms boost
   (~6 sustained) TOTAL, and decode tok/s is flat across
   CANDLE_METAL_COMPUTE_PER_BUFFER 10→1000 — death-by-dispatch REFUTED as the
   main story; the old 48.6 ms fusion prize does not exist. Revised attack:
   (a) **f16 attention — DONE 2026-07-22** (decode 11.7 → 18.2 tok/s sustained
   LPM, ALL parity tiers pass at the f32-era anchors). Shipped as VENDORED
   mixed-dtype kernels (`src/ops/f16.metal`, f16 weights × f32 activations →
   f32, ggml's convention) after two rejected intermediates (all-f16 chain,
   cast-based hybrid) — full arc + lessons in docs/log.md. `LAGUNA_ATTN_F32`
   kill-switch; strict tier + reference dumps pin it.
   (b) fuse the remaining attention glue (~6 ms sustained: QK-norm, rope,
   softplus, casts around sdpa) — NEXT; (c) MoE mv_id gather latency (~14 ms
   sustained vs ~7 ms bandwidth floor) folds into P2. Same-mode power
   calibration DONE 2026-07-22: decode parity vs fork CONFIRMED in both power
   modes (18.2/18.5 LPM, 38.6/39.2 full); fork's historical bench figures were
   LPM; prefill gap is mode-independent 0.42-0.49x (2×2 matrix + traps in
   docs/log.md power-calibration entry).
2. **MoE combine fusion + shared map0** (prefill; ours is 0.42-0.49x the
   fork's pp512 in BOTH power modes — 174 vs 354 LPM, 415 vs 990 full).
   RESCOPED 2026-07-22 by the measured prefill budget (docs/log.md P2-phase-0
   entry; `prefill_*` benches in moe.rs/attention.rs): the ~10-dispatch
   routing glue is only ~0.6 ms/layer (~1% of prefill) — routing-glue fusion
   DEMOTED (folded into the encoder-takeover ledger item); the COMBINE chain
   (3 broadcast_muls + sum(1) over the [512,10,3072]/63MB expert output) is
   ~14.8 ms/layer ≈ 23% of prefill, ~9x its bandwidth floor — candle's
   broadcast ops take slow strided paths. Do now: (a) one fused combine
   kernel, `sum_i(down_i × col_l2 × 1/32768 × w_i)` reading the expert output
   once, f32 accumulation mirroring candle's reduce order, kill-switch + gated
   like every owned kernel; (b) compute the mm_id map0 row-map ONCE per MoE
   block instead of 3x (bit-identical by construction; the fork recomputes
   too, so it's an absolute win over it — and the gate 3.0 vs up 7.9 ms
   anomaly on identical shapes needs explaining first). Expected ~25% prefill.
3. **Attention block cost, prefill edition** — the prefill budget puts
   attention at 23.6 (full) / 30.1 (SWA) ms/layer ≈ 46% of the whole forward:
   the single biggest prefill category, ahead of everything MoE. Subsumes the
   old P1(b) decode glue fusion (~6 ms sustained: QK-norm, rope, softplus,
   transpose/contiguous copies, sdpa casts + mask materialization) — attack
   both seq regimes together.
4. **MoE-block encoder takeover** (ledger item below) — concurrency +
   range-tracked barriers; now also owns the demoted routing-glue fusion.
5. **mmap + no-copy model load** (see ledger item below) — dev-velocity
   multiplier: every parity/bench cycle pays ~30 s/load today.
6. **DFlash speculative decoding** — multiplies whatever decode speed exists,
   so it lands last.

- [x] **Prefill mm_id kernel (biggest perf item)** — DONE. Vendored ggml's
  two-pass token→expert row-map kernels (`src/ops/mm_id.metal`, runtime-compiled
  via `src/ops/pipelines.rs`); `FusedExperts::forward` uses mm_id for seq>=32,
  mv_id below. Prefill ~60 → ~188 tok/s (3.1x): classic simdgroup port first
  (~151), then the cooperative-tensor `matmul2d` path (now default, ~188). Still
  short of the fork's ~361 pp512 — remaining gap is surrounding candle dispatch
  overhead, not the matmul (see profile below). Follow-ups:
  - [x] **mm_id tile precision — RESOLVED**: f32 tiles (`_hp`) are the default;
    `LAGUNA_MM_ID_F16=1` selects the f16 variant for A/B vs the future tensor-path
    port. f32 tiles removed the f16 operand rounding (oracle rel 2.6e-4 → 1.94e-7,
    ~1330x, at ~0 throughput cost). The residual model-level drift is tiled f32
    K-accumulation ORDER, not precision — fork-equivalent (the fork's tensor-path
    prefill is ~0.9962 raw and fails the strict 0.999 gate the same way). code-
    short cosines vs Reference by config: mm_id hp glue-off (shipped default)
    0.99687 top-1 268; mm_id hp with the (now-removed) rescale glue 0.99694 top-1
    350; both are a legit near-tie there (reference 350/268 margin 0.319 < 0.5),
    so the mm tier passes either. docs/parity.md §3b documents the two-tier gate:
    strict cos>=0.999 for mv_id/decode, fork-equivalence
    (cos>=0.995 + top-1 match-or-near-tie) for mm_id prefill.
  - [x] **Cooperative-tensor mm_id path PORTED** (the fork's Metal-4 `matmul2d`
    fast path, `kernel_mul_mm_id_t` in mm_id.metal). It is now the DEFAULT prefill
    variant. Probe (mm_id.rs `tensor_matmul2d_probe`) confirmed `<metal_tensor>` +
    `mpp::tensor_ops::matmul2d` compile under candle's default options on the M5.
    Prefill: 151 -> 188 tok/s @925-tok chunk (+24%), 142 -> 157 @4230 (+10%);
    decode unchanged (mv_id). Still short of the fork's ~361 pp512 (see profile
    below). Parity: fork-equivalent mm tier (cos 0.99699 top-1 268), passes; the
    tensor path agrees with the classic-f16 path to 4.4e-7 (per-matmul).
    Runtime-selectable variants (cached env, `ops::MmVariant`):
      - default: tensor f16 tiles (`_t`) — fastest (188), f16-operand, needs rescale.
      - `LAGUNA_MM_ID_TENSOR_HP=1`: tensor f32 tiles (`_t_hp`) — float `matmul2d`
        DOES compile and is f32-precise (1.94e-7/matmul), but SLOWER at model scale
        (163 tok/s: double-width tiles cost more than the rescale they save) and
        still mm-tier at model scale (tiled drift, 0.99687), so not the default.
      - `LAGUNA_MM_ID_CLASSIC=1`: classic f32 simdgroup tiles (`_hp`), 151 tok/s.
      - `LAGUNA_MM_ID_F16=1`: classic f16 simdgroup tiles.
  - [ ] **Remaining gap to the fork's ~361 pp512** — the mm itself is 2 dispatches
    (map0 + matmul2d) per projection and is well-amortized in prefill; the ~188 vs
    361 gap is the ~50 surrounding candle dispatches/MoE+attn layer (route/sigmoid/
    argsort/gather, the ~6-op L2 rescale glue, silu*mul, combine, shared expert,
    and the attention chain) that ggml fuses into far fewer, plus candle per-op
    overhead. The rescale glue is NOT the bottleneck (tensor-hp has none yet is
    slower). 2026-07-22 measured budget re-attributed this gap: combine chain
    ~23%, attention ~46%, routing glue only ~1% — see priority items 2-4.
- [ ] **Decode kernel work** — 12.5 -> 13.5 tok/s after glue removal; fork is
  18.1 @512ctx. Per-token budget (2026-07-22, ~512 ctx). FFN sweep (80.0 ms
  total): routed mv_id gate/up/down gather 18.4 ms; routing+combine 4.3 ms;
  rescale glue 3.1 ms (now removed by default); shared expert 1.3 ms; non-FFN
  52.9 ms. Attention sub-sweep (76.9 ms total): attention all-48-layers 50.9 ms
  (66%), of which the sdpa core is only 2.3 ms and the ~24 non-sdpa dispatches/
  layer (q/k/v/g projections, QK-norm, rope, f16 casts, 3x transpose-contiguous,
  cache append, softplus gate, o_proj) are ~48.6 ms; lm_head 6.5 ms; norms+
  sampler+embedding+residuals ~19.5 ms. STORY: death-by-dispatch — sdpa math is
  cheap, the per-layer dispatch overhead around it dominates.
  - [x] **Rescale glue removed from the default path** — the L2 rescale only
    guarded the f16 activation cast in the mm_id f16-tile kernel (opt-in
    `LAGUNA_MM_ID_F16`, staged as half). The default down paths never cast the
    activation to f16: mv_id reads f32 and accumulates f32 (candle
    quantized.metal kernel_mul_mv_q{4,6}_K_f32_impl :4889/4930, :5188/5225);
    mm_id-hp stages src1 as float. So the glue is skipped by default (kept only
    when `LAGUNA_MM_ID_F16`). +8% decode (12.5 -> 13.5), prefill ~149 -> 157.
    Verified: no inf/nan on code/mixed/long-text (609) prefill or greedy decode;
    strict mv tier 0.99906 (>=0.999), mm tier 0.99687; the code fixture's 350/268
    is a genuine near-tie (mv gap 0.16, mm gap 0.23), mixed/long are decisive 350.
  - [x] **Fused-activation kernels RETIRED (both) — LESSON: the MoE router is a
    chaos amplifier for activation-path rounding.** Two vendored kernels were
    built and removed: (a) a fused silu/mul/L2/rescale kernel whose f32 L2
    reduction-order differed from candle's by ~1e-6, and (b) a plain elementwise
    silu*mul kernel differing by ~1e-7 (division vs candle's multiply-by-
    reciprocal). Both were per-op-correct (end-to-end 1.6e-7/layer vs candle) yet
    cascaded through the router — a ~1e-6 activation nudge flips near-tie expert
    selections in later layers — to 1.3e-3..1.5e-3 final-logit divergence, under
    the strict gate. CONSTRAINT for future kernel work: do NOT reimplement any op
    upstream of the router (activation, norm, router logits) unless it is
    BIT-IDENTICAL to candle; post-router ops (down output, lm_head) are safe to
    fuse (no cascade). The glue-removal win needed no kernel — it just dropped the
    now-unnecessary rescale and kept candle's silu*mul.
  - [x] **Phase-0 re-measurement of this budget — DONE 2026-07-22; the numbers
    above are superseded by the P1 entry in the priority list.** The 50.9 ms
    attention total was roughly right at sustained clocks, but its attribution
    was wrong: it is ~80% f32 weight streaming (candle dequantizes F16 attn
    weights to f32), not dispatch overhead (measured 2.4 µs/dispatch; PER_BUFFER
    sweep flat). lm_head is ~1.3 ms sustained (not 6.5); MoE half ~24 ms
    (mv_id gather ~14). "Fuse the dispatch chain" as originally scoped is
    retired; see priority list P1 for the f16-first plan. Measurement harness:
    ignored benches `attn_decode_chain_bench` / `attn_decode_ablation_bench` /
    `attn_proj_f16_bench` / `attn_decode_f16_chain_bench` /
    `dispatch_overhead_bench` (attention.rs) and `moe_decode_ffn_bench` /
    `sampler_decode_bench` / `token_tail_bench` / `full_stack_decode_bench`
    (moe.rs); run via `cargo test --release <name> -- --ignored --nocapture`,
    iters via LAGUNA_BENCH_ITERS/WARMUP. CAVEAT baked into the harness: compare
    plateau means (time series printed by full_stack bench), not means-vs-mins
    across variants — boost-clock decay otherwise poisons ablation deltas.
  - [x] **Vendored ggml mv geometry for the routed gather + lm_head — DONE, but
    NO measurable decode gain (LESSON: the mv compute was not the bottleneck).**
    Ported ggml's CURRENT `kernel_mul_mv_{id_,}q{4,6}_K_f32_impl` geometry
    (N_R0=2, N_SG=2, `(r0*NSG+sgitg)*nr0` row fan-out, nr0 register-row f32
    accumulate) into `src/ops/mv.metal` (separate library — no Metal-4 tensor
    dep), host dispatch in `dispatch.rs` (`encode_mul_mv_id_vendored`,
    `run_plain_mv`), default for q4_K/q6_K with `LAGUNA_MV_CLASSIC` kill-switch
    reverting to candle's baked kernels. lm_head bypass at seq==1 over a retained
    shared buffer (`gguf::qlinear_with_buffer`, same zero-copy trick as
    ExpertStack). Correctness solid: greedy decode gate passes all three fixtures
    (code-short 62/2 excused, text-mixed 64/0, long-swa 59/5 excused, 0
    non-excused); decode-tier diagnostic cosine 0.99789 (top1 350=ref, top5 4/5;
    the accumulation-reorder drop from classic's 0.99906 is expected per §3b).
    BUT end-to-end decode is FLAT: 13.1 (vendored) vs 13.0 (classic) tok/s @512ctx,
    256-tok warm bench. The premise that candle's mv "runs ~15x under bandwidth /
    lm_head ~6.5 ms" does not reproduce in isolation: a `[100352x3072]` q6_K
    matvec at seq==1 is 0.685 ms vendored vs 0.738 ms QMatMul (both near the
    ~0.62 ms/250MB bandwidth floor; microbench `plain_mv_lmhead_bench`, ignored).
    So both hot mv paths were already ~bandwidth-optimal in candle; the 6.5 ms
    lm_head / 18.4 ms gather line items are per-dispatch LATENCY inside the full
    decode pipeline, not mv compute — geometry can't recover them. The vendored
    kernels are strictly not slower and are more fork-faithful (ggml's current
    geometry), so kept as default, but the real decode prize remains the
    attention per-layer dispatch chain (48.6 ms). DECIDED (Orvar, 2026-07-22):
    vendored stays the default — insulates these two paths from upstream candle
    kernel changes; `LAGUNA_MV_CLASSIC` remains the escape hatch.
- [ ] **MoE-block encoder takeover: concurrency + range-tracked barriers**
  (deferred follow-up to P2, decided 2026-07-22) — the fork's 2x prefill edge on
  Metal is NOT fusion (it has no Metal topk_moe kernel; ~30 dispatches/block)
  but scheduling: one compute encoder with concurrent dispatch where a memory
  barrier is inserted ONLY when a dispatch's buffer ranges overlap a pending
  write (ggml-metal-ops.cpp mem_ranges :150-210), graph reorder into concurrent
  sets (ggml-metal-common.cpp:209/375), and multi-threaded command-buffer
  encoding (ggml-metal-context.m:550, up to 8 threads). Candle's eager op-per-
  encoder submission serializes all of it — e.g. the shared expert and the tiny
  routing kernels could overlap the heavy expert GEMMs but don't. Taking over
  encoding of the whole MoE block inside our dispatch layer (all ops between
  ffn_norm and the residual add issued into one concurrent encoder with
  ggml-style range tracking) is the remaining big prefill lever after routing-
  glue fusion + map0 sharing land. Requires every op in the block to be an
  owned kernel (or a candle op we re-encode) — the strategic endgame of
  vendoring.
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
- [ ] **mmap + no-copy model load** — warm load is ~30s because gguf.rs reads
  each tensor into a heap Vec and `QStorage::from_data` copies it again into a
  fresh MTLBuffer (~2 full passes over 75GB), plus the expert_stack re-layout
  pass. The fork loads near-instantly warm via mmap + `newBufferWithBytesNoCopy`
  (unified memory: GPU reads the page-cache pages in place; one page-aligned
  buffer over the whole file, per-tensor offsets). For us that needs (a) a
  one-time repack cache file with experts ALREADY stacked in our layout (no-copy
  can't reorder, and the stacks are most of the 75GB), mmapped on subsequent
  loads, and (b) a check whether the pinned candle rev can wrap an existing
  Metal buffer as QStorage/Tensor for the QMatMul + F16 attention paths (our
  vendored kernels take raw buffers already). Expected: warm load → a few
  seconds; first-touch page wiring moves into the first forward (already
  excluded from steady-state numbers).

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
- [ ] **Chat REPL display edge cases** (src/bin/laguna/repl.rs) — the raw-mode
  editor repaints the input block relative to the cursor row, so (a) an input
  taller than the terminal window glitches visually (buffer/submission stay
  correct), and (b) a terminal resize that reflows already-printed rows can
  misplace the repaint anchor until the next submit. Fine for chat-sized input;
  fix = cap the visible block to a viewport (scroll within it) if it ever bites.
  No persistent input history across sessions (in-memory only).
- [ ] **mm_id-dispatch counter in dump provenance** — the greedy/full-logit gates
  now enforce runner provenance, but `provenance` records the mm-*eligibility*
  predicate (`moe_impl == "fused" && seq_len >= mm_min_seq && !no_mm_id`), not
  whether the mm_id kernel actually dispatched at runtime. A checkpoint whose
  dtype/top_k falls back to mv_id (via `supported()`) still reports the mm path as
  "active", so a fused dump can pass the mm tier without any mm_id dispatch — a
  residual false-pass. A runtime mm_id-dispatch counter surfaced into dump
  provenance would close it. Deferred: the ops dispatch layer is under concurrent
  rework, so touching it now would collide.
- [ ] **`LAGUNA_MV_CLASSIC` in dump provenance** — the strict tier's mv-classic env
  can't be asserted from the dump (`attn_dtype`/`no_mm_id` now are); close if
  provenance grows a runtime kernel-dispatch record (same mechanism as the
  mm_id-dispatch counter item above).
- [ ] **Small-batch f16 attention gemv (`mul_mv_ext`)** — the vendored f16-weight
  attention matmul (src/ops/f16.metal) mirrors ggml's mv/mm split at ne11 > 8, but
  skips ggml's `mul_mv_ext` small-batch kernels (its preferred path for ne11 2..8);
  those seqs ride the plain gemv with one grid.y column per token. Only reachable
  by a sub-9-token prefill (never on the decode or chunked-prefill hot paths), so
  perf-only and tiny. Vendor `kernel_mul_mv_ext_f16_f32_r1_N` if it ever matters.
- [ ] **Cooperative-tensor f16 attention prefill gemm** — the vendored f16-weight
  prefill gemm (src/ops/f16.metal) uses the classic simdgroup path; the fork on M5
  would take the cooperative-tensor mul_mm for f16×f32. Potential prefill perf
  follow-up; correctness unaffected.
- [x] **Combine-library reassociation hardening — RESOLVED via source pragma**
  (2026-07-22). The fused combine kernels' bit-identity to candle's chain rests
  on the rescale multiply chain (`r1 = d*l; r2 = r1*2^-15; r3 = r2*ww`) NOT
  being reassociated; `fp contract(off)` alone doesn't cover that. The
  MTLCompileOptions route is impossible at the pinned candle rev (no re-export
  of `objc2_metal`, no factory returning options — adding objc2-metal is the
  documented trap), so combine.metal carries `#pragma clang fp reassociate(off)`
  at file scope instead. clang REJECTS unknown `fp` pragma options, so the
  library compiling (it does — `fused_matches_candle_bitwise` passes) proves the
  pragma is honored. Revisit only if a candle bump ever exposes compile options.
