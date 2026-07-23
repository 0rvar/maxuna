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
   (b) fuse the remaining attention glue — DONE via 3(c) glue fusion + the
   glue phase 2 ledger item (3 more cast dispatches/layer folded);
   (c) MoE mv_id gather latency (~14 ms
   sustained vs ~7 ms bandwidth floor) folds into P2. Same-mode power
   calibration DONE 2026-07-22: decode parity vs fork CONFIRMED in both power
   modes (18.2/18.5 LPM, 38.6/39.2 full); fork's historical bench figures were
   LPM; prefill gap is mode-independent 0.42-0.49x (2×2 matrix + traps in
   docs/log.md power-calibration entry).
2. **MoE combine fusion + shared map0 — DONE 2026-07-23.** Prefill 174 → 211
   tok/s @925 (+21%), 150-160 → 184 @4k; decode unchanged; all parity tiers
   pass at the exact prior anchors with regenerated references because the
   fused combine is BIT-IDENTICAL to candle's chain by construction (bitwise
   acceptance test, `src/ops/combine.metal` mirrors candle's reduce geometry
   at candle's rounding boundaries; `fp contract(off)` + `fp reassociate(off)`
   pin it toolchain-wide). `LAGUNA_COMBINE_CLASSIC` kill-switch; `combine`
   provenance enforced per tier. map0 computed once per MoE block via typed
   `Map0Scratch`. Review round (3 models): five hardening items applied, no
   live bugs; the gate-3.0/up-7.9 bench anomaly was DVFS burst fiction (bench
   now interleaves). Full arc: docs/log.md 2026-07-23 entry. (Routing-glue
   fusion stays demoted into the encoder-takeover item — measured ~1%.)
3. **Attention block cost, prefill edition** — sub-budget measured 2026-07-23
   (`prefill_attn_*` benches; per-forward: projections 585 ms, mask 415 → NOW
   ~16, transposes 119, sdpa 90, gate 82, rope 75, qk-norm 21).
   (a) **mask hoisting — DONE 2026-07-23** (48 builds/forward → 2 per chunk,
   byte-identical, no kill-switch; prefill 211 → 228 @925, 184 → 237 @4k;
   docs/log.md entry; SUBSUMED by (d) — the flash path builds no mask at
   all). (b) projections: cooperative-tensor f16 gemm port — PORTED then
   REVERTED to opt-in (gate-rejected as default; ledger item), the
   mixed-operand escape hatch PROBED AND CLOSED (bit-identical but
   classic-speed; ledger item) — then UNBLOCKED 2026-07-23 by (d) and made
   the DEFAULT the same day (owner-approved; kill-switch
   LAGUNA_ATTN_MM_CLASSIC; ledger item has the gate/bench numbers).
   (c) **glue fusion — DONE 2026-07-23**: three fused kernels, each
   bit-identical by construction (combine playbook): fused softplus-gate
   (kernel_attn_gate, ~10-op candle chain + broadcast_mul → 1), fused
   permute+cast copies (kernel_permute_cast_*, transpose+contiguous+f16-cast
   in one pass), vendored partial-rotary rope (kernel_rope_neox, no
   narrow/cat). Kill-switch `LAGUNA_ATTN_GLUE_CLASSIC`, provenance `attn_glue`
   enforced per tier. Prefill dispatches/attn-layer 26 → 10 (full) and
   20 → 10 (SWA); decode 22 → 7 (the fused gate/rope/cast run at seq==1 too —
   subsumed the old P1(b) decode glue item). Measured (LPM): prefill 238.5 →
   262.1 @925, 235.8 → 255.6 @4k; decode 18.2 → 19.1 (now 1.03x the fork's
   18.5). All six gates PASS at the exact known-good anchors (bit-identity
   confirmed at model level); references regenerated for the attn_glue
   provenance field, reference-ppl mean_nll bit-identical 4th regen running.
   Review round: 4 reviewers (Claude/Codex/GLM/DeepSeek); hardening applied:
   `#pragma METAL fp math_mode(fast)` in all 9 vendored .metal sources (pins
   the documented nil-options default against future OS flips; the fork also
   compiles Fast — checked ggml-metal-device.m), load-time hard error if
   CANDLE_METAL_ENABLE_FAST_MATH is set falsy (mixed-mode would silently void
   every bitwise-identity contract), overflow-checked size products +
   same-device checks in the glue dispatches, offset-view bitwise test,
   fail-fast attn_glue in isReferenceDump. Glue was a TRAFFIC problem, not
   scheduling (territory map): fusion removed the copies concurrency could
   only have overlapped; the takeover now schedules fewer, fatter kernels.
   (d) **flash-attention kernel — DONE 2026-07-23** (ledger item below):
   vendored steel kernel, in-kernel causal/window mask + block skip, no mask
   tensor, f32 boundaries; prefill 264.0 @925 / 268.8 @4k flash-only,
   301.5/303.3 with tensor (0.85x/0.87x fork); all six gates pass in both
   configs; the sdpa/mask/transpose+cast rows of the sub-budget above are
   gone or absorbed on the flash path.
4. **MoE-block encoder takeover — CLOSED 2026-07-23, premise REFUTED**
   (ledger item below has the full analysis): a static trace of candle's
   hazard tracker over one layer's 51 dispatches shows 37 barriers, all true
   RAW dependencies (whole-buffer keying adds ~0-3 false barriers worst case,
   <1% of GPU time), and the independent fat ops already share barrier-free
   windows; empirical bracket `CANDLE_METAL_COMPUTE_PER_BUFFER=1` (one
   encoder per dispatch, max serialization) leaves prefill IDENTICAL
   (304.0 tok/s both ways). Scheduling is not where the remaining prefill
   gap lives — the fork's edge is per-kernel efficiency. The demoted
   routing-glue fusion dies with it (~1%, measured twice).
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
- [x] **MoE-block encoder takeover — CLOSED 2026-07-23, REFUTED before
  build.** The re-measure-first clause below fired: a static simulation of
  candle's verified tracker semantics (encoder.rs:104-149: RAW checked vs the
  accumulated window, WAR/WAW on outputs; barrier = global
  MTLBarrierScope::Buffers, window resets to the triggering dispatch) over
  the full 51-dispatch default-prefill layer trace found 37 barriers — ALL
  true producer→consumer RAW deps in a near-linear chain. Whole-buffer
  keying adds ~0-3 false barriers worst case (<1% GPU time): candle's pool
  only recycles buffers whose producing Tensor dropped (strong_count==1,
  device.rs:294-311/472-486), so live producers never alias, and weights are
  input-only (never enter the hazard write-set → read-read can't barrier).
  The independent fat ops ({g,q,k,v} projections, mm_id gate/up,
  combine+shared-expert matmuls) ALREADY sit together in barrier-free
  windows; the hard RAW spine (flash, o_proj, mm_id-down, residuals) can't
  be reordered around. The fork's mechanism was also re-read: its barrier is
  EQUALLY coarse (one memory_barrier + full mem_ranges reset,
  ggml-metal-common.cpp:124-153); its wins are byte-range precision (which
  candle's allocator makes moot here) and reorder (which has nothing left to
  pack); at batch=1 it encodes on 1-2 threads, warning >2 degrades.
  EMPIRICAL BRACKET (2026-07-23, LPM): `CANDLE_METAL_COMPUTE_PER_BUFFER=1`
  — one encoder per dispatch, i.e. maximal serialization + a fence-wait per
  dispatch — leaves prefill-925 IDENTICAL at 304.0 tok/s (the 8-token decode
  tail drops 26.8→19.3, confirming the knob works and only dispatch-scale
  work cares). If 50x more fences cost prefill nothing, scheduling
  improvements can't buy anything either. VERDICT: prefill is fat-kernel
  compute-bound end to end; the remaining 0.86-0.90x fork gap is per-kernel
  efficiency, not concurrency. Byte-range patch, own-encoder takeover, and
  issue-order reorder are all dead ends here; the demoted routing-glue
  fusion (~1%) dies with the item. Original premise + constraints kept below
  for the record.
  (original item, superseded:) deferred follow-up to P2, decided 2026-07-22
  — the fork's 2x prefill edge on
  Metal is NOT fusion (it has no Metal topk_moe kernel; ~30 dispatches/block)
  but scheduling: one compute encoder with concurrent dispatch where a memory
  barrier is inserted ONLY when a dispatch's buffer ranges overlap a pending
  write (ggml-metal-ops.cpp mem_ranges :150-210), graph reorder into concurrent
  sets (ggml-metal-common.cpp:209/375), and multi-threaded command-buffer
  encoding (ggml-metal-context.m:550, up to 8 threads).
  PREMISE CORRECTED 2026-07-23 (territory map): candle does NOT do eager
  op-per-encoder submission at our pinned rev — it already shares ONE
  Concurrent-dispatch encoder across ~50 dispatches (commands.rs:160-190,
  `CANDLE_METAL_COMPUTE_PER_BUFFER`), with its own hazard tracking on
  untracked buffers (encoder.rs:104-149) — and our custom ops already ride
  that encoder and participate in its hazard tracking (dispatch.rs
  set_input_buffer/set_output_buffer). What candle LACKS vs the fork is (a)
  BYTE-RANGE hazard granularity — candle keys on whole buffer pointers and
  one hazard emits a memoryBarrier(Buffers) that resets the ENTIRE concurrent
  set, so a dependency-chained forward degrades to serial — and (b) graph
  reorder pulling independent nodes into the same concurrent window. So the
  realizable prize is range-tracking + reorder only; re-measure with a GPU
  trace how much overlap candle already achieves before investing. Known
  constraints from the map: buffer extraction for dense tensors is already
  solved (storage_and_layout → MetalStorage::buffer, no QStorage-style trick
  needed); candle's buffer POOL recycles pointers, so any range tracker must
  survive pointer aliasing of just-freed buffers; running our OWN encoder
  (vs riding candle's) requires replicating the prev_ce_outputs cross-encoder
  fence protocol (commands.rs:340-382) or risking silent RAW corruption; and
  reordering candle-issued ops requires vendoring them — a takeover that only
  wraps our own kernels buys little because interleaved candle ops keep
  flipping the shared encoder's hazard state. Sequence AFTER 3(c) glue fusion:
  fewer, fatter kernels schedule better and the copies it removes are traffic
  concurrency could only overlap, not eliminate.
- [x] **Attention glue phase 2 — DONE 2026-07-23** (scope revised post-flash:
  flash's f32 output had already killed the prefill post-sdpa cast, and the v
  cast was already folded into permute_01_f16 — the map found prefill had
  exactly ONE standalone traffic pass left). Landed: `kernel_rope_neox` is
  templated on the store type (rope math stays f32; one RTNE rounding at the
  f16 store = the old separate cast, passthrough dims included) —
  `run_rope`/`Rope::apply_dt` take an out_dtype (F32/F16 only, enforced on
  both the fused and chain paths); `kernel_attn_gate` templated on the attn
  input type (f16 widened in-kernel, exact). Call-site policy: k rope-stores
  f16 always on the fused path (feeds the f16 cache directly; the standalone
  cast_f16(k) is gone); q rope-stores f16 only at seq==1 off the
  LAGUNA_SDPA_F32 experiment (whose sdpa takes f32 q); flash prefill keeps
  f32 q (flash requires it); decode consumes the sdpa f16 output straight
  into the f16-input gate (post-sdpa cast_f32 gone). Decode v cast stays (v
  is never roped — no fold target). No new kill-switch/provenance/schema:
  bit-identical by construction, LAGUNA_ATTN_GLUE_CLASSIC path untouched.
  Two new bitwise tests (rope f16-store vs f32+cast incl. n_rot=64
  passthrough + q/k prefill/decode shapes; f16-input gate vs widen+f32
  gate). All six gates PASS digit-exact at the pre-change anchors (strict
  0.999057, mm 0.996564, decode 63/63/60 + 0 unexcused, Δnll 0.003426) with
  references reused — model-level bit-identity confirmed. Bench (LPM):
  prefill 304.7 → 304.0 @925 (flat), 303.8 → 311.9 @4k (+2.7%, the removed
  k-cast pass scales with seq), decode 18.8 → 19.2 (±5% band, directionally
  right — 3 dispatches/layer gone). Review round: Claude + Codex + GLM +
  DeepSeek, zero correctness findings; applied hardening: dtype guard on the
  rope chain fallback, q-decode shape added to the bitwise test.
- [ ] **Prefill per-kernel efficiency vs fork — locate the ~0.9x gap**
  (opened 2026-07-23 by the encoder-takeover refutation: scheduling is
  exonerated, so the remaining prefill deficit — 304/312 vs the fork's
  354/348 LPM — must live inside individual kernels). Plan: per-op GPU-time
  comparison against the fork on the same shapes — our side via the existing
  isolation benches (`prefill_attn_*`, mm_id/f16 bench modules) or xctrace;
  fork side via GGML_METAL_PERF or xctrace on llama-bench. Suspects, by
  share of prefill time: mm_id tensor-tile gemm vs ggml's mul_mm_id
  (fork's simdgroup kernel may still win at Q4_K), the f16 projections
  matmul2d vs ggml mul_mm_f16_f32, flash kernel vs ggml
  flash_attn_ext_f16 (BQ/BK tiling + softmax layout differ), and candle's
  rms_norm. NOTE the LPM caveat: compare ratios measured in the same power
  mode only.
- [x] **Provenance schema version — DONE 2026-07-23** (landed with the flash
  work: PROVENANCE_SCHEMA_VERSION=3 in src/parity_schema.rs, `flash`
  introduced at v3 grandfathered to "classic"; cached references survived
  both the flash field and glue phase 2 with ZERO regen — the tax it was
  built to kill). Original proposal: every new provenance field
  invalidates all cached/committed reference dumps (missing field = hard fail
  = ~40 min GPU regen; paid 3x now: combine, attn_mm, attn_glue). Fix: a
  PROVENANCE_SCHEMA_VERSION const shared by logits-dump and tests/parity.rs;
  a dump missing a field but stamped with a schema version OLDER than the one
  that introduced it is grandfathered to that field's classic-era value (the
  only path its binary had); missing at current version stays a hard fail
  (true stale-binary). Preserves the stale-binary defense, makes references
  durable across field additions.
- [ ] **Sweep the pre-glue dispatches for the review-round hardening**
  (2026-07-23): run_combine (and any other pre-existing multi-input dispatch)
  still has unchecked size products and no same-device check on secondary
  inputs (weights/col_l2) — same latent pattern the glue ops just got fixed
  for (checked_elems + Device::same_device). Also isReferenceDump in
  parity-gate.ts fail-fasts on moe_impl/attn_dtype/combine/attn_glue but not
  attn_mm ("f32-bypass") — the Rust gate catches it, but only after an
  expensive candidate run.
- [x] **Per-op drift attribution — DONE 2026-07-23, hypothesis CONFIRMED,
  perf blocked on flash attention.** Three-config gate matrix via the new
  `--sdpa-f32` / `--attn-mm-tensor` experiment flags (env-gated f32 sdpa
  path: LAGUNA_SDPA_F32, candle's native f32 Metal sdpa kernels, per-call
  exact widening of cached f16 k/v; provenance field `sdpa`, schema v2):
  (1) tensor-only control reproduced the rejection exactly (mm 0.995842,
  Δnll 0.004732, step-29 flip) on the post-glue binary; (2) sdpa-f32 alone
  passes but drifts slightly off the f16-sdpa reference (mm 0.996568, Δnll
  0.004305 — pure differential noise, as predicted for a common-mode
  change); (3) tensor + sdpa-f32 passes ALL SIX and step 29 AGREES OUTRIGHT
  (63/64 + the perennial step-0 excuse; Δnll 0.003426 — better than either
  single change). Mechanism confirmed: f16 sdpa AMPLIFIES upstream f16
  staging noise; f32 accumulation damps it. So the decode envelope does not
  block tensor projections — our sdpa precision does. BUT the experiment
  implementation is perf-negative (LPM: prefill 235.2 @925 / 204.1 @4k vs
  shipped 262/256; decode 15.2 vs 19.1): f32 sdpa doubles O(T²) attention
  traffic and re-widens the whole KV cache every forward. No shippable
  configuration of the current pieces is net-positive. The unlock is the
  mixed-precision flash-attention kernel (next item). Defaults unchanged;
  experiment path kept env-gated.
- [x] **Flash-attention kernel — DONE 2026-07-23, both payoffs delivered.**
  Vendored candle's MLX steel attention kernel (`src/ops/flash.metal`, own
  runtime-compiled library, no Metal-4 dep) rather than porting ggml's — the
  mapping phase found candle's steel kernel ALREADY accumulates QK/softmax/O
  in f32 (MLX lineage upcasts operands before the MMA, stronger than ggml's
  f16-operand MMA), so the differential noise in the old default was purely
  the boundary casts (q f32→f16 in, out f16 store) plus the materialized
  mask. The vendored config: Q device-f32 → f32 smem, K/V device-f16
  head-strided cache views (consumed strided, never packed), f32
  accumulation, f32 O store — value-identical to the LAGUNA_SDPA_F32
  experiment without widening KV. Mask tensor DELETED: in-kernel
  causal+window visibility (`(j+k_off) <= (i+q_off) && (i+q_off)-(j+k_off)
  < window`) with block-level skip (kb_start/kb_lim), bit-equivalent to the
  additive 0/-inf mask (exp(-inf) terms are exact zeros); model.rs skips the
  PrefillMask build entirely on the flash path (was 1.5-2.3GB of mask at
  4k). Prefill (seq>1) only; decode untouched. Kill-switch
  LAGUNA_FLASH_CLASSIC; provenance field `flash` at schema v3 grandfathered
  "classic" (references reused, zero regen — the schema-versioning payoff).
  Unit tests are BITWISE vs the composed f32-sdpa reference (all cases incl.
  strided views, ring k_off, unaligned tails, block-skip exactness). Gates:
  ALL SIX PASS as shipped default (mm 0.996568, Δnll 0.004305 — digit-exact
  the sdpa-f32-only matrix column, confirming value-identity); flash+tensor
  matrix ALL SIX PASS (Δnll 0.003426 = combined-config column). Review round
  (Claude + Codex + GLM + DeepSeek): zero correctness findings. LPM bench:
  prefill 262.1→264.0 @925, 255.6→268.8 @4k (T² attention term gone — 4k now
  faster per-token than 925), decode 19.2 (untouched); with
  LAGUNA_ATTN_MM_TENSOR: 301.5 @925 / 303.3 @4k (0.85x/0.87x fork, from
  0.74x). The fork's bench numbers DO run flash attention (AUTO-resolved on,
  kernel_flash_attn_ext_f16_dk128_dv128) — this closed a real structural gap.
  - [x] **Tensor projections as DEFAULT — DONE 2026-07-23** (owner-approved).
    The opt-in `LAGUNA_ATTN_MM_TENSOR` is replaced by the opt-out kill-switch
    `LAGUNA_ATTN_MM_CLASSIC` (repo convention); mm/decode/ppl candidate
    expectation flips to `attn_mm: "tensor"`
    (LAGUNA_PARITY_EXPECT_ATTN_MM-overridable), strict keeps its hardcoded
    "f32-bypass" pin (strict runs LAGUNA_ATTN_F32, which bypasses the f16
    library entirely — a "classic" expectation would never match a real
    strict run) plus LAGUNA_ATTN_MM_CLASSIC=1 pinned in the strict/reference
    envs as belt-and-suspenders; gate flag `--attn-mm-tensor` →
    `--attn-mm-classic`; isReferenceDump now fail-fasts on attn_mm (closed
    that ledger sub-item). Gate as shipped default: ALL SIX PASS, digit-exact
    the experiment matrix (mm 0.996564, Δnll 0.003426, decode 63/63/60 with
    0 unexcused). Shipped-default bench (LPM): prefill 304.7 @925 /
    303.8 @4k (0.86x/0.87x fork), decode 18.8 (±5% band, still ahead).
  - [ ] **Decode vec-kernel port (flash phase 2)** — vendor the
    sdpa_vector/2-pass kernels with the same f32-boundary treatment: removes
    the seq==1 q/out casts (~1% decode), insulates decode attention from
    candle, and gives long-context decode the split-K path on our terms.
    Low priority: decode is mask-free and already ahead of the fork.
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
  PROTOTYPED 2026-07-23 (src/ops/f16_t_proto.metal + the `proto_bench` module in
  f16.rs, both #[ignore]d — NOT production-wired). Two Metal-4 `matmul2d` ports of
  the gemm, mirroring the mm_id `_t`/`_t_hp` precedent: variant B (`_t`, half
  operand tiles) and variant A (`_t_hp`, float operand tiles). Findings on the four
  attention projection shapes at seq=512:
  - **B (half tiles) is the clear win: ~1.8–2.0x over the shipped classic kernel**
    (burst-min ms: SWA-q 9216 2.80 vs 5.17; FULL-q 6144 2.01 vs 3.64; k/v 1024 0.46
    vs 0.81; o_proj 3072 2.69 vs 3.82). Cost: it rounds the f32 activation to f16,
    so rel_l2 ~1.85e-4 / max_abs ≤1.4e-2 vs classic — the fork's own prefill
    precision, i.e. the **mm** parity tier, NOT strict. This regresses attention
    prefill off its current strict-clean status (float-tile classic is
    MMA-bit-identical to the f32 reference); acceptable only because the prefill
    full-logit gate is ALREADY the mm tier (mm_id drifts the same way).
  - **A (float tiles) is not worth porting**: bit-IDENTICAL to classic (rel_l2 0.0)
    but ~break-even on speed (no tensor-core throughput gain from float operands) —
    exactly the mm_id `_t_hp` precedent ("compiles but slower than f16 tiles").
  PORTED then REVERTED to opt-in 2026-07-23. Variant B graduated to production
  (src/ops/f16_t.metal, own lazy-compiled Metal-4 library; numerics test + timing
  bench in f16.rs) and was shipped as the mm-branch default. The parity gate then
  REJECTED it as default on the decode tier: its f16 activation staging flips a
  0.6001-margin reference decode decision at code-short step 29 (cand 33586 vs ref
  785), a flip the FORK itself does NOT make (fork argmax 785, margin 0.32; 33586
  is 4th in the fork's top-10) — so our drift exceeds the fork envelope. (mm passed
  but headroom shrank to 8e-4 / cos 0.995842; ppl consumed 79% of its bound.) The
  tensor kernel is kept as opt-in `LAGUNA_ATTN_MM_TENSOR`; the classic simdgroup
  gemm stays the default.
  MIXED-OPERAND FOLLOW-UP PROBED AND CLOSED 2026-07-23: `matmul2d` with f16
  weight tiles × f32 activation tiles COMPILES (the SDK header's dtype table
  lists float×half→float; the "input types must match" static_assert that bit
  llama.cpp/whisper.cpp/ollama applies to mismatched 16-bit pairs, not this
  combo) and is bit-IDENTICAL to the classic kernel (rel_l2 exactly 0.0 on all
  six shapes) — but it is also classic-SPEED (~1.02-1.07x): a float operand
  forfeits f16 tensor-core issue and lowers to the same K-ascending f32
  simdgroup MMA sequence classic hand-codes (which is exactly why the bits
  match). The tensor path's 2x is inseparable from half-operand staging, so
  there is no rounding-free tensor speedup; classic stays default. Probe kept
  as evidence: `src/ops/f16_t_mixed.metal` + `f16_tensor_mixed_matches_classic`
  / `f16_tensor_mixed_vs_classic_timing` in f16.rs (test-only, never
  production-selected). Operand-order trap for future matmul2d work: f16_t.metal
  passes `mm.run(sB, sA, ..)` with a `<tA, tB, float>` destination template —
  harmless with same-type tiles, wrong with distinct types; the mixed kernel's
  corrected order is the reference.
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
