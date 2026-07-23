# Engineering log

What was tried, what worked, what didn't, and why. Append new entries AT THE
TOP (reverse-chronological). TODO.md is the forward ledger; this is the history.
Dates marked `~` are reconstructed from git/TODO records, not contemporaneous.

## 2026-07-23 — mmap + no-copy load, part 2: the ~10% sustained-prefill deficit was GPU residency; candle vendored + patched (queue-attached residency set), gap closed

**The deficit.** The freshly-shipped mmap loader benched clean on decode and
short prefill but lost ~10% at sustained 4k prefill (275-277 vs classic
305.7, reproduced 4x; the 925 bench rides the LPM burst window and hid it).
Diagnosis was a full elimination chain: (1) mlock the mapping — no change
(host page wiring is not GPU residency); (2) "copy attention planes to
Private storage" — no change, and the review round later exposed WHY: at
this candle rev Metal's `Storage::try_clone` is a shallow Arc clone, so
`Tensor::copy()` copies nothing — the experiment was a no-op (the shipped
code now documents this trap instead of using it); (3) GPU-blit all 70GB
into Private buffers — thrashed the box (70GB wired + 70GB page cache >
128GB; killed, removed); (4) the decisive control: the FORK pays no mmap
tax at all (llama-bench pp4096: 345.2 mmap vs 341.1 copied), so no-copy
file-backed memory streams at full rate when set up right. The one
structural difference left: candle attaches its residency set to its
command queue (`Commands::new` → `addResidencySet`) and registers every
pool buffer in it — permanently GPU-resident, no per-command-buffer
residency bookkeeping — while foreign no-copy views can't join it (the
set and the queue are both pub(crate), no objc side-door exists; encoders
don't expose their command buffer).

**The fix: candle is now VENDORED at vendor/candle** (same pinned rev
27f20fea, wired via `[patch."https://github.com/huggingface/candle.git"]`
for all five crates that git source provides — patching a subset would
build two copies of the shared types) with a ~50-line patch, upstream-PR
candidate (vendor/candle is a git clone of huggingface/candle, branch
`metal-residency-set-registration` cut at the pin, patch uncommitted for
Orvar's authorship): candle-core
`MetalDevice::{register,unregister}_external_buffers` (batch, one
residency-set commit each) on new candle-metal-kernels
`ResidencySet::{insert_batch, remove_batch}`. The API went through two
shapes: first a raw `residency_set()` accessor + staged adds — measured
correct (4k recovered) but review-dinged (exposes remove() on pool
buffers; staged adds get flushed incidentally by candle's own pool-alloc
commits, so "one commit at the end" was the guarantee, not the
mechanism) — then the semantic register/unregister pair (batching still
essential: per-view synchronous commits measured ~7s of load, 4.0s →
11.2s). MmapSource collects views, batch-registers once after load, and
in Drop quiesces (wait_until_completed — an in-flight command buffer
retains its bound MTLBuffers but that does NOT keep the mmap'd pages
alive) then unregisters everything it registered: the set RETAINS its
allocations, so without the unregister a dumped model would pin every
view's GPU mapping forever. Load→drop cycles (a serve-then-unload
server) are leak-free; drop ordering is structural (every view holder
owns an Arc<MmapSource>, so its Drop runs last). Residency is perf-only
(setBuffer-bound buffers are made resident per command buffer
regardless) — the pre-patch builds were bit-correct, just slower.

**Results (same-state A/B, LPM, the honest comparison after a day of
thermal drift — classic itself read 290 this round):** 4k sustained: mmap
293.1 vs classic 290.1 — DEAD EVEN (was -10%; 294.3 re-confirmed on the
final register/unregister API); decode 18.3 vs 18.4 — even; load 4.2s
(vs 11.2 with per-view commits, 4.0 with no residency, ~20 classic).
All six parity gates pass at the exact anchors on the
vendored build. Review round (Claude + GLM): the shallow-copy discovery
above (fix: deleted the no-op copy + documented the trap), comment
honesty on the commit mechanism (fixed), an upstream-PR doc note on the
accessor (added), [patch] completeness verified against Cargo.lock, the
supposed 5.5GB residency leak dissolved (the "transient" views ARE the
permanent tensor storage — registering them is required, not a leak).

## 2026-07-23 — mmap + no-copy load: warm load 20s → 4s, fork parity, no repack file, no candle patch

**The territory map killed both halves of the planned design.** The ledger
item assumed (a) a one-time repack cache file (because "no-copy can't
reorder" the expert stacks) and (b) a candle patch to wrap foreign buffers
as QTensors. Neither exists in the shipped version: the GGUF already stores
every expert stack expert-major-contiguous — `expert_stack` never did a
re-layout, it uploaded the bytes verbatim — and per-tensor file alignment
is irrelevant because only the BUFFER base needs page alignment: mmap the
whole file once (memmap2 was already a dep), then give each aliased tensor
its own `newBufferWithBytesNoCopy` view over the page-floored range, with
the sub-page remainder carried as a `base_off` byte offset into the
dispatch (GGUF's 32B data alignment keeps it vector-aligned). Per-tensor
views also sidestep the fork's overlapping-giant-view maxBufferLength
scheme entirely. And no QTensor wrap is needed: the ~70GB of expert stacks
feed our vendored kernels, which take raw (Buffer, byte-offset) —
`ExpertStack.qtensor` became `Option`, None on the mmap path — and the
attention f16 planes alias through the pub `MetalStorage::new` +
`Tensor::from_storage` + narrow route (GGUF F16 bytes ARE the dense plane;
the old dequantize_f16 f16→f32→f16 round-trip was exact, so aliasing is
bit-identical; the f16 dispatch already honored layout start_offset).
Everything needing a real QTensor (lm_head, layer-0, shared experts,
Reference oracle) or too small to matter keeps the copying path (~1.5GB).

**Deps decision.** `newBufferWithBytesNoCopy` needs objc2-metal, which
CLAUDE.md had banned as a mismatch trap. Neutralized instead of obeyed:
the three objc2 crates are `=`-exact-pinned to the versions candle's rev
resolves, so cargo unifies them to one copy — the trap only existed for
version ranges. Pin discipline documented in Cargo.toml and CLAUDE.md.

**Results.** Warm load 20.0s → 4.0s (same-page-cache A/B on this binary;
the historical "~30s" included colder cache); end-to-end vs the fork's
llama-cli on identical tiny runs: 5.05s vs 5.75s — load parity reached.
All six parity gates PASS at the exact anchors (same bytes ⇒ same logits;
references reused). Smoke generation byte-identical vs
`LAGUNA_LOAD_CLASSIC` (the kill-switch that restores the copying loader).
Process RSS collapses from ~75GB to a few GB: aliased weights are clean
file-backed page-cache pages (Activity Monitor charges them to "Cached
Files"), which also means memory pressure can evict them → re-faults on a
later forward. A standalone ResidencySet keeps the views
residency-requested (candle's queue-attached set is pub(crate)-unreachable
— revisit if the pin moves). Bitwise unit tests cover the f16 alias (gemv
+ tensor-mm branches), expert-stack mmap-vs-classic through mv_id and
mm_id at nonzero base_off, and the LAGUNA_MV_CLASSIC candle-baked encode
site. Review round (Claude + GLM, per new two-reviewer policy): clean;
one hardening applied (16B-alignment ensure on the expert alias path).

## 2026-07-23 — glue phase 2 shipped (last standalone casts folded); encoder takeover REFUTED before build

**Glue phase 2 (revised scope).** The territory map showed the ledger item
predated flash and had half-died: flash's f32 output already killed the
prefill post-sdpa cast, and the v cast was already folded into
permute_01_f16. What actually remained: ONE standalone traffic pass on the
whole default prefill path (`cast_f16(k)` after rope) and three small decode
casts. Landed: `kernel_rope_neox` templated on the store type (rope math
stays f32, one RTNE rounding at the f16 store — bit-identical to the old
rope→cast chain, passthrough dims included; `Rope::apply_dt` picks per-call
dtypes) and `kernel_attn_gate` templated on the attn input type (f16 widened
in-kernel, exact). Policy: k rope-stores f16 always (feeds the f16 cache
directly); q rope-stores f16 only at seq==1 — flash requires f32 q at
prefill, and LAGUNA_SDPA_F32's sdpa takes f32 q so the experiment branch
keeps f32 too; decode's sdpa f16 output goes straight into the f16-input
gate. Decode v cast stays (never roped, no fold target). No new
kill-switch/provenance: bit-identical by construction under the existing
LAGUNA_ATTN_GLUE_CLASSIC umbrella. Two new bitwise unit tests; all six
gates PASS digit-exact at the pre-change anchors with references reused
(second consecutive zero-regen change — the schema-version machinery and
the bit-identity playbook are compounding). Bench (LPM): 304.7 → 304.0
@925 (flat), 303.8 → 311.9 @4k (+2.7% — the folded k-cast pass scales with
seq), decode 18.8 → 19.2 (3 dispatches/layer gone; ±5% band). Review round
(Claude + Codex + GLM + DeepSeek): zero correctness findings; hardening
applied (dtype guard on the rope chain fallback so both routes enforce the
F32/F16 contract; q-decode shape added to the bitwise test).

**Encoder takeover: premise refuted, item closed.** The ledger said
"re-measure before investing" — it was right to. A static simulation of
candle's hazard tracker (semantics verified line-by-line at the pin:
RAW/WAR/WAW keyed on whole buffer pointers vs the accumulated window;
barrier = global MTLBarrierScope::Buffers that resets the window) over the
full 51-dispatch default-prefill layer: 37 barriers, ALL true
producer→consumer dependencies — the layer is a near-linear RAW chain.
Whole-buffer keying adds ~0-3 false barriers worst case (<1% of GPU time)
because candle's pool only recycles dropped-producer buffers
(strong_count==1) and weights never enter the write-set. The independent
fat ops ({g,q,k,v} projections, mm_id gate/up, combine+shared-expert)
already share barrier-free windows. The fork's mechanism re-read: its
barrier is equally coarse (full mem_ranges reset); its wins — byte-range
precision and reorder — have nothing to bite on here, and at batch=1 it
encodes on 1-2 threads. Empirical bracket sealed it:
`CANDLE_METAL_COMPUTE_PER_BUFFER=1` (one encoder per dispatch = maximal
serialization + per-dispatch fences) left prefill-925 IDENTICAL at 304.0
tok/s. If 50x more fences cost nothing, no scheduling improvement can gain
anything. Verdict: prefill is fat-kernel compute-bound; the remaining
0.86x/0.90x fork gap lives INSIDE kernels. Byte-range patch, own-encoder
takeover, issue-order reorder, and the demoted routing-glue fusion are all
dead ends — closed with the analysis in the ledger, replaced by a
per-kernel-efficiency profiling item (compare mm_id tile gemm, f16
projections, flash, rms_norm against ggml's per-op times, same power
mode).

## 2026-07-23 — flash attention shipped: vendored steel kernel, in-kernel SWA masking, T² term gone; tensor projections pass the full gate matrix (+14% prefill on top)

**What landed.** Prefill (seq>1) attention no longer calls candle's sdpa: a
vendored, modified copy of candle's MLX steel attention kernel
(`src/ops/flash.metal` + `src/ops/flash.rs`, own runtime-compiled library,
no Metal-4 dep, MIT/MLX attribution kept) runs the whole thing. Decode
(seq==1) untouched. Kill-switch `LAGUNA_FLASH_CLASSIC`; provenance field
`flash` on schema v3, grandfathered "classic" — cached references stayed
valid, zero regen (the schema-versioning machinery built yesterday paid for
itself on its first field addition).

**The mapping surprise that set the design.** Three-agent territory map
(fork's flash_attn_ext, our sdpa boundary, candle's steel kernels) found:
(1) the fork's bench numbers we chase DO run flash attention — AUTO
resolves on for Metal, kernel_flash_attn_ext_f16_dk128_dv128, GQA 6/9
unconstrained; (2) candle's steel kernel ALREADY accumulates QK/softmax/O
in f32 (MLX lineage; it even upcasts f16 smem operands to f32 before the
simdgroup MMA — stronger than ggml, which runs half-operand MMA). So the
"f32 accumulation" the drift-attribution experiment proved we needed was
never missing from the kernel — the differential noise was the BOUNDARY:
our q f32→f16 cast in, f16 output store + widen out, and the materialized
mask. The port therefore targets exactly those: Q device-f32 → f32 smem
(16.9KB, total smem 23.0KB < 32KB), K/V read device-f16 via the
head-strided cache views (never packed/widened — no bandwidth tax), f32 O
store straight into the `[n_head, seq, 128]` f32 contract of the fused
gate. Value-identical to the LAGUNA_SDPA_F32 experiment config by
construction — and the unit tests came out BITWISE vs the composed f32-sdpa
reference on every case (full/SWA, pos 0/>0, ring k_off, unaligned seq%32 /
K%16 tails, seq=2, real 48/8 and 72/8 head geometry, strided-vs-packed,
skip-vs-noskip).

**Mask deleted, not fused.** The old path materialized additive f16 masks
`[1, heads, T, K]` (at 4k: ~1.5GB full + ~2.3GB SWA per forward, re-read by
every layer) and did full T×T sdpa compute on SWA layers with ~7/8 of it
masked. The vendored kernel takes `q_off/k_off/window` args instead: key j
visible to query i iff `(j+k_off) <= (i+q_off) && (i+q_off)-(j+k_off) <
window` (sentinel = full), bit-equivalent to the 0/-inf mask because
exp(-inf) terms contribute exact zeros — plus block-level skip
(kb_start/kb_lim) so fully-invisible KV blocks aren't even loaded
(bit-identity of skipping proven by a dedicated test with a test-only
disable_skip arg). model.rs skips PrefillMask entirely on the flash path
(mask-hoisting subsumed). This beats the fork's own scheme (pad kernel +
mask-classifier kernel + f16 mask tensor). k_off = pos − min(pos, window)
lines the rule up with attn_mask_for's ring-reordered column semantics;
four reviewers verified the correspondence independently.

**Gates (shipped default, flash on):** ALL SIX PASS — strict 0.999057
(classic path pinned, untouched), mm cos 0.996568, decode 63/64+1exc /
63/64+1exc / 62/64+2exc (0 unexcused), Δnll 0.004305. mm and Δnll are
DIGIT-EXACT the sdpa-f32-only column of yesterday's matrix — model-level
confirmation of value-identity. **Flash+tensor matrix
(--attn-mm-tensor): ALL SIX PASS**, Δnll 0.003426 = yesterday's
combined-config column; the tensor-projections rejection is resolved.

**Perf (LPM, bench.ts protocol):** flash-only prefill 262.1→264.0 @925
(sdpa was a small share at 925) and 255.6→**268.8** @4k (16.4s→14.9s — the
T² attention term left the profile; 4k is now faster per-token than 925).
Decode 19.2 (untouched path, noise). With tensor projections:
**301.5 @925 / 303.3 @4k** — tensor now helps at 4k too (flash exposed the
gemm it was previously hidden behind). vs fork 354/348: 0.74x → 0.85x/0.87x
in one lever.

**Tensor projections flipped to DEFAULT (same day, owner-approved).**
`LAGUNA_ATTN_MM_TENSOR` (opt-in) replaced by `LAGUNA_ATTN_MM_CLASSIC`
(opt-out kill-switch, repo convention); candidate expectation `attn_mm:
"tensor"` for mm/decode/ppl, strict keeps its hardcoded "f32-bypass" pin
(strict runs under LAGUNA_ATTN_F32, which bypasses the f16 projection
library — "classic" would never match a real strict run) with
LAGUNA_ATTN_MM_CLASSIC=1 additionally pinned in strict/reference envs;
gate A/B flag `--attn-mm-classic`; isReferenceDump now fail-fasts on
attn_mm (closing that hardening ledger sub-item). Revalidated as the
shipped default, not just the experiment config: ALL SIX PASS digit-exact
the matrix run (mm 0.996564, Δnll 0.003426, decode 63/63/60 + 0
unexcused), shipped-default bench 304.7 @925 / 303.8 @4k / decode 18.8
(±5% LPM band, still ahead of the fork's 18.5). Day summary: prefill
0.74x → 0.86x/0.87x of the fork; the remaining prefill levers are encoder
takeover and glue phase 2.

**Review round:** Claude reviewer + Codex + GLM + DeepSeek, all four clean
on kernel math, block-skip bounds, smem aliasing/barriers, ABI, stride
handling, and provenance grandfathering — zero correctness findings; one
cosmetic fix applied (comment documenting the intentional Ks/Vs smem
aliasing). Deferred to ledger: decode vec-kernel port (flash phase 2),
tensor-default flip.

## 2026-07-23 — drift attribution: f16 sdpa amplifies staging noise (CONFIRMED); tensor projections numerically unlocked, perf-blocked on flash attention

**The experiment.** Question from the tensor-gemm rejection: WHERE does the
fused pipeline spend its decode envelope, and can reallocating it let the 2x
tensor projections pass? Hypothesis: our sdpa-in-f16 (candidate AND reference
both — common-mode, so not a direct drift source) AMPLIFIES upstream f16
staging noise nonlinearly, and f32 accumulation would damp it. Machinery
built for it (all env-gated, no default changes): LAGUNA_SDPA_F32 f32 sdpa
path (candle's native f32 Metal sdpa kernels — steel_attention_float32 +
sdpa_vector_float, head_dim 128 supported; cached f16 k/v widened exactly
per call), provenance field `sdpa` on the new schema v2, gate flags
`--sdpa-f32`/`--attn-mm-tensor` with LAGUNA_PARITY_EXPECT_* test overrides
(no more editing the test for experiments), and provenance SCHEMA VERSIONING
(src/parity_schema.rs): missing fields grandfather to their classic-era
value iff the dump's schema_version predates the field — field additions no
longer invalidate references (the ~40-min regen tax, paid 3x, is dead;
`committed_ppl_reference_fixture_stays_valid` proves it against the real
fixture).

**Results (three-config gate matrix, candidate-only cycles):**
- tensor-only control: rejection reproduced EXACTLY on the post-glue binary
  (mm 0.995842, Δnll 0.004732, step-29 flip) — glue bit-identity confirmed
  yet again, and yesterday's adjudication is stable.
- sdpa-f32 alone: all pass, but anchors move slightly AWAY from the
  reference (mm 0.996987→0.996568, Δnll 0.001937→0.004305). Exactly the
  predicted signature of changing a common-mode op on one side only: pure
  differential noise, no benefit alone.
- tensor + sdpa-f32: ALL SIX PASS and step 29 AGREES OUTRIGHT (63/64 + the
  perennial step-0 near-tie; Δnll 0.003426 < either single change). The
  interaction term is the finding: precise accumulation damps the staging
  noise rather than adding its own — amplification confirmed.

**Perf verdict: not shippable as-is.** LPM benches of the combined config:
prefill 235.2 @925 / 204.1 @4k (shipped: 262/256), decode 15.2 (shipped:
19.1). f32 sdpa doubles the O(T²) attention traffic (worst at 4k) and the
experiment path re-widens the entire cached KV every forward. The tensor
projection gain cannot outrun that; no configuration of the current pieces
is net-positive.

**Strategic readout.** The gate no longer blocks tensor projections — our
attention precision does, and the fork already told us the answer: its
flash_attn_ext takes f16 KV and accumulates in f32, giving the damping
WITHOUT the bandwidth (f16 inputs, f32 registers). The flash-attention port
is promoted to the top attention lever with dual payoff: removes the
sdpa/mask/cast structural overhead AND unlocks the 2x projections as
default. Defaults unchanged today; the experiment machinery stays for the
flash+tensor rerun.

## 2026-07-23 — attention glue fusion SHIPPED: prefill 238→262 @925, decode 18.2→19.1 (passes the fork in LPM)

**What shipped.** Three fused Metal kernels replacing ~275 ms/forward of
candle-op glue traffic in the attention block, each bit-identical by
construction to the chain it replaces (the combine playbook: exact per-op
rounding replication, f32::to_bits equality tests vs the live candle chain):
`kernel_attn_gate` (the ~10-dispatch softplus chain + broadcast_mul → one
pass; candle's affine is fma, exp/log plain fast-math intrinsics, fp
contract/reassociate off), `kernel_permute_cast_*` (transpose+contiguous+
f16-cast collapsed to one pass each), `kernel_rope_neox` (partial rotary
internal to the kernel — no narrow/contiguous/cat; body is candle's rope
template verbatim, deliberately compiled WITHOUT fp pragmas in its own
library to inherit candle's contraction decisions). Kill-switch
`LAGUNA_ATTN_GLUE_CLASSIC`; provenance `attn_glue` enforced per tier (fused
for mm/decode/ppl candidates, classic pinned for strict + references).
Dispatches per attention layer: prefill 26→10 (full) / 20→10 (SWA), decode
22→7 — the fused gate/rope/cast run at seq==1 too, which subsumed the old
P1(b) decode-glue item and bought decode 18.2 → 19.1 tok/s: LPM decode now
PASSES the fork (18.5). Prefill: 238.5 → 262.1 @925, 235.8 → 255.6 @4k
(0.74x fork LPM, from 0.49x at yesterday's open).

**Gate.** All six grades PASS at the exact known-good anchors (strict
0.999057, mm 0.996987, decode 62/64+2exc / 64/64 / 59/64+5exc, Δnll
0.001937) — bit-identity confirmed at model level. References were
regenerated only because the new provenance field hard-fails on dumps that
predate it (the enforcement working as designed, but the 3rd ~40-min regen
tax paid for a field addition — schema-version grandfathering proposed in
TODO to make it the last). reference-ppl mean_nll bit-identical across the
regen (4th time), which is live proof the classic fallback reproduces the
pre-fusion chain exactly.

**Review round (Claude + Codex + GLM + DeepSeek) and the math-mode arc.**
No live bugs; the substantive find (DeepSeek, corroborated) was that all our
vendored libraries compile with nil MTLCompileOptions while candle pins
MathMode::Fast explicitly (private factory, unreachable at the pinned rev —
consistent with the combine-arc finding). Research vs the MSL v4.1 spec:
nil options resolve to Fast/Fast by documented default (fastMathEnabled
bridging), so behavior already matched — the exposure was only a future OS
default flip. Hardening applied: `#pragma METAL fp math_mode(fast)` at file
scope in all NINE vendored .metal sources (spec §1.6; bad options in the
recognized `METAL fp` namespace hard-error, so compiling proves it honored —
the reassociate(off) precedent), plus a load-time hard error if
CANDLE_METAL_ENABLE_FAST_MATH is set falsy (candle would compile
Relaxed/Precise while our libs stay pinned Fast — mixed modes silently void
every bitwise contract). Checked the FORK's compile too
(ggml-metal-device.m:228): `[MTLCompileOptions new]` defaults, i.e. also
Fast — refuting the hypothesis that fast-math explains the tensor-gemm
envelope difference (it is common-mode across candidate, reference, and
fork; the open per-op question — where our envelope IS spent, sdpa-in-f16
being the prime suspect — is a TODO item). Other hardening from the round:
overflow-checked size products (checked_elems), same-device checks
(Device::same_device exists at the pinned rev), offset-view bitwise test,
fail-fast attn_glue in isReferenceDump. Deferred sweeps + phase-2 cast
folding: TODO ledger.

**Tooling.** `scripts/bench.ts` checked in (the guarded bench protocol —
pgrep model-binary guard, pmset verification at start AND end with
mode-stamped results, committed bench prompts in
tests/fixtures/bench-prompts/, --gate chaining). Writing it immediately
caught a real bug: parity-gate's process guard pattern matched a reviewer's
`git diff -- src/bin/logits-dump.rs` command line as a "model process";
both guards now require actual binary-path signatures.

## 2026-07-23 — mixed-operand matmul2d probed and CLOSED; encoder-takeover premise corrected; glue fusion begins

**Mixed-operand probe (the tensor-gemm escape hatch): dead, conclusively.**
The hope was `matmul2d` with f16 weight tiles × f32 activation tiles — tensor
speed with zero activation rounding, sidestepping the decode-gate rejection.
Two agents ran in parallel and *disagreed*: web research found four projects
(llama.cpp #17986, whisper.cpp #3601, ollama ×2) hitting an "Input types must
match cooperative tensor types" static_assert and concluded mixed operands are
compiler-rejected; the empirical probe on THIS toolchain compiled and ran fine
(`src/ops/f16_t_mixed.metal` — the SDK header's dtype table lists
float×half→float; the assert those projects hit applies to mismatched 16-bit
pairs). Empirics beat literature — but the answer is still no: the mixed
kernel is BIT-IDENTICAL to classic (rel_l2 exactly 0.0, all six shapes) and
classic-SPEED (~1.02-1.07x, vs 1.6-2x for half tiles). A float operand
forfeits f16 tensor-core issue and lowers to the same K-ascending f32
simdgroup MMA sequence classic hand-codes — the bit-identity and the missing
speedup are the same fact. There is no rounding-free tensor speedup; the 2x
is inseparable from f16 activation staging, which the decode envelope
rejects. Projections stay classic; probe kernel + tests kept as evidence
(test-only, never production-selected). Trap for future matmul2d work:
f16_t.metal's `mm.run(sB, sA, ..)` vs `<tA, tB, float>` destination template
order is accidental-harmless with same-type tiles and WRONG with distinct
types; the mixed kernel carries the corrected order.

**Encoder-takeover premise corrected (territory map).** TODO's claim that
"candle's eager op-per-encoder submission serializes all of it" is false at
our pinned rev: candle shares ONE Concurrent-dispatch encoder across ~50
dispatches with whole-buffer-pointer hazard tracking, and our custom ops
already ride it. The fork's real edge is byte-RANGE hazard granularity (candle
emits a full memoryBarrier that resets the whole concurrent set on any
pointer-level hazard, so a dependency chain degrades to serial) plus graph
reorder into concurrent windows. Realizable prize = range-tracking + reorder
only; details and risks (buffer-pool pointer aliasing, prev_ce_outputs fence
protocol, vendoring prerequisite) recorded in the TODO ledger item. The
strategically load-bearing corollary: the ~275 ms attention glue is a TRAFFIC
problem — concurrency would only overlap those copies, fusion removes them —
so 3(c) glue fusion sequences BEFORE any takeover, which then schedules
fewer, fatter kernels.

## 2026-07-23 — tensor attention gemm: 2x in isolation, REJECTED as default by the decode gate; kept as opt-in

**Arc.** The attention sub-budget put the prefill projections (classic
simdgroup f16 gemm) at 585 ms/forward — the biggest category. Prototyped two
Metal-4 `matmul2d` ports on the production shapes: float-operand tiles
(`_t_hp`-analog) were BIT-IDENTICAL to classic but zero speedup — the mm_id
precedent repeating exactly; half-operand tiles (`_t`-analog, fork-faithful
f16-staged activations) were ~2x on every shape at ~1.85e-4 rel — the fork's
own prefill numerics class. Orvar approved shipping the f16-staged variant as
default, gate to sign off. Ported with full plumbing (opt-in/kill-switch env,
`attn_mm` provenance enforced per tier, lazy-compiled separate Metal-4
library; decode gemv and LAGUNA_ATTN_F32 untouched).

**Gate verdict: REJECTED.** End-to-end it delivered 267 tok/s @925 (0.76x
fork) and passed 5 of 6 grades — but decode/code-short flipped step 29
(candidate 33586 vs reference 785, ref margin 0.6001, past the 0.5 excuse
line). The fork-calibration data was decisive: the fork ITSELF — f16-staged
prefill and all — agrees with the reference at that step (argmax 785, margin
narrowed 0.60 → 0.32; our token sits 4th in its list). So our tensor drift
exceeds the fork envelope, full stop. Supporting signals: mm cosine fell
0.996987 → 0.995842 (headroom 2e-3 → 8e-4) and Δnll rose 0.0019 → 0.0047
(79% of the bound) — same drift class as mm_id's `_t` but a bigger dose (48
layers × 5 projections), and it compounds visibly.

**Resolution.** Default inverted: classic simdgroup stays the shipped mm
branch; tensor is opt-in `LAGUNA_ATTN_MM_TENSOR`; per-tier `attn_mm`
enforcement pins candidates to "classic". Confirmation gate: all six back at
the known-good anchors (mm 0.996987, Δnll 0.001937 — bit-identical era).
Kernel + numerics test + bench kept for the follow-up: probe `matmul2d` with
MIXED operands (f16 weight tiles × f32 activation tiles — never tried in the
mm_id work either); if the API takes it, that's tensor speed with zero
activation rounding. Lessons: (1) per-matmul numerics class does NOT predict
the model-level envelope — the dose matters; (2) the layered gate design
earned its keep: cosine and ppl both PASSED while the fork-calibrated greedy
tier caught the drift; (3) an isolation 2x that costs the correctness margin
is not a win — the gate is the definition of done, and "the fork does f16
staging too" is not a license when the fork's own outputs disagree.

## 2026-07-23 — attention mask hoisting SHIPPED: prefill 211 → 228 @925 / 184 → 237 @4k, byte-identical

**Context.** The attention sub-budget (new `prefill_attn_*` benches; parts
sum-check against the chain bench within ~3%) split the 23.6/30.1 ms attention
block: projections 9.2/13.2 (42% of attention per forward), cache+mask
5.9/9.5, transposes 2.3/2.5, rope 2.7 FULL vs 1.2 SWA (partial-rotary
narrow/cat tax), sdpa core 1.4/2.0, gate 1.4/1.8, qk-norm 0.4. A follow-up
split showed the cache+mask category is 94-97% MASK: per-layer host mask
build + upload + the [1,n_head,seq,k_seq] f16 broadcast materialization
(~415 ms/forward); the KV cache append is ~0.35 ms.

**Change.** A read-only audit PROVED the mask is a pure function of (kind,
pos, seq_len, window) — `LayerCache::attn_mask` never reads per-layer state
(not even SWA ring fill), so one forward has exactly TWO distinct masks.
Hoisted: `attn_mask_for`/`MaskKind` (kv_cache.rs), `PrefillMask { raw, sdpa }`
built once per kind per chunk in run_stack, passed into AttnBlock::forward
(attention.rs no longer materializes per layer). Decode (seq==1, no mask)
untouched. 48 builds/forward → 2 (× chunks). Byte-identical by construction —
same values, same tensors, same kernels — so no kill-switch, no provenance.

**Result.** Gate: all six grades at the exact anchors (references reused).
Prefill 211 → 228.0 @925, 184 → 236.7 @4k — the 4k case gains most because
chunked prefill multiplied the per-layer waste (8 chunks × 48 masks → 8 × 2,
with masks growing as pos advances). vs fork LPM: 0.64-0.68x. Day cumulative:
174 → 228 @925 (+31%), ~155 → 237 @4k (+53%).

## 2026-07-23 — P2 tranche 1 SHIPPED: fused combine kernel + shared map0 — prefill 174 → 211 tok/s (+21%), bit-identical, all tiers green

**Change.** (a) `src/ops/combine.metal` (+ combine.rs host, own runtime-compiled
library, no Metal-4 dep): the routed-expert combine tail (3 broadcast_muls +
strided sum(1) over the [seq,10,3072]/63MB expert output — measured 14.8
ms/layer, ~9x bandwidth floor) is now ONE kernel reading `down` once.
(b) mm_id's map0 row-map is computed ONCE per MoE block (typed `Map0Scratch`
carrying n_expert/t/top_k, validated per consumer) instead of 3x.
(c) `prefill_mm_id_bench` methodology fix: gate/up/down interleaved in one
warmed loop (gate's old 3.0 ms was DVFS burst fiction; all three are ~8-10 ms
weight-streaming-bound sustained).

**The design bet that paid off: bit-identity BY CONSTRUCTION, not by
tolerance.** The fused kernel replicates candle's `fast_sum_f32_strided`
launch geometry exactly (width-8 threadgroups, ascending per-lane loader
partition, hardware `simd_sum`, lane-0 store) and computes each element
in-register at candle's per-op rounding boundaries (r1=down·col_l2 rounded,
·2^-15 exact, ·w rounded — never folded, no fma). Acceptance test is BITWISE
equality (`f32::to_bits`) vs the live candle chain across seq {1,8,512} ×
n_out {1024,3072} × both variants (rescale / plain) — no tolerance anywhere.
Consequence: logits are bit-identical, every parity tier passes at the exact
prior anchors (strict 0.999057 / mm 0.996987 / decode 62+2,64,59+5 / Δnll
0.001937) with references regenerated, and the kernel is safe under the strict
tier without gate relaxation. `LAGUNA_COMBINE_CLASSIC` kill-switch;
provenance `combine: fused|classic|reference` enforced per tier (missing =
hard fail, same as attn_dtype).

**Review round (Claude + Codex + GLM; DeepSeek skipped, degraded twice
2026-07-22).** No live bugs; five hardening items, two found by all three:
simd_sum lane-drop at top_k ≥ 66 (host bail + test), i32 index overflow at
seq ≈ 70k (host bail + test), typed map0 scratch validation, provenance
labeling for reference dumps + per-tier enforcement, shared-map0 test RAW-
ordering fix. Plus the one real design gap: `#pragma clang fp contract(off)`
does NOT cover fast-math REASSOCIATION of the multiply chain, and the pinned
candle rev cannot express MTLCompileOptions (no re-export, no factory — the
objc2-metal trap again). Resolved source-level: `#pragma clang fp
reassociate(off)` — clang rejects unknown `fp` pragma options, so the library
compiling proves it's honored. Perf spot-check after: 210.1 tok/s, no cost.

**Result.** Prefill 174 → 210.7 tok/s @925 (+21%), 150-160 → 184 @4k (+~19%),
decode unchanged (18.1). vs fork LPM: 0.49x → 0.60x pp512. First bench trio
after the change read 148.9/184/197.9 — the 925 number was thermal-ordering
fiction (ran hottest, right after the gate's ppl prefill); rested rerun 210.7.
Lesson reinforced: single-shot LPM bench numbers are only comparable at
matched thermal history.

## 2026-07-22 — P2 phase-0: prefill budget kills routing-glue fusion (~1%), promotes combine fusion (~23%); fork's Metal edge is scheduling, not fusion

**Context.** P2 was scoped as "fuse MoE route+glue+combine — the prefill gap
is surrounding dispatch overhead." Before writing kernels: map both engines'
per-layer dispatch streams, then measure a category budget (the P1 phase-0
playbook).

**Maps (explore agents, both engines).** Ours: ~69 dispatches/MoE layer at
seq=512, only 11 owned (5 attn projections + 6 mm_id incl. 3 REDUNDANT map0
passes over identical ids); routing is 10 candle dispatches; combine
materializes [512,10,3072]/63MB then reduces; shared expert rides candle
QMatMul. Fork: ~30 dispatches/block — and NO Metal routing fusion exists in
ggml (topk_moe is CUDA/Vulkan/SYCL-only). The fork's 2x edge is SCHEDULING:
one concurrent-dispatch encoder with memory barriers only on real buffer-range
overlap (ggml-metal-ops.cpp mem_ranges :150-210), graph reorder into
concurrent sets (ggml-metal-common.cpp:209/375), multi-threaded command-buffer
encode (context.m:550), plus modest fusions (ADD-chain, RMSNorm+mul, one
dedicated single-dispatch bitonic top-k). Its routing tail is nearly as chatty
as ours — it just overlaps the heavy GEMMs instead of serializing.

**Budget (new `prefill_*` isolation benches, moe.rs/attention.rs, synthetic
weights, seq=512, 100-iter plateau means, LPM).** Per MoE layer: attention
23.6 (full) / 30.1 (SWA) ≈ 46% of the forward; mm_id matmuls 19.8; COMBINE
14.8 (~23% — ~9x its ~1.6 ms bandwidth floor; candle broadcast_mul over
[512,10,3072] takes slow strided paths); silu+L2-rescale 4.1; shared expert
1.8; norms+residuals 0.8; ROUTING GLUE 0.61 (~1%). Sum of parts ≈ whole block
bench ≈ end-to-end within ~10% → execution is fully serial (no overlap), and
the categories are trustworthy. Anomaly flagged: mm_id up 7.9 vs gate 3.0
ms on identical shapes — unexplained, investigate during map0 sharing.

**Verdict — RESCOPED (Orvar agreed).** Routing-glue fusion demoted (~1%;
death-by-dispatch refuted a SECOND time, now for prefill — ten tiny dispatches
are cheap even serialized). Do now: (a) fused combine kernel
`sum_i(down_i × col_l2 × 1/32768 × w_i)`, one read of the 63MB expert output,
f32 accumulation mirroring candle's reduce order, kill-switch + gate; (b)
map0 computed once per block (bit-identical; the fork recomputes 3x too — an
absolute win over it). Expected ~25% prefill. Attention (46%) and the
encoder takeover (concurrency) hold the rest of the 2x; both promoted in
TODO.md's priority order. Lesson: the dispatch-COUNT map alone argued for
routing fusion; only the ms-budget exposed that the expensive dispatches are
the big-tensor broadcast/reduce ops, not the many tiny ones. Count dispatches
to find candidates, measure milliseconds to pick targets.

## 2026-07-22 — same-mode power calibration: decode parity CONFIRMED both modes; fork's historical numbers were LPM

**Context.** Every prior "ours vs fork" comparison carried an asterisk: our
numbers were Low Power Mode sustained, the fork's 361/328/18.1
(pp512/pp4096/tg128) were recorded in an unknown mode. Orvar toggled power
modes so both engines could be benched in each mode (2×2 matrix): our
630-ctx/256-tok sustained bench + fork `llama-bench -p 512,4096 -n 128`.

**Result (pmset-verified modes).**

| | ours LPM | fork LPM | ours full | fork full |
|---|---|---|---|---|
| decode | 18.2 (17.2 re-run) | 18.5 ± 0.5 | 38.6 | 39.2 ± 1.2 |
| prefill short | 174 (156 re-run) | 354 ± 30 | 411-415 | 990 ± 26 |
| prefill 4k | 150-160 | 348 ± 11 | 345 | 793 ± 18 |

- **Fork's historical figures were LPM**: measured fork-LPM 354/348/18.5 vs
  the recorded 361/328/18.1 — same numbers within noise. Our LPM decode-parity
  claim was same-mode all along.
- **Decode parity holds in BOTH modes**: 0.98x fork (18.2/18.5 LPM, 38.6/39.2
  full). The f16-attention result is real, not a mode artifact.
- **Prefill gap is mode-independent**: 0.42-0.49x fork in both modes (slightly
  worse at full power — as expected if part of the overhead is CPU-side
  dispatch that doesn't speed up with GPU clocks). Fork full-power prefill is
  ~990 pp512; that is the size of the P2 prize.
- **LPM clamp re-measured**: ~2.1x decode (ours 18.2→38.6, fork 18.5→39.2),
  ~2.3-2.8x prefill. Phase-0's ~1.7x estimate (below) was low — it was derived
  from within-run burst-vs-plateau decay, not a mode A/B.
- **Thermal ordering matters at full power**: the fork benched ~6% higher on a
  cool chip (41.4/1105/819) than after three of our 75GB runs (39.2/990/793);
  our decode read 30.6 hot vs 38.6 cool. Full-power numbers are ±10% by
  thermal history; LPM ±5% run-to-run (17.2-18.2 same-day). Don't compare
  numbers across thermal states; bench cool-first or report the ordering.

**Traps hit.**
- A "LPM" leg benched at full power: the System Settings Energy Mode toggle is
  PER POWER SOURCE (Battery vs Power Adapter tabs) — a flip on the wrong tab
  silently does nothing while plugged in. Caught because fork pp512 read 1105,
  ABOVE the known full-power 990. Fix, now protocol: every bench chain guards
  and logs `pmset -g | awk '/lowpowermode/{print $2}'` (1 = LPM) at start and
  end — runs self-certify their mode, intent doesn't count.
- llama-bench pp512 reps are ~0.5 s — short enough to partially ride the LPM
  burst window (±30 t/s rep noise at LPM). tg128 (multi-second) can't
  burst-cheat; trust it to arbitrate mode questions.

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
