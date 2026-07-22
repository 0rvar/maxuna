#!/usr/bin/env bun
/**
 * Parity-gate orchestrator: one command runs the full docs/parity.md cycle.
 *
 * It automates the manual runbook (docs/parity.md §3): produce the Reference
 * oracle dumps, produce the Fused candidate dumps, and run the tiered gate
 * (`tests/parity.rs`) over each. Four tiers, each with the fixtures the runbook
 * prescribes:
 *   - strict : full-logit gate, mv_id path (LAGUNA_NO_MM_ID=1), code-short.
 *   - mm     : full-logit gate, default mm_id prefill path, code-short.
 *   - decode : greedy forced-replay gate vs the Reference oracle, all 3 fixtures.
 *   - ppl    : perplexity-delta gate over the frozen corpus (single, no fixture).
 *
 * Model-run hazards this script enforces (CLAUDE.md): ONE 75GB model process at
 * a time — `pgrep -fl "laguna|llama"` before every logits-dump invocation, and
 * every model run is strictly serial (each fully awaited before the next). All
 * model output streams to log files under the parity dir; nothing is ever piped
 * through a pager.
 *
 * Usage:
 *   bun scripts/parity-gate.ts \
 *     [--tiers strict,mm,decode,ppl] \
 *     [--fixtures code-short,text-mixed,long-swa] \
 *     [--regen-ref] [--regen-ppl-ref] [--parity-dir DIR]
 *
 * Defaults: all tiers; fixtures per tier (strict/mm force code-short; decode
 * uses all three; ppl has no fixture axis). Parity dir: $LAGUNA_PARITY_DIR or
 * /tmp/laguna-parity.
 *
 * Reuse: an existing Reference dump is reused iff it carries provenance with
 * moe_impl=="reference" and attn_dtype=="f32" (the pinned oracle environment);
 * --regen-ref forces regeneration. Candidate dumps are
 * ALWAYS regenerated (they are the thing under test). The ppl reference is the
 * frozen fixture tests/fixtures/reference-ppl.json (copied in), regenerated only
 * under --regen-ppl-ref (a committed artifact — the user must review/stage it).
 */

import { openSync, closeSync, mkdirSync, existsSync } from "node:fs";
import { resolve, join, dirname } from "node:path";

const ROOT = resolve(import.meta.dir, "..");
const MODEL = join(ROOT, "models/laguna-s-2.1-Q4_K_M.gguf");
const CORPUS = join(ROOT, "tests/fixtures/ppl-corpus.txt");
const PPL_FIXTURE = join(ROOT, "tests/fixtures/reference-ppl.json");
const FIXTURES_JSON = join(ROOT, "tests/fixtures/parity-prompts.json");
const LOGITS_DUMP = join(ROOT, "target/release/logits-dump");

const ALL_TIERS = ["strict", "mm", "decode", "ppl"] as const;
const ALL_FIXTURES = ["code-short", "text-mixed", "long-swa"] as const;
type Tier = (typeof ALL_TIERS)[number];

// Fixtures each tier grades. strict/mm are calibrated on the full-logit
// code-short prompt only; decode runs all three (long-swa exercises SWA-ring
// wraparound); ppl has no per-fixture axis (single frozen corpus).
const TIER_FIXTURES: Record<Tier, readonly string[]> = {
  strict: ["code-short"],
  mm: ["code-short"],
  decode: ALL_FIXTURES,
  ppl: [],
};

const GREEDY_N = 64; // decode steps per fixture (docs/parity.md §3c default)
const PPL_MAX_CTX = 5120; // corpus (~4386 tok) + BOS exceeds the 4096 default

// -------------------------------------------------------------------- args

interface Opts {
  tiers: Tier[];
  fixtures: string[];
  regenRef: boolean;
  regenPplRef: boolean;
  parityDir: string;
}

function parseArgs(argv: string[]): Opts {
  const flags: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (!t.startsWith("--")) continue;
    const key = t.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) flags[key] = true;
    else { flags[key] = next; i++; }
  }

  const csv = (v: unknown, all: readonly string[], label: string): string[] => {
    if (v === undefined) return [...all];
    const items = String(v).split(",").map((s) => s.trim()).filter(Boolean);
    for (const it of items) {
      if (!all.includes(it)) die(`unknown ${label} ${JSON.stringify(it)} (valid: ${all.join(", ")})`);
    }
    return items;
  };

  return {
    tiers: csv(flags.tiers, ALL_TIERS, "tier") as Tier[],
    fixtures: csv(flags.fixtures, ALL_FIXTURES, "fixture"),
    regenRef: Boolean(flags["regen-ref"]),
    regenPplRef: Boolean(flags["regen-ppl-ref"]),
    parityDir: typeof flags["parity-dir"] === "string"
      ? String(flags["parity-dir"])
      : (process.env.LAGUNA_PARITY_DIR || "/tmp/laguna-parity"),
  };
}

function die(msg: string): never {
  console.error(`parity-gate: ${msg}`);
  process.exit(2);
}

// ------------------------------------------------------------------- shell

/**
 * A clean environment for a model run: strips the mm_id kernel-selection toggles
 * (all PRESENCE-BASED — a stray LAGUNA_MM_ID_F16=0 in the shell would still flip
 * the kernel) so each dump runs the path this script intends, and the parity
 * env vars (the gate sets those explicitly). Per-run additions layer on top.
 */
function baseEnv(): Record<string, string> {
  const e: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (v === undefined) continue;
    if (k === "LAGUNA_NO_MM_ID" || k === "LAGUNA_MV_CLASSIC" || k === "LAGUNA_ATTN_F32" || k === "LAGUNA_COMBINE_CLASSIC" || k.startsWith("LAGUNA_MM_ID")) continue;
    if (k === "LAGUNA_PARITY_DIR" || k === "LAGUNA_PARITY_TIER") continue;
    e[k] = v;
  }
  return e;
}

/**
 * Environment for REFERENCE-ORACLE dumps: the oracle is the maximally precise
 * path, so attention is pinned to f32 compute (`LAGUNA_ATTN_F32=1`) regardless
 * of the shipped f16 default. All cached/committed reference artifacts are
 * f32-attention; regenerating with anything else would silently move the
 * strict tier's 0.999 anchor. The routed-expert combine is likewise pinned to
 * the classic candle chain (`LAGUNA_COMBINE_CLASSIC=1`) so the oracle stays on
 * the exact path its artifacts were blessed with (the fused kernel is
 * bit-identical, so this only anchors provenance). Candidate dumps use baseEnv()
 * and run whatever path their tier is gating.
 */
function referenceEnv(): Record<string, string> {
  return { ...baseEnv(), LAGUNA_ATTN_F32: "1", LAGUNA_COMBINE_CLASSIC: "1" };
}

/** Run a command with stdout+stderr streamed to `logPath` (never a pager). */
async function runToLog(
  cmd: string[],
  logPath: string,
  env: Record<string, string>,
): Promise<number> {
  mkdirSync(dirname(logPath), { recursive: true });
  const fd = openSync(logPath, "w");
  try {
    const proc = Bun.spawn({ cmd, cwd: ROOT, env, stdout: fd, stderr: fd });
    return await proc.exited;
  } finally {
    closeSync(fd);
  }
}

/** Last ~lines of a log file, for surfacing failures without a pager. */
async function tailLog(logPath: string, lines = 30): Promise<string> {
  if (!existsSync(logPath)) return "(no log)";
  const text = await Bun.file(logPath).text();
  return text.trimEnd().split("\n").slice(-lines).join("\n");
}

// --------------------------------------------------------------- preflight

/**
 * Abort if any 75GB model process is already running (concurrent loads OOM the
 * GPU). `pgrep -fl` matches the whole command line, so a stale logits-dump or
 * llama-* is caught by the "laguna|llama" in its model-path argument. Our own
 * runner (and its bun/logits-dump children we are about to await) are excluded
 * by pid and by the script name.
 */
async function preflight(what: string): Promise<void> {
  const proc = Bun.spawn({ cmd: ["pgrep", "-fl", "laguna|llama"], stdout: "pipe", stderr: "ignore" });
  const out = await new Response(proc.stdout).text();
  await proc.exited; // exits 1 when nothing matches — expected, not an error
  const offenders = out
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .filter((l) => {
      const pid = Number.parseInt(l.split(/\s+/)[0], 10);
      if (pid === process.pid) return false;
      // `pgrep -f "laguna|llama"` also matches innocent shells/editors/watchers
      // whose command line merely mentions the laguna repo path or a parity log.
      // The hazard is another 75GB LOAD, so require a real model-process
      // signature: our logits-dump, a llama.cpp binary (`llama-*`), or a process
      // holding a `.gguf` open.
      return /logits-dump|llama-|\.gguf/.test(l);
    });
  if (offenders.length) {
    console.error(`\nparity-gate: refusing to start ${what} — a model process is already running:`);
    for (const o of offenders) console.error(`  ${o}`);
    console.error("Only ONE 75GB model process may run at a time (concurrent loads OOM the GPU).");
    console.error("Kill it (or wait for it to finish) and re-run.");
    process.exit(3);
  }
}

// ---------------------------------------------------------------- fixtures

let fixtureTokensCache: Record<string, string> | null = null;
async function fixtureTokens(id: string): Promise<string> {
  if (!fixtureTokensCache) {
    const j = await Bun.file(FIXTURES_JSON).json();
    fixtureTokensCache = {};
    for (const p of j.prompts) fixtureTokensCache[p.id] = (p.tokens as number[]).join(",");
  }
  const t = fixtureTokensCache[id];
  if (!t) die(`fixture ${JSON.stringify(id)} not found in ${FIXTURES_JSON}`);
  return t;
}

// -------------------------------------------------------------- provenance

/** True iff `path` is a JSON dump whose provenance is a genuine Reference-oracle
 *  run — the only kind the gate accepts as a reference: moe_impl=="reference",
 *  attn_dtype=="f32" (referenceEnv() pins LAGUNA_ATTN_F32=1), AND
 *  combine=="reference" (the Reference runner never dispatches ops::combine). A
 *  dump missing either field predates it and the Rust gate hard-fails on it, so
 *  regenerate here instead of failing mid-run. `kind` (when given) must also
 *  match. Any parse/shape problem returns false (regenerate). */
async function isReferenceDump(path: string, kind?: string): Promise<boolean> {
  if (!existsSync(path)) return false;
  try {
    const j = await Bun.file(path).json();
    if (kind && j.kind !== kind) return false;
    return (
      j?.provenance?.moe_impl === "reference" &&
      j?.provenance?.attn_dtype === "f32" &&
      j?.provenance?.combine === "reference"
    );
  } catch {
    return false;
  }
}

// ----------------------------------------------------------------- timings

const timings: { label: string; seconds: number }[] = [];
async function timed<T>(label: string, fn: () => Promise<T>): Promise<T> {
  const t0 = performance.now();
  const r = await fn();
  const seconds = (performance.now() - t0) / 1000;
  timings.push({ label, seconds });
  return r;
}
const fmtSecs = (s: number) => (s >= 90 ? `${(s / 60).toFixed(1)}m` : `${s.toFixed(1)}s`);

// --------------------------------------------------------------- test binary

/** Build the parity test binary once and return its executable path (parsed
 *  from cargo's JSON artifact stream — never globbed out of target/). */
async function buildParityTestBin(logPath: string): Promise<string> {
  mkdirSync(dirname(logPath), { recursive: true });
  const fd = openSync(logPath, "w");
  let stdout = "";
  try {
    const proc = Bun.spawn({
      cmd: ["cargo", "test", "--release", "--test", "parity", "--no-run", "--message-format=json"],
      cwd: ROOT,
      env: baseEnv(),
      stdout: "pipe",
      stderr: fd,
    });
    stdout = await new Response(proc.stdout).text();
    const code = await proc.exited;
    if (code !== 0) die(`cargo test --no-run failed (exit ${code}); see ${logPath}`);
  } finally {
    closeSync(fd);
  }
  let exe: string | null = null;
  for (const line of stdout.split("\n")) {
    if (!line.startsWith("{")) continue;
    let msg: any;
    try { msg = JSON.parse(line); } catch { continue; }
    if (
      msg.reason === "compiler-artifact" &&
      msg.target?.name === "parity" &&
      Array.isArray(msg.target?.kind) &&
      msg.target.kind.includes("test") &&
      msg.executable
    ) {
      exe = msg.executable; // last one wins — the freshly linked test binary
    }
  }
  if (!exe) die(`could not find the parity test executable in cargo's JSON output (see ${logPath})`);
  return exe;
}

// ------------------------------------------------------------------- gate

/** Run one ignored gate test from the prebuilt binary, streaming to a log. */
async function runGate(
  bin: string,
  testName: string,
  dir: string,
  extraEnv: Record<string, string>,
  logPath: string,
): Promise<number> {
  const env = { ...baseEnv(), LAGUNA_PARITY_DIR: dir, ...extraEnv };
  return runToLog([bin, testName, "--exact", "--ignored", "--nocapture"], logPath, env);
}

// ---------------------------------------------------------------- results

interface Row {
  tier: Tier;
  fixture: string;
  status: "PASS" | "FAIL" | "SKIPPED";
  metric: string;
  logPath?: string;
}
const rows: Row[] = [];

/** Pull a cheap headline metric out of a gate log (best-effort). */
async function gateMetric(tier: Tier, logPath: string): Promise<string> {
  const text = existsSync(logPath) ? await Bun.file(logPath).text() : "";
  if (tier === "strict" || tier === "mm") {
    const cos = text.match(/cosine=([0-9.]+)/);
    const ov = text.match(/top5 overlap=(\d+)\/5/);
    const parts = [];
    if (cos) parts.push(`cos=${cos[1]}`);
    if (ov) parts.push(`top5=${ov[1]}/5`);
    return parts.join(" ") || "(no metric parsed)";
  }
  if (tier === "decode") {
    const m = text.match(/greedy decode gate: (\d+) steps, (\d+) agreements, (\d+) excused near-ties, (\d+) non-excused/);
    if (m) return `${m[2]}/${m[1]} agree, ${m[3]} excused, ${m[4]} mismatch`;
    return "(no metric parsed)";
  }
  // ppl
  const pass = text.match(/\|Δmean_nll\| = ([0-9.]+)/);
  if (pass) return `Δnll=${pass[1]}`;
  const fail = text.match(/mean-NLL delta ([0-9.]+)/);
  if (fail) return `Δnll=${fail[1]}`;
  return "(no metric parsed)";
}

// -------------------------------------------------------------- tier: full-logit

let fullLogitRefReady = false;
/** Ensure the shared full-logit Reference dump (code-short, Reference oracle)
 *  exists at the canonical path; strict and mm both grade against it. */
async function ensureFullLogitRef(parityDir: string, regen: boolean): Promise<string> {
  const refPath = join(parityDir, "reference-full-logit.json");
  if (fullLogitRefReady) return refPath;
  const reuse = !regen && (await isReferenceDump(refPath));
  if (reuse) {
    console.log(`  reusing full-logit reference ${refPath}`);
  } else {
    const log = join(parityDir, "reference-full-logit.log");
    await preflight("full-logit reference dump");
    const tokens = await fixtureTokens("code-short");
    console.log(`  generating full-logit reference (Reference oracle, code-short) -> ${refPath}`);
    const code = await timed("full-logit reference", () =>
      runToLog(
        [LOGITS_DUMP, "--model", MODEL, "--moe-impl", "reference", "--tokens", tokens, "--output", refPath],
        log,
        referenceEnv(),
      ),
    );
    if (code !== 0) die(`full-logit reference dump failed (exit ${code}):\n${await tailLog(log)}`);
  }
  fullLogitRefReady = true;
  return refPath;
}

/** strict or mm: regenerate the candidate full-logit dump on code-short and run
 *  the logit_parity gate under the matching tier. */
async function runFullLogitTier(tier: "strict" | "mm", parityDir: string, regenRef: boolean): Promise<void> {
  const fixture = "code-short";
  const dir = join(parityDir, tier);
  mkdirSync(dir, { recursive: true });

  const refPath = await ensureFullLogitRef(parityDir, regenRef);
  await Bun.write(join(dir, "reference.json"), Bun.file(refPath));

  const candLog = join(dir, "candidate.log");
  const candPath = join(dir, "candidate.json");
  const tokens = await fixtureTokens(fixture);
  // strict gates the CLASSIC fallback path: LAGUNA_NO_MM_ID=1 forces the expert
  // path off mm_id onto the per-token mat-vec, LAGUNA_MV_CLASSIC=1 reverts
  // that mat-vec from the vendored ggml geometry to candle's classic kernels —
  // the f32-accumulation-order the 0.999 baseline was calibrated against (the
  // DEFAULT vendored mv path is gated by the decode + ppl tiers instead; its
  // full-logit cosine is a diagnostic only — see docs/parity.md §3b) — and
  // LAGUNA_ATTN_F32=1 reverts attention from the default f16 compute path to
  // the legacy f32 one, and LAGUNA_COMBINE_CLASSIC=1 pins the routed-expert
  // combine to the candle chain (bit-identical to the fused kernel, so this only
  // matches the oracle's blessed path). mm runs the default mm_id prefill
  // (code-short's 58 tokens are >= MM_ID_MIN_SEQ) with the default f16 attention.
  const env = tier === "strict"
    ? { ...baseEnv(), LAGUNA_NO_MM_ID: "1", LAGUNA_MV_CLASSIC: "1", LAGUNA_ATTN_F32: "1", LAGUNA_COMBINE_CLASSIC: "1" }
    : baseEnv();
  await preflight(`${tier} candidate dump`);
  console.log(`  generating ${tier} candidate (Fused, ${tier === "strict" ? "classic mv fallback" : "mm_id"}) -> ${candPath}`);
  const cCode = await timed(`${tier} candidate`, () =>
    runToLog(
      [LOGITS_DUMP, "--model", MODEL, "--moe-impl", "fused", "--tokens", tokens, "--output", candPath],
      candLog,
      env,
    ),
  );
  if (cCode !== 0) die(`${tier} candidate dump failed (exit ${cCode}):\n${await tailLog(candLog)}`);

  const gateLog = join(dir, "gate.log");
  const gCode = await timed(`${tier} gate`, () =>
    runGate(PARITY_BIN, "logit_parity", dir, { LAGUNA_PARITY_TIER: tier }, gateLog),
  );
  rows.push({ tier, fixture, status: gCode === 0 ? "PASS" : "FAIL", metric: await gateMetric(tier, gateLog), logPath: gateLog });
}

// -------------------------------------------------------------- tier: decode

async function runDecodeFixture(fixture: string, parityDir: string, regenRef: boolean): Promise<void> {
  const dir = join(parityDir, `decode-${fixture}`);
  mkdirSync(dir, { recursive: true });
  const refPath = join(dir, "reference-greedy.json");
  const candPath = join(dir, "candidate-greedy.json");
  const tokens = await fixtureTokens(fixture);

  // Reference greedy: reused iff it is a genuine Reference-oracle greedy dump.
  const reuse = !regenRef && (await isReferenceDump(refPath, "greedy"));
  if (reuse) {
    console.log(`  reusing greedy reference ${refPath}`);
  } else {
    const log = join(dir, "reference-greedy.log");
    await preflight(`decode reference (${fixture})`);
    console.log(`  generating greedy reference (Reference oracle, ${fixture}, ${GREEDY_N} steps) -> ${refPath}`);
    const code = await timed(`decode reference ${fixture}`, () =>
      runToLog(
        [LOGITS_DUMP, "--model", MODEL, "--moe-impl", "reference", "--tokens", tokens, "--greedy", String(GREEDY_N), "--output", refPath],
        log,
        referenceEnv(),
      ),
    );
    if (code !== 0) die(`decode reference (${fixture}) failed (exit ${code}):\n${await tailLog(log)}`);
  }

  const candLog = join(dir, "candidate-greedy.log");
  await preflight(`decode candidate (${fixture})`);
  console.log(`  generating greedy candidate (Fused, forced replay, ${fixture}) -> ${candPath}`);
  const cCode = await timed(`decode candidate ${fixture}`, () =>
    runToLog(
      [LOGITS_DUMP, "--model", MODEL, "--moe-impl", "fused", "--replay", refPath, "--output", candPath],
      candLog,
      baseEnv(),
    ),
  );
  if (cCode !== 0) die(`decode candidate (${fixture}) failed (exit ${cCode}):\n${await tailLog(candLog)}`);

  const gateLog = join(dir, "gate.log");
  const gCode = await timed(`decode gate ${fixture}`, () =>
    runGate(PARITY_BIN, "greedy_parity", dir, { LAGUNA_PARITY_TIER: "decode" }, gateLog),
  );
  rows.push({ tier: "decode", fixture, status: gCode === 0 ? "PASS" : "FAIL", metric: await gateMetric("decode", gateLog), logPath: gateLog });
}

// ----------------------------------------------------------------- tier: ppl

async function runPplTier(parityDir: string, regenPplRef: boolean): Promise<void> {
  const dir = join(parityDir, "ppl");
  mkdirSync(dir, { recursive: true });
  const refPath = join(dir, "reference-ppl.json");
  const candPath = join(dir, "candidate-ppl.json");

  // The reference NLL is a frozen checked-in fixture. Regenerating it OVERWRITES
  // a committed artifact — do it only on demand, and warn to review/stage.
  if (regenPplRef) {
    console.log("  !! --regen-ppl-ref: regenerating the COMMITTED reference fixture");
    console.log(`     ${PPL_FIXTURE} — review the diff and stage it yourself.`);
    const log = join(dir, "reference-ppl.log");
    await preflight("ppl reference (regenerating committed fixture)");
    const code = await timed("ppl reference", () =>
      runToLog(
        [LOGITS_DUMP, "--model", MODEL, "--moe-impl", "reference", "--max-ctx", String(PPL_MAX_CTX), "--ppl", CORPUS, "--output", PPL_FIXTURE],
        log,
        referenceEnv(),
      ),
    );
    if (code !== 0) die(`ppl reference regeneration failed (exit ${code}):\n${await tailLog(log)}`);
  } else if (!existsSync(PPL_FIXTURE)) {
    die(`ppl reference fixture missing: ${PPL_FIXTURE} (regenerate with --regen-ppl-ref)`);
  }
  await Bun.write(refPath, Bun.file(PPL_FIXTURE));

  const candLog = join(dir, "candidate-ppl.log");
  await preflight("ppl candidate dump");
  console.log(`  generating ppl candidate (Fused, mm_id prefill over the corpus) -> ${candPath}`);
  const cCode = await timed("ppl candidate", () =>
    runToLog(
      [LOGITS_DUMP, "--model", MODEL, "--moe-impl", "fused", "--max-ctx", String(PPL_MAX_CTX), "--ppl", CORPUS, "--output", candPath],
      candLog,
      baseEnv(),
    ),
  );
  if (cCode !== 0) die(`ppl candidate dump failed (exit ${cCode}):\n${await tailLog(candLog)}`);

  const gateLog = join(dir, "gate.log");
  const gCode = await timed("ppl gate", () => runGate(PARITY_BIN, "ppl_parity", dir, {}, gateLog));
  rows.push({ tier: "ppl", fixture: "corpus", status: gCode === 0 ? "PASS" : "FAIL", metric: await gateMetric("ppl", gateLog), logPath: gateLog });
}

// ------------------------------------------------------------------- main

let PARITY_BIN = "";

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (!existsSync(MODEL)) die(`model not found: ${MODEL}`);
  if (!existsSync(LOGITS_DUMP)) {
    // Built below; but if the target dir is missing entirely, cargo will create it.
  }
  mkdirSync(opts.parityDir, { recursive: true });

  console.log(`parity-gate: tiers=[${opts.tiers.join(",")}] fixtures=[${opts.fixtures.join(",")}] dir=${opts.parityDir}`);
  if (opts.regenRef) console.log("  --regen-ref: reference dumps will be regenerated");

  // Build once up front: the logits-dump binary and the parity test binary.
  console.log("building logits-dump (release)...");
  const buildLog = join(opts.parityDir, "build-logits-dump.log");
  const bCode = await timed("build logits-dump", () =>
    runToLog(["cargo", "build", "--release", "--bin", "logits-dump"], buildLog, baseEnv()),
  );
  if (bCode !== 0) die(`cargo build --bin logits-dump failed (exit ${bCode}):\n${await tailLog(buildLog)}`);

  console.log("building parity test binary (release, --no-run)...");
  PARITY_BIN = await timed("build parity test", () =>
    buildParityTestBin(join(opts.parityDir, "build-parity-test.log")),
  );
  console.log(`  parity test binary: ${PARITY_BIN}`);

  for (const tier of opts.tiers) {
    if (tier === "ppl") {
      console.log("\n== tier ppl ==");
      await runPplTier(opts.parityDir, opts.regenPplRef);
      continue;
    }
    const fixtures = TIER_FIXTURES[tier].filter((f) => opts.fixtures.includes(f));
    console.log(`\n== tier ${tier} ==`);
    if (fixtures.length === 0) {
      // The requested --fixtures excluded every fixture this tier grades.
      for (const f of TIER_FIXTURES[tier]) {
        rows.push({ tier, fixture: f, status: "SKIPPED", metric: "fixture not in --fixtures" });
      }
      console.log("  skipped (no requested fixture applies to this tier)");
      continue;
    }
    for (const fixture of fixtures) {
      if (tier === "decode") await runDecodeFixture(fixture, opts.parityDir, opts.regenRef);
      else await runFullLogitTier(tier, opts.parityDir, opts.regenRef);
    }
  }

  // ---- summary
  console.log("\n==================== parity-gate summary ====================");
  // The strict tier grades the classic mv fallback path, not the shipped
  // default — label it so a PASS/FAIL isn't mistaken for the default decode path.
  const tierLabel = (t: Tier) => (t === "strict" ? "strict (classic mv fallback)" : t);
  let failed = 0;
  for (const r of rows) {
    if (r.status === "FAIL") failed++;
    const line = `  ${r.status.padEnd(7)} ${tierLabel(r.tier).padEnd(28)} ${r.fixture.padEnd(11)} ${r.metric}`;
    console.log(line);
  }
  if (timings.length) {
    console.log("\n  timings:");
    for (const t of timings) console.log(`    ${fmtSecs(t.seconds).padStart(6)}  ${t.label}`);
  }

  // Surface the tail of every failing log so the failure is readable without a pager.
  const fails = rows.filter((r) => r.status === "FAIL" && r.logPath);
  for (const r of fails) {
    console.log(`\n---- FAIL ${r.tier}/${r.fixture} — tail of ${r.logPath} ----`);
    console.log(await tailLog(r.logPath!, 40));
  }

  console.log(`\n${failed === 0 ? "ALL PASS" : `${failed} FAILED`} (${rows.filter((r) => r.status !== "SKIPPED").length} graded)`);
  process.exit(failed === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error("parity-gate: unexpected error:", e);
  process.exit(1);
});
