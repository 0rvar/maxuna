#!/usr/bin/env bun
// Standard bench harness: the guarded prefill/decode measurement protocol.
//
// Encodes the operational rules that ad-hoc bench scripts kept forgetting:
//   - ONE model process at a time (pgrep guard — two concurrent 75GB loads OOM
//     the GPU).
//   - Power mode is VERIFIED via pmset at start AND end, and stamped on every
//     result line. The System Settings Energy Mode toggle is per-power-source
//     and has silently failed to apply before; never trust the UI.
//   - Model output streams to files, never through a pager.
//   - LAGUNA_BENCH=1 (warm-up forward) so numbers are steady-state, and decode
//     runs >= 256 tokens — sub-second runs report boost-clock fiction.
//
// Usage:
//   bun scripts/bench.ts                  # all three: prefill 925, prefill 4k, decode 256
//   bun scripts/bench.ts --only decode    # subset: comma-separated of prefill-925,prefill-4k,decode
//   bun scripts/bench.ts --gate           # run `bun scripts/parity-gate.ts` first (cached refs)
//   bun scripts/bench.ts --expect-mode lpm|full   # abort if the machine is not in that mode (default: lpm)
//   bun scripts/bench.ts --out-dir DIR    # where raw outputs land (default: /tmp/laguna-bench)
//
// Bench prompts are committed fixtures (tests/fixtures/bench-prompts/): the
// same token counts every run, so numbers are comparable across sessions.

import { $ } from "bun";
import { mkdirSync } from "node:fs";
import { join, dirname } from "node:path";

const repo = dirname(import.meta.dir);
const fixtures = join(repo, "tests/fixtures/bench-prompts");

const args = process.argv.slice(2);
function flag(name: string): boolean {
  return args.includes(`--${name}`);
}
function opt(name: string, dflt: string): string {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
}

const outDir = opt("out-dir", "/tmp/laguna-bench");
const expectMode = opt("expect-mode", "lpm");
const only = opt("only", "prefill-925,prefill-4k,decode").split(",");
mkdirSync(outDir, { recursive: true });

async function powerMode(): Promise<"lpm" | "full"> {
  const out = await $`pmset -g`.text();
  // macOS reports the key as `lowpowermode` or `powermode` depending on
  // OS build; both use 1 = Low Power Mode.
  const m = out.match(/\b(?:low)?powermode\s+(\d)/);
  if (!m) throw new Error("pmset -g did not report (low)powermode");
  return m[1] === "1" ? "lpm" : "full";
}

async function guardNoModelProcess() {
  // Match actual model binaries, not every process whose command line mentions
  // the repo path (editors, this script). pgrep exits 1 when nothing matches.
  const proc = Bun.spawnSync([
    "pgrep",
    "-fl",
    "target/release/(laguna|logits-dump|deps/parity)|llama-(cli|server|bench)",
  ]);
  const hits = proc.stdout
    .toString()
    .split("\n")
    .filter((l) => l.trim());
  if (hits.length > 0) {
    console.error("ABORT: model process already running:\n" + hits.join("\n"));
    process.exit(1);
  }
}

const startMode = await powerMode();
if (startMode !== expectMode) {
  console.error(
    `ABORT: machine is in '${startMode}' but --expect-mode is '${expectMode}'.` +
      ` Flip the Energy Mode toggle for the CURRENT power source (Battery and` +
      ` Power Adapter have separate settings) and re-run.`,
  );
  process.exit(1);
}
await guardNoModelProcess();
console.log(`power mode verified: ${startMode}`);

if (flag("gate")) {
  console.log("=== parity gate (cached refs) ===");
  const gate = Bun.spawnSync(["bun", join(repo, "scripts/parity-gate.ts")], {
    cwd: repo,
    stdout: "inherit",
    stderr: "inherit",
  });
  if (gate.exitCode !== 0) {
    console.error(`ABORT: parity gate failed (exit ${gate.exitCode}) — not benching a broken build`);
    process.exit(gate.exitCode ?? 1);
  }
}

interface Bench {
  name: string;
  prompt: string;
  nTokens: number;
}
const benches: Bench[] = [
  { name: "prefill-925", prompt: join(fixtures, "prefill-925.txt"), nTokens: 8 },
  { name: "prefill-4k", prompt: join(fixtures, "prefill-4k.txt"), nTokens: 8 },
  { name: "decode", prompt: join(fixtures, "decode-630.txt"), nTokens: 256 },
];

const results: string[] = [];
for (const b of benches.filter((b) => only.includes(b.name))) {
  const outFile = join(outDir, `${b.name}.txt`);
  console.log(`=== bench ${b.name} (raw -> ${outFile}) ===`);
  const prompt = await Bun.file(b.prompt).text();
  const proc = Bun.spawnSync(
    [
      join(repo, "target/release/laguna"),
      "generate",
      "--model",
      join(repo, "models/laguna-s-2.1-Q4_K_M.gguf"),
      "--prompt",
      prompt,
      "--raw",
      "-n",
      String(b.nTokens),
      "--stats",
    ],
    { cwd: repo, env: { ...process.env, LAGUNA_BENCH: "1" }, maxBuffer: 64 * 1024 * 1024 },
  );
  await Bun.write(outFile, proc.stdout.toString() + proc.stderr.toString());
  if (proc.exitCode !== 0) {
    console.error(`ABORT: ${b.name} exited ${proc.exitCode}; see ${outFile}`);
    process.exit(1);
  }
  const stats = (proc.stdout.toString() + proc.stderr.toString())
    .split("\n")
    .filter((l) => l.startsWith("prefill:") || l.startsWith("decode:"));
  for (const line of stats) results.push(`[${b.name}] ${line}`);
}

const endMode = await powerMode();
console.log(`power mode at end: ${endMode}`);
console.log(`\n=== results (mode: ${startMode}${endMode !== startMode ? ` -> ${endMode} MODE CHANGED MID-RUN, numbers invalid` : ""}) ===`);
for (const line of results) console.log(line);
if (endMode !== startMode) process.exit(1);
