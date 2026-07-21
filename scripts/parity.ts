#!/usr/bin/env bun
/**
 * Cross-implementation parity check: our logits-dump JSON vs the vendored
 * llama.cpp fork's `llama-eval-callback` trace.
 *
 * What the eval-callback trace actually contains, per graph node:
 *   - name (fork cb() name + "-{layer}"; "-1" layers are bare), op, dtype, shape
 *   - a full-tensor `sum`
 *   - TRUNCATED sample values: first-3/last-3 along each dim (n=3 in the fork's
 *     printer). The last printed row corresponds to the LAST token = the last
 *     position, which is exactly what our dump stores in full.
 * It does NOT contain full tensors, so a true full-vector cosine on the final
 * logits is not available from this oracle -- that check lives in
 * tests/parity.rs (dump vs blessed dump). Here we do what the trace supports:
 *   1. First-divergence layer: walk ref nodes in graph order, compare each to
 *      the matching tap in our dump on (a) full-tensor sum and (b) the sampled
 *      last-position row; flag the first node whose relative error exceeds the
 *      threshold. This localizes where our math first drifts from the fork.
 *   2. Final-logits agreement: our logits' sum + sampled entries vs the fork's
 *      result_output node, plus a sampled cosine over the shared entries.
 *   3. Our own top-1 / top-5 (computed from our full logits); if --ref-top1 is
 *      given (e.g. from llama-cli greedy), assert top-1 agreement.
 *
 * Usage:
 *   bun scripts/parity.ts --ours ours.json --ref eval-callback.txt \
 *       [--report parity-report.json] [--threshold 0.01] [--ref-top1 <id>]
 */

type Args = Record<string, string | boolean>;
function parseArgs(argv: string[]): Args {
  const a: Args = {};
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (t.startsWith("--")) {
      const key = t.slice(2);
      const next = argv[i + 1];
      if (next === undefined || next.startsWith("--")) a[key] = true;
      else { a[key] = next; i++; }
    }
  }
  return a;
}

interface RefNode {
  order: number;
  name: string;
  op: string;
  dtype: string;
  ne: number[];        // ggml [inner..outer], length 4
  sum: number | null;
  samples: number[];       // every sampled value in the block
  lastRowSamples: number[]; // the final innermost row = last position
}

const FLOAT_RE = /-?(?:\d+\.\d+(?:[eE][-+]?\d+)?|\d+|nan|inf)/g;
function parseFloats(line: string): number[] {
  const out: number[] = [];
  const m = line.match(FLOAT_RE);
  if (!m) return out;
  for (const s of m) {
    const v = s === "nan" ? NaN : s === "inf" ? Infinity : s === "-inf" ? -Infinity : Number(s);
    out.push(v);
  }
  return out;
}

// <fn>: <name> = (<type>) <OP>(<src0>{ne}, <src1>{ne}}) = {ne}
// The fork's debug callback logs with __func__ as the prefix; current builds
// name it `common_debug_cb_eval`, older ones `ggml_debug` (as in the README).
const NODE_RE = /(?:common_debug_cb_eval|ggml_debug):\s*(\S+)\s*=\s*\(([^)]+)\)\s*(\w+)\(.*?\)\s*=\s*\{([^}]+)\}/;

function parseEvalCallback(text: string): RefNode[] {
  const lines = text.split("\n");
  const nodes: RefNode[] = [];
  let cur: RefNode | null = null;
  let order = 0;

  for (const line of lines) {
    const hdr = line.match(NODE_RE);
    if (hdr) {
      if (cur) nodes.push(cur);
      const ne = hdr[4].split(",").map((s) => parseInt(s.trim(), 10));
      cur = { order: order++, name: hdr[1], dtype: hdr[2], op: hdr[3], ne, sum: null, samples: [], lastRowSamples: [] };
      continue;
    }
    if (!cur) continue;

    const sumMatch = line.match(/^\s*sum\s*=\s*(\S+)/);
    if (sumMatch) {
      if (cur.sum === null) cur.sum = Number(sumMatch[1]);
      continue;
    }
    // A value row: "[ -1.1135,   1.4604, ... ]," (contains a bracket and digits).
    if (/\[/.test(line) && FLOAT_RE.test(line)) {
      const vals = parseFloats(line);
      if (vals.length) {
        cur.samples.push(...vals);
        cur.lastRowSamples = vals; // overwritten each row; final row survives
      }
    }
  }
  if (cur) nodes.push(cur);
  return nodes;
}

interface Tap {
  name: string;
  shape: number[];
  sum: number;
  mean: number;
  std: number;
  l2: number;
  first8: number[];
  last_row: number[] | null;
}
interface OurDump {
  model: string;
  tokens: number[];
  n_tokens: number;
  vocab: number;
  logits: number[];
  top1: number;
  top5: [number, number][];
  taps: Tap[];
}

const relErr = (a: number, b: number) => Math.abs(a - b) / (Math.abs(b) + 1e-6);

// rel-L2 of paired samples: align the ref's truncated first-3/last-3 samples to
// our full row. When the ref row was not truncated (<= 6 elements) the two are
// the same length and compared directly.
function sampledRelL2(refSamples: number[], ourRow: number[] | null): number | null {
  if (!ourRow || refSamples.length === 0) return null;
  let pairs: [number, number][];
  if (refSamples.length >= ourRow.length) {
    pairs = ourRow.map((v, i) => [refSamples[i], v]);
  } else {
    // ref is first ceil(k/2) then last floor(k/2) of the row.
    const head = Math.ceil(refSamples.length / 2);
    const tail = refSamples.length - head;
    pairs = [];
    for (let i = 0; i < head; i++) pairs.push([refSamples[i], ourRow[i]]);
    for (let i = 0; i < tail; i++) pairs.push([refSamples[head + i], ourRow[ourRow.length - tail + i]]);
  }
  let num = 0, den = 0;
  for (const [r, o] of pairs) { num += (o - r) ** 2; den += r * r; }
  return Math.sqrt(num) / (Math.sqrt(den) + 1e-6);
}

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const oursPath = args.ours as string;
  const refPath = args.ref as string;
  if (!oursPath || !refPath) {
    console.error("usage: bun scripts/parity.ts --ours ours.json --ref eval-callback.txt [--report out.json] [--threshold 0.01] [--ref-top1 ID]");
    process.exit(2);
  }
  const threshold = args.threshold ? Number(args.threshold) : 1e-2;
  const cosThreshold = 0.999;

  const ours: OurDump = await Bun.file(oursPath).json();
  const refText = await Bun.file(refPath).text();
  const refNodes = parseEvalCallback(refText);
  const refByName = new Map(refNodes.map((n) => [n.name, n]));
  const tapByName = new Map(ours.taps.map((t) => [t.name, t]));

  const out: string[] = [];
  const p = (s = "") => out.push(s);

  p(`LAGUNA PARITY`);
  p(`  ours: ${oursPath}`);
  p(`  ref:  ${refPath}`);
  p(`  ref nodes parsed: ${refNodes.length}   our taps: ${ours.taps.length}   threshold: ${threshold}`);
  p(`  tokens: ${ours.n_tokens}  vocab: ${ours.vocab}`);
  p();

  // ---- First-divergence walk over ref nodes in graph order ----
  interface Cmp { order: number; name: string; sumRelErr: number | null; rowRelL2: number | null; diverged: boolean; }
  const cmps: Cmp[] = [];
  for (const node of refNodes) {
    const tap = tapByName.get(node.name);
    if (!tap) continue;
    const sre = node.sum !== null ? relErr(tap.sum, node.sum) : null;
    // eval-callback's samples are the last INNERMOST row. For a rank-2 tensor
    // {feature, token} that row is the whole last-position feature vector, so
    // it aligns with our last_row. For rank>2 (e.g. Qcur {head_dim, head,
    // token}) it is only one head of the last position and would misalign, so
    // trust the full-tensor sum alone there.
    const rank2 = node.ne[2] === 1 && node.ne[3] === 1;
    const rrl = rank2 ? sampledRelL2(node.lastRowSamples, tap.last_row) : null;
    const worst = Math.max(sre ?? 0, rrl ?? 0);
    cmps.push({ order: node.order, name: node.name, sumRelErr: sre, rowRelL2: rrl, diverged: worst > threshold });
  }
  const firstDiv = cmps.find((c) => c.diverged) ?? null;

  p(`FIRST DIVERGENCE  (compared ${cmps.length} taps present in both)`);
  if (cmps.length === 0) {
    p(`  (no taps to compare — run logits-dump with --taps and ensure names match)`);
  } else if (!firstDiv) {
    p(`  none — all ${cmps.length} compared taps within rel error ${threshold}`);
  } else {
    p(`  first divergent node: ${firstDiv.name}  (sumRelErr=${fmt(firstDiv.sumRelErr)}, rowRelL2=${fmt(firstDiv.rowRelL2)})`);
    p(`  ${"node".padEnd(22)} ${"sumRelErr".padStart(11)} ${"rowRelL2".padStart(11)}  status`);
    const idx = cmps.indexOf(firstDiv);
    for (let i = Math.max(0, idx - 3); i <= Math.min(cmps.length - 1, idx + 1); i++) {
      const c = cmps[i];
      p(`  ${c.name.padEnd(22)} ${fmt(c.sumRelErr).padStart(11)} ${fmt(c.rowRelL2).padStart(11)}  ${c.diverged ? "DIVERGED" : "ok"}`);
    }
  }
  p();

  // ---- Final logits agreement ----
  const refOut = refByName.get("result_output");
  const ourLogitsSum = ours.logits.reduce((a, b) => a + b, 0);
  let logitsSumRelErr: number | null = null;
  let sampledCos: number | null = null;
  p(`FINAL LOGITS`);
  p(`  our top1: ${ours.top1}   top5: ${ours.top5.map(([id, v]) => `${id}(${v.toFixed(2)})`).join(", ")}`);
  if (refOut) {
    if (refOut.sum !== null) {
      logitsSumRelErr = relErr(ourLogitsSum, refOut.sum);
      p(`  result_output sum: ours=${ourLogitsSum.toFixed(3)} ref=${refOut.sum.toFixed(3)} relErr=${fmt(logitsSumRelErr)}  [${logitsSumRelErr <= threshold ? "PASS" : "FAIL"} @ ${threshold}]`);
    }
    // Sampled logits: ref gives first-3/last-3 of the vocab row; pull the same from ours.
    const V = ours.logits.length;
    const k = Math.ceil(refOut.lastRowSamples.length / 2);
    const t = refOut.lastRowSamples.length - k;
    const ourSampled = [...ours.logits.slice(0, k), ...ours.logits.slice(V - t)];
    if (refOut.lastRowSamples.length > 0) {
      sampledCos = cosine(ourSampled, refOut.lastRowSamples);
      p(`  result_output sampled cosine (${refOut.lastRowSamples.length} pts): ${sampledCos.toFixed(6)}  [${sampledCos >= cosThreshold ? "PASS" : "FAIL"} @ ${cosThreshold}]`);
      p(`  NOTE: full-vocab cosine/top-5 need two full dumps — see tests/parity.rs. eval-callback only exposes ${refOut.lastRowSamples.length} sampled logits.`);
    }
  } else {
    p(`  (no result_output node in ref trace — cannot compare final logits)`);
  }

  // ---- top-1 agreement (optional, needs a full-vocab oracle) ----
  let top1Match: boolean | null = null;
  if (args["ref-top1"] !== undefined) {
    const refTop1 = Number(args["ref-top1"]);
    top1Match = ours.top1 === refTop1;
    p(`  top-1 vs oracle: ours=${ours.top1} ref=${refTop1}  [${top1Match ? "PASS" : "FAIL"}]`);
  }
  p();

  // ---- Verdict ----
  const checks: boolean[] = [];
  if (cmps.length) checks.push(!firstDiv);
  if (logitsSumRelErr !== null) checks.push(logitsSumRelErr <= threshold);
  if (sampledCos !== null) checks.push(sampledCos >= cosThreshold);
  if (top1Match !== null) checks.push(top1Match);
  const pass = checks.length > 0 && checks.every(Boolean);
  p(`SUMMARY: ${pass ? "PASS" : "FAIL"}  (${checks.filter(Boolean).length}/${checks.length} checks passed)`);

  console.log(out.join("\n"));

  if (args.report) {
    const report = {
      ours: oursPath, ref: refPath, threshold, cosThreshold,
      nTokens: ours.n_tokens, vocab: ours.vocab,
      comparedTaps: cmps.length,
      firstDivergence: firstDiv,
      finalLogits: { ourTop1: ours.top1, ourTop5: ours.top5, logitsSumRelErr, sampledCosine: sampledCos },
      top1Match,
      pass,
    };
    await Bun.write(args.report as string, JSON.stringify(report, null, 2));
    console.error(`report written: ${args.report}`);
  }

  process.exit(pass ? 0 : 1);
}

function fmt(x: number | null): string {
  return x === null ? "n/a" : x.toExponential(2);
}

main();
