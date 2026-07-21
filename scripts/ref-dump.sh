#!/usr/bin/env bash
# Drive the vendored llama.cpp fork to produce the reference side of a parity
# check for one prompt: the per-graph-node tensor trace (llama-eval-callback)
# and, optionally, the greedy continuation (llama-cli --temp 0) for top-1
# token agreement.
#
# llama-eval-callback takes a TEXT prompt, not raw token ids -- there is no
# raw-token input in the fork. It tokenizes the prompt itself (with BOS, with
# special-token parsing) and ECHOES the exact ids it used. Those echoed ids are
# the authoritative token sequence: this script extracts them into tokens.txt
# so our logits-dump can be fed the identical ids. A separate llama-tokenize
# run is done only as a cross-check and warns on any mismatch.
#
# Usage:
#   scripts/ref-dump.sh -m models/laguna-s-2.1-Q4_K_M.gguf -p "def fib(n):" -o /tmp/ref-code
#   scripts/ref-dump.sh -m MODEL --fixture code-short -o /tmp/ref-code
#   scripts/ref-dump.sh -m MODEL -p "..." -o OUTDIR --gen 24   # also greedy-decode 24 tokens
#
# Outputs (in OUTDIR):
#   eval-callback.txt  full per-node trace (stdout+stderr), the numeric oracle
#   tokens.txt         authoritative token ids, comma-separated, as echoed by eval-callback
#   llama-cli.txt      greedy continuation (only with --gen N>0), top-1 oracle
#   ref-cmd.txt        the exact logits-dump command to produce our matching dump
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
bin="$root/reference/llama.cpp-laguna-branch/build/bin"
fixtures="$root/tests/fixtures/parity-prompts.json"

model=""
prompt=""
fixture=""
outdir=""
ngl=999
ctx=4096
gen=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)   model="$2"; shift 2;;
    -p|--prompt)  prompt="$2"; shift 2;;
    --fixture)    fixture="$2"; shift 2;;
    -o|--out)     outdir="$2"; shift 2;;
    --ngl)        ngl="$2"; shift 2;;
    -c|--ctx)     ctx="$2"; shift 2;;
    --gen)        gen="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[[ -n "$model"  ]] || { echo "error: -m/--model is required" >&2; exit 2; }
[[ -n "$outdir" ]] || { echo "error: -o/--out is required" >&2; exit 2; }
[[ -f "$model"  ]] || { echo "error: model not found: $model" >&2; exit 2; }

# Resolve prompt from the fixture file if --fixture was given.
if [[ -n "$fixture" ]]; then
  prompt="$(bun -e '
    const f = await Bun.file(process.argv[1]).json();
    const p = f.prompts.find(p => p.id === process.argv[2]);
    if (!p) { console.error("no such fixture id: " + process.argv[2]); process.exit(1); }
    process.stdout.write(p.text);
  ' "$fixtures" "$fixture")"
fi
[[ -n "$prompt" ]] || { echo "error: provide -p/--prompt or --fixture" >&2; exit 2; }

mkdir -p "$outdir"
export DYLD_FALLBACK_LIBRARY_PATH="$bin:${DYLD_FALLBACK_LIBRARY_PATH:-}"

echo ">> eval-callback trace -> $outdir/eval-callback.txt"
"$bin/llama-eval-callback" \
  --model "$model" \
  --prompt "$prompt" \
  -ngl "$ngl" \
  -c "$ctx" \
  --seed 42 \
  > "$outdir/eval-callback.txt" 2>&1 || {
    echo "eval-callback failed; tail of output:" >&2
    tail -n 20 "$outdir/eval-callback.txt" >&2
    exit 1
  }

# Extract the authoritative ids. The fork logs "number of input tokens = N"
# then N lines, each prefixed with a timestamp+level, ending in one id (e.g.
# "0.04.057 I   1172"). The id is the last whitespace-delimited field; N caps
# how many we read so we stop before the first ggml_debug node.
awk '
  /number of input tokens =/ { want=$NF+0; grab=1; got=0; next }
  grab {
    if ($NF ~ /^[0-9]+$/) {
      printf "%s%s", (got ? "," : ""), $NF
      got++
      if (got >= want) grab=0
    } else {
      grab=0
    }
    next
  }
' "$outdir/eval-callback.txt" > "$outdir/tokens.txt"

ntok="$(tr ',' '\n' < "$outdir/tokens.txt" | grep -c .)"
[[ "$ntok" -gt 0 ]] || { echo "error: no token ids parsed from eval-callback output" >&2; exit 1; }
echo ">> $ntok tokens -> $outdir/tokens.txt"

# Cross-check against llama-tokenize (add-bos on, matching the model default).
if [[ -x "$bin/llama-tokenize" ]]; then
  xt="$("$bin/llama-tokenize" --model "$model" --prompt "$prompt" --ids 2>/dev/null \
        | tr -d '[] \n' || true)"
  auth="$(tr -d ' ' < "$outdir/tokens.txt")"
  if [[ -n "$xt" && "$xt" != "$auth" ]]; then
    echo "!! WARNING: llama-tokenize ids differ from eval-callback ids" >&2
    echo "   eval-callback: $auth" >&2
    echo "   llama-tokenize: $xt" >&2
    echo "   (using eval-callback ids as authoritative; the difference is usually parse_special)" >&2
  fi
fi

# Optional greedy continuation for top-1 token agreement.
if [[ "$gen" -gt 0 ]]; then
  echo ">> greedy decode $gen tokens -> $outdir/llama-cli.txt"
  # -st -no-cnv: single-turn, non-conversation. With a predefined --prompt this
  # runs once and exits instead of dropping into the interactive chat loop
  # (which otherwise spins forever on an empty stdin). stdin from /dev/null too.
  "$bin/llama-cli" \
    --model "$model" \
    --prompt "$prompt" \
    -n "$gen" \
    --temp 0 \
    -ngl "$ngl" \
    -c "$ctx" \
    --seed 42 \
    -st -no-cnv \
    --no-display-prompt \
    < /dev/null \
    > "$outdir/llama-cli.txt" 2>&1 || {
      echo "llama-cli failed; see $outdir/llama-cli.txt" >&2
    }
fi

# Emit the exact command to produce our matching dump.
cat > "$outdir/ref-cmd.txt" <<EOF
cargo run --release --bin logits-dump -- \\
  --model "$model" \\
  --tokens "\$(cat "$outdir/tokens.txt")" \\
  --taps \\
  --output "$outdir/ours.json"

bun scripts/parity.ts \\
  --ours "$outdir/ours.json" \\
  --ref "$outdir/eval-callback.txt" \\
  --report "$outdir/parity-report.json"
EOF

echo ">> done. next steps written to $outdir/ref-cmd.txt"
cat "$outdir/ref-cmd.txt"
