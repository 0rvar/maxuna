# Parity verification

How we prove the Rust/candle engine reproduces the vendored llama.cpp fork
(`reference/llama.cpp-laguna-branch`, poolside's branch with native Laguna
support) on Laguna S 2.1. Two independent tracks, because the two oracles expose
different things.

## Two tracks

**Track A — first-divergence bisection (`scripts/parity.ts` vs `llama-eval-callback`).**
The fork's `llama-eval-callback` runs the real graph and prints, for every node,
a full-tensor `sum` plus a handful of sampled values. It does *not* print full
tensors. That is exactly enough to walk the graph in execution order and find
the *first* layer where our intermediate math drifts from the fork — which is
what you want when something is wrong. It is not enough for a real full-vector
cosine on the logits.

**Track B — full-logit gate (`tests/parity.rs`, dump vs dump).**
Two `logits-dump` JSON files in the same schema are compared on the full
last-position logit vector: cosine, top-1, top-5. The reference dump is a
blessed run (initially of our own engine as a regression guard; in WP8, one
derived from the fork). This is the track that enforces `cos >= 0.999`.

Neither track needs the other; run Track A to *locate* a bug, Track B to *gate*
a release.

## The dump format (`logits-dump`)

`src/bin/logits-dump.rs` feeds raw token ids through one forward pass and writes
JSON. The format is final as of WP6; the tool compiles now but only produces
output once WP7 implements `LagunaModel::{load,forward,set_tap_capture,take_taps}`.

```jsonc
{
  "model": "models/laguna-s-2.1-Q4_K_M.gguf",
  "prompt": "def fib(n):",       // provenance only; may be null. The tool never tokenizes.
  "moe_impl": "reference",        // reference (oracle) or fused
  "tokens": [2, 1288, ...],       // input token ids (u32)
  "n_tokens": 12,
  "vocab": 100352,
  "logits": [ ...vocab f32... ],  // FULL last-position logits
  "top1": 1288,
  "top5": [[1288, 21.5], ...],    // (token_id, logit), descending
  "taps": [
    {
      "name": "attn_norm-0",      // fork cb() name + "-{layer}"; layer -1 nodes are bare
      "shape": [12, 3072],        // candle dims, outer..inner (last dim = feature)
      "sum": 12.34,               // whole-tensor sum — directly comparable to eval-callback `sum`
      "mean": 0.001, "std": 0.98, "l2": 34.2,
      "first8": [ ... ],          // first 8 of the last-position row
      "last_row": [ ...feature f32... ] | null  // full last-position row; null if > 16384 elems
    }
  ]
}
```

Why "last-position row": eval-callback truncates every tensor to first-3/last-3
along each dim, and its last printed row is the last token. So the last-position
feature vector is the one row we can compare in detail on both sides. We store
it in full; the fork gives its first-3/last-3.

## Tap names (`cb()` in `reference/.../src/models/laguna.cpp`)

The fork names graph nodes with `cb(tensor, name, il)`. The printed node name is
`"{name}-{il}"` for `il >= 0` and bare `"{name}"` for `il == -1`
(`src/llama-context.cpp:2437`). `logits-dump` uses the identical names so
`parity.ts` can match by name.

Per-layer taps, in graph order (`il` = 0..n_layer-1; S 2.1 has 48 layers):

| tap name             | what it is                                              | layers        |
|----------------------|---------------------------------------------------------|---------------|
| `attn_norm`          | pre-attention RMSNorm output                            | all           |
| `attn_gate_proj`     | `g_proj` on the pre-attention hidden state (pre-softplus)| all          |
| `Qcur_normed`        | Q after head-dim QK-RMSNorm, pre-RoPE                   | all           |
| `Kcur_normed`        | K after head-dim QK-RMSNorm, pre-RoPE                   | all           |
| `Qcur_rope`          | Q after RoPE (YaRN on full layers, plain on SWA)        | all           |
| `Kcur_rope`          | K after RoPE                                            | all           |
| `attn_out`           | attention output before the softplus gate              | all           |
| `attn_gate_softplus` | the gate after `softplus(g_proj)`                       | all           |
| `attn_gated`         | attention output after multiplying by the gate         | all           |
| `attn_o_proj`        | after the output projection `wo`                        | all           |
| `ffn_inp`            | residual sum feeding the FFN block                      | all           |
| `ffn_norm`           | pre-FFN RMSNorm output                                  | all           |
| `ffn_moe_out`        | routed mixture-of-experts output (scaled, summed)       | MoE only (il>=1)|
| `ffn_shexp`          | shared-expert SwiGLU output                             | MoE only (il>=1)|
| `ffn_out`            | MoE: `moe_out + shexp`; dense layer 0: dense SwiGLU     | all           |
| `l_out`              | layer output (post-FFN residual, after control vector)  | all           |

Global taps (`il == -1`, bare names), in order:

| tap name        | what it is                                                     |
|-----------------|----------------------------------------------------------------|
| `h_nextn`       | pre-final-norm residual stream (DFlash drafter capture point)  |
| `result_norm`   | final RMSNorm output (the embedding fed to the lm head)        |
| `result_output` | final logits = `output @ result_norm` (last position)          |

S 2.1 has one leading dense layer (`leading_dense_block_count = 1`), so layer 0
emits `ffn_out` from a dense SwiGLU and *no* `ffn_moe_out`/`ffn_shexp`; layers
1..47 are MoE. `llama-eval-callback` additionally prints the generic helper
nodes that `build_attn`/`build_moe_ffn`/`build_ffn`/`build_norm` emit internally
(e.g. `kq`, `kqv`, `ffn_moe_logits`, `ffn_moe_weights`); those are not in our
explicit tap set, and `parity.ts` simply ignores nodes with no matching tap.

## eval-callback output format

`common_debug_cb_eval` (`common/debug.cpp`) prints, per node:

```
ggml_debug:                 inp_embd = (f32)   GET_ROWS(token_embd.weight{2560, 51200, 1, 1}, inp_tokens{1, 1, 1, 1}}) = {2560, 1, 1, 1}
    [
     [
      [ -0.0181,   0.0272,   0.0272,   ...,   0.19,   0.02,  -0.11  ],
      ...
     ],
    ]
    sum = -3.214000
```

Findings that shaped the tooling:

- **Header:** `ggml_debug: <name> = (<dtype>) <OP>(<src0>{ne}, <src1>{ne}}) = {ne}`.
  `ne` is ggml order — `ne[0]` is the innermost/feature dim, `ne[1]` the token
  dim (the transpose of our candle `shape`).
- **`sum`** is over the *entire* tensor (computed before the truncated print), so
  it is a real full-tensor signal even though the values are truncated. This is
  the backbone of the divergence walk.
- **Values are truncated** to first-3/last-3 along each dimension (`n = 3`,
  hardcoded in `common_debug_print_tensor`). You cannot recover a full tensor —
  hence Track B for full-vector cosine. The final printed innermost row is the
  last token = last position.
- **Token ids are echoed.** `run()` in `examples/eval-callback/eval-callback.cpp`
  prints `number of input tokens = N` followed by N lines of one id each. There
  is no raw-token input to the fork, so these echoed ids are authoritative;
  `ref-dump.sh` extracts them so our dump runs on the identical sequence.
- Everything goes to stdout+stderr via the common logger, so capture with `2>&1`.

Optional full-vocab top-k oracle: `llama-server`'s `/completion` accepts
`"n_probs": 5` and returns per-position top-5 tokens with logprobs
(`tools/server/server-task.cpp`). Useful in WP8 if you want top-5 agreement
straight from the fork rather than dump-vs-dump.

## Runbook

Prerequisites: the fork is built at `reference/llama.cpp-laguna-branch/build/bin`
and the model is at `models/laguna-s-2.1-Q4_K_M.gguf`.

**1. Produce the reference side** (per prompt). `ref-dump.sh` runs
eval-callback, extracts the authoritative token ids, cross-checks against
`llama-tokenize`, and optionally greedy-decodes for a top-1 oracle:

```bash
scripts/ref-dump.sh -m models/laguna-s-2.1-Q4_K_M.gguf --fixture code-short -o /tmp/ref-code --gen 24
# or with an ad-hoc prompt:
scripts/ref-dump.sh -m models/laguna-s-2.1-Q4_K_M.gguf -p "def fib(n):" -o /tmp/ref-code
```

Outputs in the out dir: `eval-callback.txt` (the trace), `tokens.txt`
(authoritative ids, comma-separated), `llama-cli.txt` (greedy continuation, with
`--gen`), `ref-cmd.txt` (the exact next commands).

**2. Produce our side** on the identical ids (needs WP7):

```bash
cargo run --release --bin logits-dump -- \
  --model models/laguna-s-2.1-Q4_K_M.gguf \
  --tokens "$(cat /tmp/ref-code/tokens.txt)" \
  --taps \
  --output /tmp/ref-code/ours.json
```

**3a. Track A — bisection:**

```bash
bun scripts/parity.ts \
  --ours /tmp/ref-code/ours.json \
  --ref  /tmp/ref-code/eval-callback.txt \
  --report /tmp/ref-code/parity-report.json
```

Prints a pass/fail summary: first divergent node (with `sumRelErr`/`rowRelL2`),
final-logits sum + sampled cosine. Divergence threshold defaults to rel error
`1e-2`; override with `--threshold`. Exit code 0 = pass.

**3b. Track B — full-logit gate:** put two dumps in a directory as
`candidate.json` and `reference.json`, then:

```bash
LAGUNA_PARITY_DIR=/tmp/ref-code cargo test --test parity -- --ignored --nocapture
```

## Pass criteria

- Track A: what disqualifies is a divergence *cliff* — one node whose deviation
  jumps orders of magnitude above its layer neighborhood. Smooth monotonic drift
  across layers (observed in practice: sampled-row rel-L2 growing ~0.001 at layer
  0 to ~0.2 at layer 47) is expected candle-Metal vs ggml-Metal kernel noise on
  identical Q4_K_M weights, not a bug. parity.ts's `--threshold` (default `1e-2`)
  flags candidates; judge them against the drift profile.
- Track B: final-logit cosine `>= 0.999`; top-1 agreement; top-5 overlap `>= 4/5`;
  identical input token ids.
- Greedy: sequences must agree except at near-ties. A divergence is acceptable
  when the logit gap between the two candidate tokens at the divergence point is
  `< 0.15` — the empirical Q4_K_M cross-kernel noise floor is ~0.1 logit, so
  demanding tighter (the original 1e-3) fails correct engines.
  (WP8 baseline, 2026-07-21: code-short agreed 107 tokens then split on a 0.015
  gap; text-mixed 16 tokens / 0.0053; long-swa 9 post-609-prompt tokens / 0.079.)

## Raw greedy oracle

`llama-cli -st -no-cnv` APPLIES THE CHAT TEMPLATE — its `llama-cli.txt` is a
chat-wrapped continuation, not a raw-completion oracle, and will "diverge" at
token 1 against a raw run. For raw greedy parity use `llama-server /completion`
with a token-id ARRAY as the prompt (no template, no tokenizer):

```bash
curl -s localhost:8080/completion -d '{
  "prompt": [2, 750, 15243, ...], "n_predict": 200, "temperature": 0,
  "samplers": []
}'
```

## Limitations

- eval-callback exposes only per-node sums + first-3/last-3 samples, so Track A's
  logit comparison is coarse (sum + a few sampled logits, plus a sampled cosine
  over those points). Full-vector cosine is Track B's job.
- eval-callback's samples are the last *innermost* row. For rank-2 tensors
  ({feature, token}) that is the whole last-position vector and lines up with our
  `last_row`. For rank>2 tensors (`Qcur_normed`/`Kcur_normed`, shaped {head_dim,
  head, token}) it is only one head of the last position, so `parity.ts` skips
  the sampled-row check there and relies on the full-tensor `sum`, which is exact
  at every rank.
- eval-callback computes logits for the last position only (`llama_batch_get_one`),
  so per-position logit parity is not available; both tracks compare the last
  position. This matches `logits-dump`, which also returns last-position logits.
- CPU vs Metal floating-point accumulation in the fork can differ slightly; run
  eval-callback with `-ngl 999` (the `ref-dump.sh` default) so the fork uses the
  Metal path, closest to our engine.
