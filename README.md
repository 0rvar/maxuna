# laguna

A from-scratch Rust inference engine for [poolside's Laguna S 2.1](https://huggingface.co/poolside/Laguna-S-2.1)
(118B-total / 8B-active MoE coding model), built on [candle](https://github.com/huggingface/candle).

Design target: **maximum tokens/sec on one machine** — Apple Silicon (M5 Max,
128GB unified), Metal only, batch=1, official GGUF quants. No CPU fallback, no
multi-GPU, no server (yet — see `TODO.md`).

## Quick start

```bash
scripts/fetch-model.sh            # downloads Q4_K_M (75GB) + DFlash draft into models/
cargo build --release

# one-shot generation (chat template + thinking enabled by default)
cargo run --release --bin laguna -- generate \
  -m models/laguna-s-2.1-Q4_K_M.gguf -p "Write fizzbuzz in Rust." --stats

# raw completion (no template, BOS prepended)
cargo run --release --bin laguna -- generate \
  -m models/laguna-s-2.1-Q4_K_M.gguf --raw -p "def fibonacci(n):" --temp 0

# interactive chat REPL
cargo run --release --bin laguna -- chat -m models/laguna-s-2.1-Q4_K_M.gguf

# dump GGUF metadata + parsed config
cargo run --release --bin laguna -- inspect -m models/laguna-s-2.1-Q4_K_M.gguf
```

Expect ~30s model load (warm page cache). `--moe-impl reference` swaps in the
slow-but-oracle expert path (~50x slower decode); the default `fused` path is
parity-blessed against it and against the llama.cpp fork.

## How it works

Laguna S 2.1: 48 layers; interleaved attention (every 4th layer global with
YaRN rope, the rest sliding-window 512 with plain rope); per-layer head counts
(48/72); QK-norm; per-head softplus attention output gating; 256 routed experts
(sigmoid router, top-10, DeepSeek-style selection bias) + 1 shared expert.

The performance-critical trick: candle's Metal kernel library ships llama.cpp's
quantized indexed MoE matmuls (`kernel_mul_mv_id_*`, `kernel_mul_mm_id_*`)
compiled but with no Rust host code — `src/ops/` provides the dispatch, running
expert selection and the expert matmuls entirely on-GPU over the stacked
quantized tensors, zero-copy from the GGUF upload.

Module map: `config.rs` (GGUF metadata → config), `gguf.rs` (loader, quant-
agnostic weights, zero-copy `ExpertStack`), `rope.rs` (dual YaRN/plain),
`kv_cache.rs` (prealloc full / 512-ring SWA), `attention.rs`, `moe.rs` (routing
+ fused/reference expert runners), `ops/` (Metal kernel dispatch), `model.rs`
(assembly + parity taps), `tokenizer.rs`/`chat.rs`/`sampler.rs` (I/O),
`generate.rs` (prefill/decode loop), `bin/laguna.rs` (CLI),
`bin/logits-dump.rs` (parity harness).

## Verification

Correctness is proven by parity against poolside's llama.cpp fork (vendored at
`reference/llama.cpp-laguna-branch`, built via `scripts/build-llamacpp.sh`),
which runs the identical GGUF. See `docs/parity.md` for the runbook: full-logit
reference-vs-fused gate, per-layer bisection against `llama-eval-callback`, and
raw greedy token parity. Unit tests: `cargo test --release` (the `ops` tests
require the Metal device).

## Status

Correct and parity-gated end-to-end (2026-07-21). Perf work ongoing — see
`TODO.md` for the deferred-work ledger (DFlash speculative decoding, HTTP
server, KV reuse across chat turns, 1M-context, …).
