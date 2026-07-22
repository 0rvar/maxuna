//! Parity harness: feed raw token ids, run one forward pass, and dump the
//! final-position logits (plus, optionally, the per-layer intermediate taps
//! named after the fork's cb() names) as JSON for comparison against the
//! vendored llama.cpp fork.
//!
//! Two consumers read this JSON:
//!   * `scripts/parity.ts` cross-checks it against `llama-eval-callback` output
//!     (per-node sums + samples) to localize the first divergent layer.
//!   * `tests/parity.rs` compares two of these dumps (candidate vs blessed
//!     reference) on the full logit vectors (cosine / top-1 / top-5).
//!
//! Dump format is FINAL as of WP6 (the model methods it calls are `todo!()`
//! until WP7, so it compiles now but only produces output once WP7 lands).
//!
//! Schema (see `docs/parity.md` for the authoritative description):
//! ```json
//! {
//!   "model": "models/laguna-s-2.1-Q4_K_M.gguf",
//!   "prompt": "def fib(n):",          // optional provenance, null when omitted
//!   "moe_impl": "reference",
//!   "tokens": [2, 1288, ...],          // input token ids (u32)
//!   "n_tokens": 12,
//!   "vocab": 151936,
//!   "logits": [ ...vocab f32... ],     // FULL last-position logits
//!   "top1": 1288,
//!   "top5": [[1288, 21.5], ...],       // (token_id, logit), descending
//!   "taps": [
//!     {
//!       "name": "attn_norm-0",         // fork cb() name + "-{layer}" ("-1" layers are bare)
//!       "shape": [12, 3072],           // candle dims, outer..inner; last dim = feature
//!       "sum": 12.34,                  // whole-tensor sum (matches eval-callback `sum`)
//!       "mean": 0.001, "std": 0.98, "l2": 34.2,
//!       "first8": [ ...<=8 f32... ],    // first 8 of the last-position row
//!       "last_row": [ ...feature f32... ] | null  // full last-position row, null if > CAP
//!     }, ...
//!   ]
//! }
//! ```
use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use clap::Parser;
use serde_json::{Value, json};

use laguna::LagunaConfig;
use laguna::gguf;
use laguna::model::LagunaModel;
use laguna::ops::ExpertRunner;
use laguna::tokenizer::LagunaTokenizer;

/// Full `last_row` arrays above this many elements are dropped (summary stats
/// only). 16384 keeps hidden-sized rows (~3072) but drops vocab-sized rows;
/// the full last-position logits live in the top-level `logits` field anyway.
const LAST_ROW_CAP: usize = 16384;

#[derive(Parser)]
#[command(name = "logits-dump", about = "Dump Laguna logits + taps as JSON for parity checks")]
struct Cli {
    #[arg(short, long)]
    model: PathBuf,

    /// Token ids: comma- or space-separated, brackets optional, so the output
    /// of `llama-tokenize --ids` or the token echo of `llama-eval-callback`
    /// can be pasted straight through (e.g. "[2, 1288, 40]" or "2 1288 40").
    /// Required except in `--replay` mode, where the prompt is taken from the
    /// greedy dump being replayed.
    #[arg(short, long)]
    tokens: Option<String>,

    /// Optional prompt text, recorded in the dump for provenance only (this
    /// tool never tokenizes — feed ids via --tokens so both sides agree).
    #[arg(short, long)]
    prompt: Option<String>,

    /// Also capture the per-layer intermediate taps (the fork cb() names).
    #[arg(long)]
    taps: bool,

    /// Decode-parity gate, reference side: after prefill, free-run greedy decode
    /// N tokens (argmax over logits, no sampling) and emit a `kind:"greedy"`
    /// dump. Never stops early on EOG — always emits exactly N steps so the gate
    /// can compare equal-length sequences. Ignores --taps.
    #[arg(long, value_name = "N")]
    greedy: Option<usize>,

    /// Decode-parity gate, candidate side: load a `kind:"greedy"` dump, prefill
    /// its prompt, then teacher-force its step tokens one at a time — recording
    /// THIS runner's own argmax (top-1/top-2) at each step BEFORE forcing the
    /// reference token. Emits a `kind:"replay"` dump. The prompt comes from the
    /// dump, so --tokens is not needed (and is ignored if given).
    #[arg(long, value_name = "GREEDY_DUMP")]
    replay: Option<PathBuf>,

    /// Perplexity-parity gate: tokenize the given raw-text corpus (via the
    /// crate tokenizer, add_special_tokens=false, one leading BOS), score the
    /// whole corpus in a single continuous chunked-prefill pass, and emit a
    /// `kind:"ppl"` dump (mean next-token NLL + per-chunk means). Mutually
    /// exclusive with --greedy/--replay; ignores --taps/--tokens. See
    /// docs/parity.md "Perplexity gate".
    #[arg(long, value_name = "CORPUS")]
    ppl: Option<PathBuf>,

    /// Tokenizer JSON for --ppl (the crate errors on GGUF-embedded vocab).
    #[arg(long, default_value = "reference/tokenizer.json")]
    tokenizer: PathBuf,

    /// Expert FFN implementation: "reference" (correctness oracle) or "fused".
    #[arg(long, default_value = "reference")]
    moe_impl: String,

    /// KV-cache context budget; must exceed the longest parity prompt.
    #[arg(long, default_value_t = 4096)]
    max_ctx: usize,

    #[arg(short, long)]
    output: PathBuf,
}

fn parse_tokens(s: &str) -> Result<Vec<u32>> {
    s.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split([',', ' ', '\n', '\t'])
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<u32>().with_context(|| format!("bad token id {t:?}")))
        .collect()
}

fn expert_runner(name: &str) -> Result<ExpertRunner> {
    match name {
        "reference" | "ref" => Ok(ExpertRunner::Reference),
        "fused" => Ok(ExpertRunner::Fused),
        other => anyhow::bail!("unknown --moe-impl {other:?} (expected reference|fused)"),
    }
}

/// Top-`k` (token id, logit) pairs, descending by logit.
fn topk(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    idx.into_iter().take(k).map(|i| (i, logits[i as usize])).collect()
}

/// Whole-tensor stats + the full last-position row, as a JSON tap object.
/// Treats dim 0 as the token/position axis: the "last row" is every feature at
/// the final position (for [seq, hidden] -> [hidden]; for [seq, n_head,
/// head_dim] -> flattened head features). Matches the last printed row of
/// `llama-eval-callback`, whose ggml layout is the transpose (ne[0]=feature
/// innermost, ne[1]=token).
fn tap_value(name: &str, t: &Tensor) -> Result<Value> {
    let t = t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let shape = t.dims().to_vec();
    let flat = t.flatten_all()?.to_vec1::<f32>()?;
    let n = flat.len().max(1);

    let sum: f32 = flat.iter().copied().sum();
    let mean = sum / n as f32;
    let var = flat.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let std = var.sqrt();
    let l2 = flat.iter().map(|&x| x * x).sum::<f32>().sqrt();

    let row: Vec<f32> = if shape.len() <= 1 || shape[0] <= 1 {
        flat.clone()
    } else {
        let row_len = flat.len() / shape[0];
        flat[(shape[0] - 1) * row_len..].to_vec()
    };
    let first8: Vec<f32> = row.iter().take(8).copied().collect();
    let last_row = if row.len() <= LAST_ROW_CAP { Value::from(row) } else { Value::Null };

    Ok(json!({
        "name": name,
        "shape": shape,
        "sum": sum,
        "mean": mean,
        "std": std,
        "l2": l2,
        "first8": first8,
        "last_row": last_row,
    }))
}

/// (top-1, top-2) as (token id, logit) pairs. Argmax is `top2(..).0.0`.
/// Caller guarantees `logits.len() >= 2` (checked once against `vocab` in `main`).
fn top2(logits: &[f32]) -> ((u32, f32), (u32, f32)) {
    let t = topk(logits, 2);
    (t[0], t[1])
}

/// (L2 norm in f64, count of non-finite entries) over a full logit vector. The
/// greedy/replay dumps carry no full logits, so these per-step scalars are the
/// only scale/finiteness signal the decode gate has. Non-finite entries are
/// excluded from the norm (so `l2` stays finite and usable) and reported
/// separately via the count.
fn logit_scale(logits: &[f32]) -> (f64, u64) {
    let mut sumsq = 0.0f64;
    let mut nonfinite = 0u64;
    for &x in logits {
        if x.is_finite() {
            sumsq += x as f64 * x as f64;
        } else {
            nonfinite += 1;
        }
    }
    (sumsq.sqrt(), nonfinite)
}

/// Read a forward's last-position logits back to a host `Vec<f32>`.
fn logits_to_host(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?)
}

/// Records HOW a dump was produced so the parity gate can validate the tier it is
/// graded under (a decode/mv_id candidate graded under the loose mm tier would
/// mask a regression). `seq_len` is the prefill length; `mm_min_seq` /
/// `mm_variant` / `no_mm_id` are the fused-MoE kernel-selection state, so
/// "mm_id path active" is derivable as `moe_impl == "fused" && seq_len >=
/// mm_min_seq && !no_mm_id`. `attn_dtype` is the attention weight dtype the
/// model resolved at load ("f16" default, "f32" under LAGUNA_ATTN_F32) — the
/// gate enforces it per side/tier, so a dump from a binary that predates the
/// f16 attention path (and thus omits the field) cannot pass as current.
/// `combine` records the routed-expert combine path: "reference" for the
/// Reference-oracle runner (which never touches `ops::combine` — it combines via
/// its own per-expert index_add), else "fused" (default) or "classic" (under
/// LAGUNA_COMBINE_CLASSIC) for the fused runner. The gate enforces it per
/// side/tier (like `attn_dtype`), so a dump predating the field cannot pass as
/// current. Additive: readers that ignore it still parse older/newer dumps.
fn provenance(model: &LagunaModel, moe_impl: &str, seq_len: usize) -> Value {
    let attn_dtype = match model.attn_dtype() {
        DType::F32 => "f32",
        DType::F16 => "f16",
        other => unreachable!("attention computes in f16 or f32, not {other:?}"),
    };
    json!({
        "moe_impl": moe_impl,
        "seq_len": seq_len,
        "mm_variant": laguna::ops::active_mm_variant_name(),
        "no_mm_id": laguna::ops::no_mm_id_forced(),
        "mm_min_seq": laguna::ops::MM_ID_MIN_SEQ,
        "attn_dtype": attn_dtype,
        // Reference runner never dispatches ops::combine, so it is neither
        // "fused" nor "classic" — mirror how moe_impl distinguishes reference.
        "combine": if matches!(moe_impl, "reference" | "ref") {
            "reference"
        } else if laguna::ops::combine_classic() {
            "classic"
        } else {
            "fused"
        },
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let runner = expert_runner(&cli.moe_impl)?;

    let device = gguf::metal_device()?;
    let gguf = gguf::open(&cli.model, &device)?;
    let cfg = LagunaConfig::from_gguf(&gguf.content)?;
    let vocab = cfg.vocab;
    // `top2` (used by the greedy/replay decode dumps) indexes the top-2 logits;
    // a degenerate vocab would panic there. Fail with a clear error instead.
    anyhow::ensure!(vocab >= 2, "vocab {vocab} < 2: cannot form a top-2 for the parity dumps");
    let model = LagunaModel::load(gguf, runner, cli.max_ctx)?;

    if let Some(corpus) = cli.ppl.clone() {
        anyhow::ensure!(
            cli.greedy.is_none() && cli.replay.is_none(),
            "--ppl is mutually exclusive with --greedy/--replay"
        );
        return run_ppl(&cli, model, &device, vocab, &corpus);
    }

    match (cli.greedy, &cli.replay) {
        (Some(_), Some(_)) => anyhow::bail!("--greedy and --replay are mutually exclusive"),
        (Some(n), None) => run_greedy(&cli, model, &device, vocab, n),
        (None, Some(path)) => run_replay(&cli, model, &device, vocab, &path.clone()),
        (None, None) => run_single(&cli, model, &device, vocab),
    }
}

/// Prompt chunk size for the perplexity pass — matches `generate::PREFILL_CHUNK`
/// so the fused side exercises the same 512-token mm_id prefill kernel the real
/// generate path uses (and `>= MM_ID_MIN_SEQ`, so the mm_id path is active).
const PPL_CHUNK: usize = 512;

/// FNV-1a 64-bit over the little-endian token bytes: a stable, dependency-free
/// digest of the exact scored token stream, so the gate can re-verify alignment
/// even against a stored reference dump whose full `tokens` array was trimmed.
fn token_hash(tokens: &[u32]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    format!("{h:016x}")
}

/// `--ppl <corpus>`: single continuous chunked-prefill pass over the corpus,
/// gathering the next-token log-probability at every position. Perplexity is
/// blind to argmax flips (the greedy gate's job) but sensitive to how the fused
/// path reshapes the whole distribution's tails, so it is the scale-sensitive
/// complement to the decode greedy gate. Both the Reference and Fused runners
/// use this identical protocol, so protocol quirks cancel in the fused−reference
/// delta the gate bounds.
///
/// Scoring convention: `logits[p]` predicts `tokens[p+1]`; we score every
/// position `p` in `0..T-1` (each target is a real corpus token). The BOS token
/// (`tokens[0]`) is never itself a target — there is no position predicting it —
/// so it drops out naturally. The final position has no successor and is skipped.
fn run_ppl(cli: &Cli, mut model: LagunaModel, device: &Device, vocab: usize, corpus: &std::path::Path) -> Result<()> {
    let tokenizer = LagunaTokenizer::from_file(&cli.tokenizer)
        .with_context(|| format!("loading tokenizer {}", cli.tokenizer.display()))?;
    let text = std::fs::read_to_string(corpus)
        .with_context(|| format!("reading corpus {}", corpus.display()))?;

    // add_special_tokens=false (the crate default), one leading BOS prepended by
    // hand — the standard LM-perplexity setup, and identical on both runners.
    let mut tokens = vec![LagunaTokenizer::BOS];
    tokens.extend(tokenizer.encode(&text)?);
    let n_tokens = tokens.len();
    anyhow::ensure!(n_tokens >= 2, "corpus tokenized to {n_tokens} tokens: need at least 2 to score one prediction");
    anyhow::ensure!(
        n_tokens <= model.max_ctx(),
        "corpus is {n_tokens} tokens but max_ctx is {} — raise --max-ctx to cover the whole corpus \
         (the pass is one continuous context, never truncated)",
        model.max_ctx()
    );

    // Continuous pass: feed 512-token chunks with a monotonically advancing
    // absolute position and never reset the KV cache (the model was just loaded,
    // so it starts empty). Positions are one unbroken context across chunks.
    let mut logprobs: Vec<f64> = Vec::with_capacity(n_tokens.saturating_sub(1));
    let mut per_chunk_means: Vec<f64> = Vec::new();
    let mut nonfinite: u64 = 0;
    let mut pos = 0usize;
    for chunk in tokens.chunks(PPL_CHUNK) {
        let input = Tensor::new(chunk, device)?;
        let chunk_logits = model.forward_all_logits(&input, pos)?; // [chunk, vocab]
        let host = chunk_logits.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;

        let mut chunk_sum = 0.0f64;
        let mut chunk_scored = 0usize;
        for i in 0..chunk.len() {
            let p = pos + i; // absolute position whose logits predict tokens[p+1]
            if p + 1 >= n_tokens {
                break; // final corpus position: no successor to score
            }
            let target = tokens[p + 1] as usize;
            let row = &host[i * vocab..(i + 1) * vocab];
            let lp = target_logprob(row, target);
            if lp.is_finite() {
                logprobs.push(lp);
                chunk_sum += lp;
                chunk_scored += 1;
            } else {
                nonfinite += 1;
            }
        }
        if chunk_scored > 0 {
            per_chunk_means.push(-chunk_sum / chunk_scored as f64);
        }
        pos += chunk.len();
    }

    let n_scored = logprobs.len();
    anyhow::ensure!(n_scored > 0, "no positions scored (corpus too short?)");
    let mean_nll = -logprobs.iter().sum::<f64>() / n_scored as f64;

    let seq_len = n_tokens.min(PPL_CHUNK); // the prefill chunk length the runner actually saw
    let dump = json!({
        "kind": "ppl",
        "model": cli.model.display().to_string(),
        "corpus": corpus.display().to_string(),
        "moe_impl": cli.moe_impl,
        "provenance": provenance(&model, &cli.moe_impl, seq_len),
        "tokens": tokens,
        "n_tokens": n_tokens,
        "token_hash": token_hash(&tokens),
        "n_scored": n_scored,
        "vocab": vocab,
        "nonfinite": nonfinite,
        "mean_nll": mean_nll,
        "per_chunk_means": per_chunk_means,
    });
    std::fs::write(&cli.output, serde_json::to_string(&dump)?)
        .with_context(|| format!("writing {}", cli.output.display()))?;
    eprintln!(
        "wrote {} (ppl, runner {}, {} tokens, {} scored, mean_nll {:.6}, {} nonfinite)",
        cli.output.display(),
        cli.moe_impl,
        n_tokens,
        n_scored,
        mean_nll,
        nonfinite,
    );
    Ok(())
}

/// Next-token log-probability `log_softmax(row)[target]` in f64. A numerically
/// stable logsumexp (subtract the row max). Returns a non-finite value if the
/// row itself contains non-finite logits, so the caller counts and excludes it.
fn target_logprob(row: &[f32], target: usize) -> f64 {
    let mut max = f32::NEG_INFINITY;
    for &x in row {
        if x > max {
            max = x;
        }
    }
    if !max.is_finite() {
        return f64::NAN;
    }
    let m = max as f64;
    let mut sumexp = 0.0f64;
    for &x in row {
        sumexp += (x as f64 - m).exp();
    }
    let lse = m + sumexp.ln();
    row[target] as f64 - lse
}

/// Default mode: one forward pass, dump the full last-position logits + taps.
fn run_single(cli: &Cli, mut model: LagunaModel, device: &Device, vocab: usize) -> Result<()> {
    let tokens = parse_tokens(cli.tokens.as_deref().context("--tokens is required")?)?;
    anyhow::ensure!(!tokens.is_empty(), "no token ids parsed from --tokens");
    if cli.taps {
        model.set_tap_capture(true);
    }

    let input = Tensor::new(tokens.as_slice(), device)?;
    let logits_t = model.forward(&input, 0)?;
    let logits = logits_to_host(&logits_t)?;

    let top5 = topk(&logits, 5);
    let top1 = top5.first().map(|&(id, _)| id).unwrap_or(0);
    let top5_json: Vec<Value> = top5.iter().map(|&(id, v)| json!([id, v])).collect();

    let taps: Vec<Value> = if cli.taps {
        model.take_taps().iter().map(|(name, t)| tap_value(name, t)).collect::<Result<_>>()?
    } else {
        Vec::new()
    };

    let dump = json!({
        "model": cli.model.display().to_string(),
        "prompt": cli.prompt,
        "moe_impl": cli.moe_impl,
        "provenance": provenance(&model, &cli.moe_impl, tokens.len()),
        "tokens": tokens,
        "n_tokens": tokens.len(),
        "vocab": vocab,
        "logits": logits,
        "top1": top1,
        "top5": top5_json,
        "taps": taps,
    });

    std::fs::write(&cli.output, serde_json::to_string(&dump)?)
        .with_context(|| format!("writing {}", cli.output.display()))?;
    eprintln!(
        "wrote {} ({} tokens, vocab {}, {} taps) -> top1={}",
        cli.output.display(),
        tokens.len(),
        vocab,
        taps.len(),
        top1
    );
    Ok(())
}

/// Prefill `prompt` in one forward at position 0 and return the last-position
/// logits plus the next decode position. Mirrors `run_single`'s single-shot
/// prefill so the first decode step sees the same logits the strict gate does.
fn prefill(model: &mut LagunaModel, device: &Device, prompt: &[u32]) -> Result<(Vec<f32>, usize)> {
    anyhow::ensure!(!prompt.is_empty(), "empty prompt");
    let input = Tensor::new(prompt, device)?;
    let logits = logits_to_host(&model.forward(&input, 0)?)?;
    Ok((logits, prompt.len()))
}

/// `--greedy N`: free-run greedy decode, recording at each step the token
/// produced (argmax) and the top-1/top-2 of the logits that produced it. Runs
/// the full N steps regardless of EOG so the gate compares equal lengths.
fn run_greedy(cli: &Cli, mut model: LagunaModel, device: &Device, vocab: usize, n: usize) -> Result<()> {
    let tokens = parse_tokens(cli.tokens.as_deref().context("--tokens is required for --greedy")?)?;
    let (mut logits, mut pos) = prefill(&mut model, device, &tokens)?;

    let mut steps: Vec<Value> = Vec::with_capacity(n);
    for i in 0..n {
        let (t1, t2) = top2(&logits);
        let token = t1.0;
        let (l2, nonfinite) = logit_scale(&logits);
        steps.push(json!({
            "token": token,
            "top1": [t1.0, t1.1],
            "top2": [t2.0, t2.1],
            "l2": l2,
            "nonfinite": nonfinite,
        }));
        // Skip the trailing forward: the logits after the last emitted token are
        // never inspected.
        if i + 1 < n {
            let input = Tensor::new(&[token], device)?;
            logits = logits_to_host(&model.forward(&input, pos)?)?;
            pos += 1;
        }
    }

    let dump = json!({
        "kind": "greedy",
        "model": cli.model.display().to_string(),
        "prompt": cli.prompt,
        "moe_impl": cli.moe_impl,
        "provenance": provenance(&model, &cli.moe_impl, tokens.len()),
        "tokens": tokens,
        "n_tokens": tokens.len(),
        "vocab": vocab,
        "steps": steps,
    });
    std::fs::write(&cli.output, serde_json::to_string(&dump)?)
        .with_context(|| format!("writing {}", cli.output.display()))?;
    eprintln!(
        "wrote {} (greedy, {} prompt tokens, {} steps, runner {})",
        cli.output.display(),
        tokens.len(),
        n,
        cli.moe_impl
    );
    Ok(())
}

/// `--replay <greedy-dump>`: teacher-force the dump's step tokens, recording at
/// each step THIS runner's own argmax (top-1/top-2) BEFORE forcing the
/// reference token. The prompt is the greedy dump's prompt.
fn run_replay(cli: &Cli, mut model: LagunaModel, device: &Device, vocab: usize, dump_path: &std::path::Path) -> Result<()> {
    let text = std::fs::read_to_string(dump_path)
        .with_context(|| format!("reading greedy dump {}", dump_path.display()))?;
    let ref_dump: Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing greedy dump {}", dump_path.display()))?;
    anyhow::ensure!(
        ref_dump["kind"].as_str() == Some("greedy"),
        "--replay expects a kind:\"greedy\" dump, got kind={:?}",
        ref_dump["kind"]
    );

    let prompt: Vec<u32> = ref_dump["tokens"]
        .as_array()
        .context("greedy dump missing `tokens`")?
        .iter()
        .map(|x| {
            let n = x.as_u64().context("non-integer prompt token")?;
            u32::try_from(n).with_context(|| format!("prompt token {n} exceeds u32"))
        })
        .collect::<Result<_>>()?;
    let ref_steps = ref_dump["steps"].as_array().context("greedy dump missing `steps`")?;

    let (mut logits, mut pos) = prefill(&mut model, device, &prompt)?;

    let mut steps: Vec<Value> = Vec::with_capacity(ref_steps.len());
    for (i, step) in ref_steps.iter().enumerate() {
        let forced_raw = step["token"].as_u64().with_context(|| format!("step {i} missing `token`"))?;
        let forced = u32::try_from(forced_raw).with_context(|| format!("step {i} token {forced_raw} exceeds u32"))?;
        let (t1, t2) = top2(&logits);
        let (l2, nonfinite) = logit_scale(&logits);
        steps.push(json!({
            "top1": [t1.0, t1.1],
            "top2": [t2.0, t2.1],
            "forced_token": forced,
            "l2": l2,
            "nonfinite": nonfinite,
        }));
        // Force the reference token to keep the two sequences aligned. The
        // trailing force is still executed only when another step follows.
        if i + 1 < ref_steps.len() {
            let input = Tensor::new(&[forced], device)?;
            logits = logits_to_host(&model.forward(&input, pos)?)?;
            pos += 1;
        }
    }

    let dump = json!({
        "kind": "replay",
        "model": cli.model.display().to_string(),
        "prompt": cli.prompt,
        "moe_impl": cli.moe_impl,
        "provenance": provenance(&model, &cli.moe_impl, prompt.len()),
        "tokens": prompt,
        "n_tokens": prompt.len(),
        "vocab": vocab,
        "steps": steps,
    });
    std::fs::write(&cli.output, serde_json::to_string(&dump)?)
        .with_context(|| format!("writing {}", cli.output.display()))?;
    eprintln!(
        "wrote {} (replay of {}, {} prompt tokens, {} steps, runner {})",
        cli.output.display(),
        dump_path.display(),
        prompt.len(),
        ref_steps.len(),
        cli.moe_impl
    );
    Ok(())
}
