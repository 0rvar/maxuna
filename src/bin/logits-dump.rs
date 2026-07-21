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
    #[arg(short, long)]
    tokens: String,

    /// Optional prompt text, recorded in the dump for provenance only (this
    /// tool never tokenizes — feed ids via --tokens so both sides agree).
    #[arg(short, long)]
    prompt: Option<String>,

    /// Also capture the per-layer intermediate taps (the fork cb() names).
    #[arg(long)]
    taps: bool,

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

fn main() -> Result<()> {
    let cli = Cli::parse();
    let tokens = parse_tokens(&cli.tokens)?;
    anyhow::ensure!(!tokens.is_empty(), "no token ids parsed from --tokens");
    let runner = expert_runner(&cli.moe_impl)?;

    let device = gguf::metal_device()?;
    let gguf = gguf::open(&cli.model, &device)?;
    let cfg = LagunaConfig::from_gguf(&gguf.content)?;
    let vocab = cfg.vocab;

    let mut model = LagunaModel::load(gguf, runner, cli.max_ctx)?;
    if cli.taps {
        model.set_tap_capture(true);
    }

    let input = Tensor::new(tokens.as_slice(), &device)?;
    let logits_t = model.forward(&input, 0)?;
    let logits = logits_t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;

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
