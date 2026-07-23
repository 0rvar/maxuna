//! Microbench for the speculative-verify forward: checkpoint → span forward
//! (`forward_all_logits`) → rollback, at a sweep of span lengths, over real
//! fixture tokens. This is the exact per-round verify operation generate_spec
//! performs, so it isolates the cost the drafter must amortize — and, run twice
//! with `LAGUNA_MM_ID_MIN_SEQ` toggled, it measures the mv_id/mm_id crossover
//! for short spans (mm_id's per-expert compaction is the "expert union" dedup;
//! mv_id re-reads each routed expert per token).
//!
//! Usage:
//!   spec-verify-bench --model <gguf> [--n-past 512] [--reps 20]
//!   LAGUNA_MM_ID_MIN_SEQ=2 spec-verify-bench --model <gguf>   # force mm_id

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use laguna::gguf;
use laguna::model::LagunaModel;
use laguna::ops::ExpertRunner;
use laguna::LagunaConfig;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    model: PathBuf,
    /// Tokens prefilled before the verify sweep (cache depth during the bench;
    /// > 512 exercises the wrapped SWA ring like a real mid-generation verify).
    #[arg(long, default_value_t = 512)]
    n_past: usize,
    #[arg(long, default_value_t = 20)]
    reps: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = gguf::metal_device()?;
    let file = gguf::open(&args.model, &device)?;
    let _cfg = LagunaConfig::from_gguf(&file.content)?;
    let mut model = LagunaModel::load(file, ExpertRunner::Fused, 4096)?;

    let fixture: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/parity-prompts.json"),
    )?)?;
    let tokens: Vec<u32> = fixture["prompts"]
        .as_array()
        .context("prompts array")?
        .iter()
        .find(|p| p["id"] == "long-swa")
        .context("long-swa fixture")?["tokens"]
        .as_array()
        .context("tokens array")?
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect();
    let max_span = 48;
    anyhow::ensure!(tokens.len() >= args.n_past + max_span, "fixture too short");

    eprintln!(
        "effective mm_id_min_seq = {} (default {})",
        laguna::ops::mm_id_min_seq(),
        laguna::ops::MM_ID_MIN_SEQ
    );

    // Prefill the context the sweep runs on top of (also serves as the
    // steady-state warm-up: weights resident, pipelines compiled).
    let ctx = candle_core::Tensor::from_vec(tokens[..args.n_past].to_vec(), args.n_past, &device)?;
    let _ = model.forward(&ctx, 0)?;
    device.synchronize()?;

    println!("span\tms/verify\ttok/s-equiv");
    for span in [2usize, 4, 6, 8, 12, 16, 24, 32, 48] {
        let ids = tokens[args.n_past..args.n_past + span].to_vec();
        let vinput = candle_core::Tensor::from_vec(ids, span, &device)?;

        // Warm-up reps outside the timed window.
        for _ in 0..3 {
            let ckpt = model.kv_checkpoint(span)?;
            let logits = model.forward_all_logits(&vinput, args.n_past)?;
            drop(logits);
            model.kv_rollback(&ckpt, 0)?;
        }
        device.synchronize()?;

        let t0 = std::time::Instant::now();
        for _ in 0..args.reps {
            let ckpt = model.kv_checkpoint(span)?;
            // Match generate_spec's real readback: the sampler walks CPU logits.
            let logits = model.forward_all_logits(&vinput, args.n_past)?;
            let _cpu = logits
                .to_dtype(candle_core::DType::F32)?
                .to_device(&candle_core::Device::Cpu)?;
            model.kv_rollback(&ckpt, 0)?;
        }
        device.synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / args.reps as f64;
        println!("{span}\t{ms:.2}\t{:.1}", span as f64 / (ms / 1000.0));
    }
    Ok(())
}
