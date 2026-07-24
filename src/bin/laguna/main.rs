mod repl;

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

use laguna::chat::{ChatOptions, Message, build_prompt};
use laguna::dflash::DflashDrafter;
use laguna::generate::{Generator, SpecParams};
use laguna::gguf;
use laguna::model::LagunaModel;
use laguna::ops::ExpertRunner;
use laguna::sampler::{Sampler, SamplerOptions};
use laguna::tokenizer::LagunaTokenizer;
use laguna::LagunaConfig;

/// BOS token text (tokenizer added-token id 2). The chat template emits this
/// verbatim; in `--raw` mode the CLI prepends it so raw prompts still open with
/// a BOS, matching llama.cpp's default `add_bos`.
const BOS: &str = "\u{3008}|EOS|\u{3009}";

/// Default tokenizer shipped alongside every checkpoint (from_gguf intentionally
/// errors, so the vocab is loaded from this JSON).
const DEFAULT_TOKENIZER: &str = "reference/tokenizer.json";

#[derive(Parser)]
#[command(name = "laguna", about = "Laguna S 2.1 inference on Metal")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

/// Shared sampling knobs (generation_config.json defaults: temp 1.0, top-k 20).
#[derive(Parser)]
struct SamplingArgs {
    #[arg(long, default_value_t = 1.0)]
    temp: f64,
    #[arg(long, default_value_t = 20)]
    top_k: usize,
    #[arg(long, default_value_t = 1.0)]
    top_p: f64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

impl SamplingArgs {
    fn options(&self) -> SamplerOptions {
        SamplerOptions { temperature: self.temp, top_k: self.top_k, top_p: self.top_p, seed: self.seed }
    }
}

/// DFlash speculative-decode knobs. Passing `--draft <gguf>` turns speculation
/// on; without it, generation runs the plain single-token decode loop.
#[derive(Parser)]
struct DraftArgs {
    /// Path to a DFlash drafter GGUF. Enables speculative decoding when set.
    #[arg(long)]
    draft: Option<PathBuf>,
    /// Max draft tokens proposed per verify round (clamped to block_size-1).
    #[arg(long, default_value_t = 15)]
    draft_max: usize,
    /// Discard a round's whole draft if fewer than this many are collected.
    #[arg(long, default_value_t = 0)]
    draft_min: usize,
    /// Stop drafting at the first token whose full-vocab softmax prob is below
    /// this. Adaptive draft length; 0.5 measured best across prompt kinds.
    #[arg(long, default_value_t = 0.5)]
    draft_p_min: f32,
    /// Auto-pause speculation when its wall-clock cost per committed token
    /// exceeds a plain decode step's cost times this factor (keeps `--draft`
    /// from losing to plain decode on low-acceptance text). With auto-pause on,
    /// temperature>0 runs are not run-to-run reproducible for a fixed seed
    /// (which rounds batch-verify depends on wall-clock timing, and batched
    /// rounds differ from plain at near-ties); `0` disables auto-pause (always
    /// draft) and restores fully deterministic fixed-seed behavior.
    #[arg(long, default_value_t = 1.0)]
    draft_pause_margin: f32,
}

impl DraftArgs {
    fn params(&self) -> SpecParams {
        SpecParams {
            draft_max: self.draft_max,
            draft_min: self.draft_min,
            draft_p_min: self.draft_p_min,
            pause_margin: self.draft_pause_margin,
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Dump GGUF metadata and tensor listing.
    Inspect {
        #[arg(short, long)]
        model: PathBuf,
    },
    /// One-shot generation from a prompt.
    Generate {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long)]
        prompt: String,
        #[arg(long, default_value = DEFAULT_TOKENIZER)]
        tokenizer: PathBuf,
        #[arg(short = 'n', long, default_value_t = 512)]
        max_tokens: usize,
        #[arg(long, default_value = "fused")]
        moe_impl: String,
        #[arg(long, default_value_t = 8192)]
        max_ctx: usize,
        /// Skip the chat template and feed the prompt raw (BOS prepended).
        #[arg(long)]
        raw: bool,
        #[arg(long)]
        stats: bool,
        /// Force at least this many decode tokens of <think> reasoning before
        /// `</think>` (or an EOG) may be sampled; 0 lets the model decide.
        /// Meaningful only without --raw (the chat template ends the prompt
        /// inside an open <think> block).
        #[arg(long, default_value_t = 0)]
        min_think: usize,
        #[command(flatten)]
        sampling: SamplingArgs,
        #[command(flatten)]
        draft: DraftArgs,
    },
    /// Interactive chat REPL.
    Chat {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(long, default_value = DEFAULT_TOKENIZER)]
        tokenizer: PathBuf,
        #[arg(short = 'n', long, default_value_t = 2048)]
        max_tokens: usize,
        #[arg(long, default_value = "fused")]
        moe_impl: String,
        #[arg(long, default_value_t = 8192)]
        max_ctx: usize,
        /// Show the model's <think> reasoning (dimmed) instead of hiding it.
        #[arg(long)]
        show_thinking: bool,
        /// Force at least this many decode tokens of <think> reasoning per
        /// turn before `</think>` (or an EOG) may be sampled; 0 lets the
        /// model decide (it skips reasoning on conversational prompts).
        #[arg(long, default_value_t = 0)]
        min_think: usize,
        #[command(flatten)]
        sampling: SamplingArgs,
    },
}

fn expert_runner(name: &str) -> Result<ExpertRunner> {
    match name {
        "reference" | "ref" => Ok(ExpertRunner::Reference),
        "fused" => Ok(ExpertRunner::Fused),
        other => bail!("unknown --moe-impl {other:?} (expected reference|fused)"),
    }
}

/// Load the model + tokenizer + sampler and assemble a Generator on Metal. When
/// `draft` carries a `--draft` path, the DFlash drafter is loaded on the SAME
/// Metal device (its ops interleave with the target's shared embeddings/lm_head,
/// so they must share a device) and attached for speculative decoding.
fn build_generator(
    model: &PathBuf,
    tokenizer: &PathBuf,
    moe_impl: &str,
    max_ctx: usize,
    sampling: SamplerOptions,
    draft: Option<&DraftArgs>,
) -> Result<Generator> {
    let runner = expert_runner(moe_impl)?;
    let device = gguf::metal_device()?;

    let load_start = std::time::Instant::now();
    let gguf = gguf::open(model, &device)?;
    let cfg = LagunaConfig::from_gguf(&gguf.content)?;
    let tok = LagunaTokenizer::from_file(tokenizer)?;
    let sampler = Sampler::new(sampling, cfg.eog_tokens.clone());
    let model = LagunaModel::load(gguf, runner, max_ctx)?;
    eprintln!("laguna: model loaded in {:.1}s", load_start.elapsed().as_secs_f64());

    let mut generator = Generator::new(model, tok, sampler);

    if let Some(draft) = draft.filter(|d| d.draft.is_some()) {
        let path = draft.draft.as_ref().unwrap();
        let draft_start = std::time::Instant::now();
        let dgguf = gguf::open(path, &device)?;
        let drafter = DflashDrafter::load(&dgguf, &device, max_ctx)?;
        generator.attach_drafter(drafter, draft.params())?;
        eprintln!("laguna: drafter loaded in {:.1}s", draft_start.elapsed().as_secs_f64());
    }

    Ok(generator)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Inspect { model } => {
            let device = candle_core::Device::Cpu;
            let gguf = gguf::open(&model, &device)?;
            print!("{}", gguf::describe(&gguf.content));
            let cfg = LagunaConfig::from_gguf(&gguf.content)?;
            println!("\nparsed config: {cfg:#?}");
            Ok(())
        }
        Cmd::Generate {
            model,
            prompt,
            tokenizer,
            max_tokens,
            moe_impl,
            max_ctx,
            raw,
            stats,
            min_think,
            sampling,
            draft,
        } => {
            let mut generator =
                build_generator(&model, &tokenizer, &moe_impl, max_ctx, sampling.options(), Some(&draft))?;
            generator.set_min_think(min_think);

            // BOS ownership: the chat template writes it as literal text; raw
            // mode prepends it here so generate() itself never injects a BOS.
            let text = if raw {
                format!("{BOS}{prompt}")
            } else {
                build_prompt(&[Message::User(prompt)], &ChatOptions { enable_thinking: true })?
            };

            let mut stdout = std::io::stdout();
            let gstats = generator.generate(
                &text,
                max_tokens,
                &mut |chunk| {
                    print!("{chunk}");
                    let _ = stdout.flush();
                },
                &mut || false,
            )?;
            println!();

            if stats {
                eprintln!(
                    "\nprefill: {} tokens in {:.2}s ({:.1} tok/s)\ndecode:  {} tokens in {:.2}s ({:.1} tok/s)",
                    gstats.prefill_tokens,
                    gstats.prefill_secs,
                    gstats.prefill_tps(),
                    gstats.decode_tokens,
                    gstats.decode_secs,
                    gstats.decode_tps(),
                );
                if let Some(spec) = &gstats.spec {
                    let paused = if spec.paused_rounds > 0 {
                        format!(" ({} paused)", spec.paused_rounds)
                    } else {
                        String::new()
                    };
                    eprintln!(
                        "spec:    {} rounds{paused}, {} drafted, {} accepted ({:.1}%), {} rejected",
                        spec.rounds,
                        spec.drafted,
                        spec.accepted,
                        spec.acceptance_rate() * 100.0,
                        spec.rejected(),
                    );
                    // Per-round averages divide by rounds (>= 1 here since a spec
                    // line only prints after decode ran); guard the zero case.
                    let rounds = spec.rounds.max(1) as f64;
                    eprintln!(
                        "         {} verified positions; draft {:.1}s ({:.0}ms/round), verify {:.1}s ({:.0}ms/round)",
                        spec.verify_positions,
                        spec.draft_ms / 1000.0,
                        spec.draft_ms / rounds,
                        spec.verify_ms / 1000.0,
                        spec.verify_ms / rounds,
                    );
                }
            }
            Ok(())
        }
        Cmd::Chat { model, tokenizer, max_tokens, moe_impl, max_ctx, show_thinking, min_think, sampling } => {
            let mut generator =
                build_generator(&model, &tokenizer, &moe_impl, max_ctx, sampling.options(), None)?;
            generator.set_min_think(min_think);
            repl::run(&mut generator, max_tokens, show_thinking)
        }
    }
}
