mod repl;

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

use laguna::chat::{ChatOptions, Message, build_prompt};
use laguna::generate::Generator;
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
        #[command(flatten)]
        sampling: SamplingArgs,
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

/// Load the model + tokenizer + sampler and assemble a Generator on Metal.
fn build_generator(
    model: &PathBuf,
    tokenizer: &PathBuf,
    moe_impl: &str,
    max_ctx: usize,
    sampling: SamplerOptions,
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

    Ok(Generator::new(model, tok, sampler))
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
            sampling,
        } => {
            let mut generator =
                build_generator(&model, &tokenizer, &moe_impl, max_ctx, sampling.options())?;

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
            }
            Ok(())
        }
        Cmd::Chat { model, tokenizer, max_tokens, moe_impl, max_ctx, show_thinking, sampling } => {
            let mut generator =
                build_generator(&model, &tokenizer, &moe_impl, max_ctx, sampling.options())?;
            repl::run(&mut generator, max_tokens, show_thinking)
        }
    }
}
