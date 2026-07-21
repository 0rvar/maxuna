use anyhow::{Context, Result, bail};
use candle_core::quantized::gguf_file::{Content, Value};

/// RoPE parameters for one attention layer type.
#[derive(Debug, Clone)]
pub enum RopeKind {
    /// Full-attention layers: YaRN-scaled, partial rotary (n_rot < head_dim).
    Yarn {
        freq_base: f32,
        factor: f32,
        original_ctx: usize,
        beta_fast: f32,
        beta_slow: f32,
        /// mscale applied to cos/sin (config.json `attention_factor`).
        attn_factor: f32,
        n_rot: usize,
    },
    /// SWA layers: unscaled rope over the full head_dim.
    Plain { freq_base: f32, n_rot: usize },
}

#[derive(Debug, Clone)]
pub struct LagunaConfig {
    pub n_layer: usize,
    pub hidden: usize,
    pub vocab: usize,
    /// Per-layer query-head counts (S 2.1: 48 on full-attention layers, 72 on SWA).
    pub n_head: Vec<usize>,
    pub n_kv_head: usize,
    pub head_dim: usize,
    /// Leading dense-MLP layers (before MoE starts). S 2.1: 1.
    pub dense_layers: usize,
    /// Dense-MLP intermediate size (layer 0).
    pub dense_ff: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub expert_ff: usize,
    pub shared_expert_ff: usize,
    /// Routed-expert output scale (S 2.1: 2.5), applied after top-k normalization.
    pub expert_weights_scale: f64,
    /// Sum-normalize the top-k routing weights before scaling.
    pub expert_weights_norm: bool,
    pub rms_eps: f64,
    pub sliding_window: usize,
    /// Layer `il` is full-attention iff `il % swa_period == 0`.
    pub swa_period: usize,
    pub n_ctx_train: usize,
    pub rope_full: RopeKind,
    pub rope_swa: RopeKind,
    pub bos_token: u32,
    /// All end-of-generation tokens (S 2.1: 2 = EOS, 24 = `</assistant>`).
    pub eog_tokens: Vec<u32>,
}

impl LagunaConfig {
    pub fn is_full_attn(&self, il: usize) -> bool {
        il % self.swa_period == 0
    }

    pub fn n_head(&self, il: usize) -> usize {
        self.n_head[il]
    }

    pub fn rope(&self, il: usize) -> &RopeKind {
        if self.is_full_attn(il) { &self.rope_full } else { &self.rope_swa }
    }

    pub fn from_gguf(content: &Content) -> Result<Self> {
        let md = Meta(content);
        let arch = md.str("general.architecture")?;
        if arch != "laguna" {
            bail!("expected a laguna GGUF, got architecture {arch:?}");
        }

        let n_layer = md.usize("laguna.block_count")?;
        let head_dim = md.usize("laguna.attention.key_length")?;
        let n_rot_full = md.usize_or("laguna.rope.dimension_count", head_dim / 2);
        let n_rot_swa = md.usize_or("laguna.rope.dimension_count_swa", head_dim);

        // The official Q4_K_M GGUF is a 256k-context conversion: factor 32 over
        // an 8192 original context (the HF 1M checkpoint config uses factor 128).
        let factor = md.f32_or("laguna.rope.scaling.factor", 32.0);
        // Net cos/sin magnitude scale, matching the fork's llama-context.cpp
        // cparams dance + ggml rope_yarn's internal mscale: the two
        // (1 + 0.1*ln(factor)) terms cancel, leaving get_mscale(factor) times the
        // GGUF's rope.scaling.attn_factor (absent here, default 1.0). The
        // `rope.scaling.yarn_attn_factor` key in the file is a model-saver
        // artifact the fork never reads for laguna.
        let attn_factor =
            (1.0 + 0.1 * factor.max(1.0).ln()) * md.f32_or("laguna.rope.scaling.attn_factor", 1.0);
        let rope_full = RopeKind::Yarn {
            freq_base: md.f32_or("laguna.rope.freq_base", 500_000.0),
            factor,
            original_ctx: md.usize_or("laguna.rope.scaling.original_context_length", 8192),
            beta_fast: md.f32_or("laguna.rope.scaling.yarn_beta_fast", 32.0),
            beta_slow: md.f32_or("laguna.rope.scaling.yarn_beta_slow", 1.0),
            attn_factor,
            n_rot: n_rot_full,
        };
        let rope_swa = RopeKind::Plain {
            freq_base: md.f32_or("laguna.rope.freq_base_swa", 10_000.0),
            n_rot: n_rot_swa,
        };

        let gating_func = md.usize_or("laguna.expert_gating_func", 2);
        if gating_func != 2 {
            bail!("only sigmoid expert gating (2) is supported, got {gating_func}");
        }

        let mut eog_tokens = vec![md.u32_or("tokenizer.ggml.eos_token_id", 2)];
        if let Ok(eot) = md.u32("tokenizer.ggml.eot_token_id") {
            eog_tokens.push(eot);
        }

        Ok(Self {
            n_layer,
            hidden: md.usize("laguna.embedding_length")?,
            vocab: md.usize("laguna.vocab_size")?,
            n_head: md.usize_per_layer("laguna.attention.head_count", n_layer)?,
            n_kv_head: md.usize("laguna.attention.head_count_kv")?,
            head_dim,
            dense_layers: md.usize_or("laguna.leading_dense_block_count", 1),
            dense_ff: md.usize("laguna.feed_forward_length")?,
            n_expert: md.usize("laguna.expert_count")?,
            n_expert_used: md.usize("laguna.expert_used_count")?,
            expert_ff: md.usize("laguna.expert_feed_forward_length")?,
            shared_expert_ff: md.usize_or("laguna.expert_shared_feed_forward_length", 1024),
            expert_weights_scale: md.f32_or("laguna.expert_weights_scale", 2.5) as f64,
            expert_weights_norm: md.bool_or("laguna.expert_weights_norm", true),
            rms_eps: md.f32_or("laguna.attention.layer_norm_rms_epsilon", 1e-6) as f64,
            sliding_window: md.usize_or("laguna.attention.sliding_window", 512),
            swa_period: md.usize_or("laguna.attention.sliding_window_pattern", 4),
            n_ctx_train: md.usize("laguna.context_length")?,
            rope_full,
            rope_swa,
            bos_token: md.u32_or("tokenizer.ggml.bos_token_id", 2),
            eog_tokens,
        })
    }
}

/// Tolerant typed accessors over GGUF metadata.
struct Meta<'a>(&'a Content);

impl Meta<'_> {
    fn get(&self, key: &str) -> Result<&Value> {
        self.0.metadata.get(key).with_context(|| format!("missing GGUF key {key}"))
    }

    fn str(&self, key: &str) -> Result<&str> {
        Ok(self.get(key)?.to_string()?.as_str())
    }

    fn usize(&self, key: &str) -> Result<usize> {
        value_as_usize(self.get(key)?).with_context(|| format!("GGUF key {key} is not a non-negative integer"))
    }

    fn u32(&self, key: &str) -> Result<u32> {
        Ok(self.usize(key)? as u32)
    }

    fn f32(&self, key: &str) -> Result<f32> {
        let v = self.get(key)?;
        v.to_f32().or_else(|_| v.to_f64().map(|v| v as f32)).with_context(|| format!("GGUF key {key} is not a float"))
    }

    fn bool(&self, key: &str) -> Result<bool> {
        Ok(self.get(key)?.to_bool()?)
    }

    fn usize_or(&self, key: &str, default: usize) -> usize {
        self.usize(key).unwrap_or(default)
    }

    fn u32_or(&self, key: &str, default: u32) -> u32 {
        self.u32(key).unwrap_or(default)
    }

    fn f32_or(&self, key: &str, default: f32) -> f32 {
        self.f32(key).unwrap_or(default)
    }

    fn bool_or(&self, key: &str, default: bool) -> bool {
        self.bool(key).unwrap_or(default)
    }

    /// A per-layer array key, expanding a scalar to a uniform vec.
    fn usize_per_layer(&self, key: &str, n_layer: usize) -> Result<Vec<usize>> {
        match self.get(key)? {
            Value::Array(vals) => {
                let out: Vec<usize> = vals
                    .iter()
                    .map(value_as_usize)
                    .collect::<Result<_>>()
                    .with_context(|| format!("GGUF key {key} has non-integer entries"))?;
                if out.len() != n_layer {
                    bail!("GGUF key {key} has {} entries, expected {n_layer}", out.len());
                }
                Ok(out)
            }
            _ => Ok(vec![self.usize(key)?; n_layer]),
        }
    }
}

/// GGUF writers emit integers as any width and signedness (the per-layer head
/// counts are i32, most scalars u32); accept them all as long as they fit.
fn value_as_usize(v: &Value) -> Result<usize> {
    if let Ok(u) = v.to_u64() {
        return Ok(u as usize);
    }
    if let Ok(u) = v.to_u32() {
        return Ok(u as usize);
    }
    let i = v.to_i64().or_else(|_| v.to_i32().map(i64::from))?;
    if i < 0 {
        bail!("negative integer {i}");
    }
    Ok(i as usize)
}
