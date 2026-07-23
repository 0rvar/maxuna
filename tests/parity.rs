//! Full-logit parity gate: compare two logits-dump JSON files (see
//! `src/bin/logits-dump.rs` for the schema) on the criteria that matter for
//! token-level equivalence.
//!
//! This is the dump-vs-dump track. It gates our engine against a blessed
//! reference dump in the SAME JSON format — either a previously blessed run of
//! our own engine (regression guard) or, in WP8, a reference produced from the
//! llama.cpp fork. The cross-implementation track (our dump vs the fork's
//! truncated eval-callback trace, which localizes the first divergent layer)
//! lives in `scripts/parity.ts`, because eval-callback does not expose full
//! logit vectors.
//!
//! Ignored by default (needs real dumps). Run once dumps exist, selecting the
//! tier that matches how the candidate was produced (see docs/parity.md §3b):
//!   LAGUNA_PARITY_DIR=/tmp/ref-code LAGUNA_PARITY_TIER=strict \
//!     cargo test --test parity -- --ignored --nocapture
//! The directory must contain `candidate.json` and `reference.json`.
//!
//! Two-tier gate (docs/parity.md §3b), both require identical input token ids
//! and top-5 overlap >= 4/5:
//!   * strict (default; mv_id/decode candidate): cosine >= 0.999, top-1 match.
//!   * mm (mm_id prefill candidate): cosine >= 0.995, and top-1 matches OR is a
//!     genuine near-tie — candidate's top-1 is the reference's top-1/top-2 AND
//!     the reference's top-1/top-2 logit margin is < 0.5.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

const COS_MIN_STRICT: f64 = 0.999;
const COS_MIN_MM: f64 = 0.995;
const NEAR_TIE_MARGIN: f64 = 0.5;
const TOP5_OVERLAP_MIN: usize = 4;

/// Max allowed candidate/reference L2-norm ratio (and its reciprocal is the min):
/// the cosine + top-1 + top-5 checks are all scale-INVARIANT, so a uniformly
/// rescaled candidate (`candidate = c * reference`) sails through them at cosine
/// 1.0. This catches that. Calibrated on the code-short dump pairs in the parity
/// dir (2026-07-22): the worst same-prompt norm ratio across all mv_id/mm_id
/// configs was 1.0178 (a 1.78% drift), so ~10x headroom → 1.18 (see docs/parity.md
/// §3b). NOTE the decode tier's greedy gate is likewise scale-invariant (argmax);
/// this norm-ratio check and the proposed perplexity gate are the scale-sensitive
/// layers.
const NORM_RATIO_MAX: f64 = 1.18;

/// Max allowed |mean_NLL(fused) − mean_NLL(reference)| over the frozen ppl corpus
/// (docs/parity.md "Perplexity gate"). The greedy decode gate (`greedy_parity`)
/// catches argmax flips but is blind to a change that reshapes the distribution's
/// tails while keeping every argmax; the perplexity delta is that scale-sensitive
/// complement. Calibrate-then-freeze: `max(3 × measured |delta|, 0.002)` nats, the
/// 0.002 floor keeping a near-zero measured delta from setting an impossibly tight
/// bound. Frozen from the 2026-07-22 calibration (see docs/parity.md): the fused
/// (mm_id prefill) vs Reference-oracle delta on the 4386-token wikitext-2 corpus
/// was 0.001937 nats (reference 2.020392, fused 2.018455), so `max(3 × 0.001937,
/// 0.002)` = 0.005811, rounded up to a clean 0.006 (keeps the >=3x margin).
const PPL_NLL_DELTA_MAX: f64 = 0.006;

#[derive(Clone, Copy, PartialEq)]
enum Tier {
    Strict,
    Mm,
    /// Decode-kernel changes: the full-logit cosine/top-1/top-5 are reported as
    /// diagnostics only (every remaining decode lever reorders f32 accumulation,
    /// so the strict 0.999 cosine can never accept them). The real gate for this
    /// tier is greedy agreement vs the Reference oracle — see `greedy_parity`.
    Decode,
}

impl Tier {
    fn from_env() -> Result<Tier> {
        match std::env::var("LAGUNA_PARITY_TIER").as_deref() {
            Ok("mm") => Ok(Tier::Mm),
            Ok("decode") => Ok(Tier::Decode),
            Ok("strict") | Err(_) => Ok(Tier::Strict),
            Ok(other) => bail!("LAGUNA_PARITY_TIER must be `strict`, `mm`, or `decode`, got {other:?}"),
        }
    }
}

struct Dump {
    tokens: Vec<u32>,
    logits: Vec<f32>,
    top1: u32,
    /// (token id, logit) descending — the logits are needed for the near-tie margin.
    top5: Vec<(u32, f32)>,
    taps: Vec<Tap>,
    /// How the dump was produced (written by current `logits-dump`); `None` for
    /// older dumps that predate the field. The mm tier requires it (see `compare`).
    provenance: Option<Provenance>,
}

/// The `logits-dump` `provenance` object: enough to tell whether a candidate dump
/// actually exercised the mm_id prefill path it is being graded against, and
/// which attention weight dtype the run used.
///
/// Missing-field handling is VERSION-AWARE (`laguna::parity_schema`): each
/// `Option` field below that is `None` is resolved against the dump's
/// `schema_version` — a field missing from a dump that predates the field's
/// introduction takes its grandfather value (the dump stays valid; this is what
/// keeps cached/committed references usable across field additions), while a
/// field missing at/after its introduction is the hard "stale binary" fail.
struct Provenance {
    /// The field set this dump carries. Dumps written before versioning carry
    /// no `schema_version` and are version 1 (the baseline set: every field up
    /// to `attn_glue`, all required with no grandfathering).
    schema_version: u32,
    moe_impl: String,
    seq_len: usize,
    mm_variant: String,
    no_mm_id: bool,
    mm_min_seq: usize,
    /// Attention weight dtype: "f16" (the shipped default — f16 weights, f32
    /// activations) or "f32" (the legacy path, `LAGUNA_ATTN_F32`). `None` for
    /// dumps written by a
    /// `logits-dump` binary that predates the field — the gate hard-fails on
    /// that (a stale binary's dumps are otherwise indistinguishable from
    /// current ones).
    attn_dtype: Option<String>,
    /// Routed-expert combine path: "reference" (Reference oracle — never touches
    /// `ops::combine`), "classic" (`LAGUNA_COMBINE_CLASSIC`, the candle chain), or
    /// "fused" (the shipped vendored kernel). `None` for dumps predating the combine
    /// provenance field — the gate hard-fails on that per side (same reasoning as
    /// `attn_dtype`).
    combine: Option<String>,
    /// Attention prefill gemm path: "tensor" (the shipped Metal-4
    /// cooperative-tensor default), "classic" (the simdgroup kernel, under the
    /// `LAGUNA_ATTN_MM_CLASSIC` kill-switch), or "f32-bypass" (`LAGUNA_ATTN_F32`
    /// — attention runs the legacy dequant-f32 QMatMul, so the f16 library's mm
    /// branch never runs). `None` for dumps predating the field — the gate
    /// hard-fails on that per side (same reasoning as `attn_dtype`).
    attn_mm: Option<String>,
    /// Attention-glue path (fused softplus gate / permute-cast copies /
    /// partial-rotary rope): "fused" (the shipped vendored kernels) or "classic"
    /// (the candle chains, `LAGUNA_ATTN_GLUE_CLASSIC`). Env-derived for BOTH
    /// runners — the Reference oracle executes the glue too, anchored by the env
    /// pin — so reference dumps must record "classic". `None` for dumps predating
    /// the field — the gate hard-fails on that per side (same reasoning as
    /// `attn_dtype`).
    attn_glue: Option<String>,
    /// sdpa compute dtype: "f16" (the shipped kernel) or "f32" (the
    /// `LAGUNA_SDPA_F32` experiment hook). Introduced at schema version 2 with
    /// grandfather "f16" — every earlier binary ran the f16 sdpa kernel
    /// unconditionally, so a v1 dump missing the field resolves to "f16";
    /// missing at v2+ is the stale-binary hard fail.
    sdpa: Option<String>,
    /// Prefill attention kernel: "fused" (the vendored flash kernel, the Metal
    /// default) or "classic" (the candle sdpa chain, `LAGUNA_FLASH_CLASSIC`).
    /// Introduced at schema version 3 with grandfather "classic" — every
    /// earlier binary ran the candle sdpa prefill, so a v1/v2 dump missing the
    /// field resolves to "classic"; missing at v3+ is the stale-binary hard fail.
    flash: Option<String>,
    /// Routed-expert SwiGLU activation path: "fused" (the vendored `ops::silu_mul`
    /// kernel, the Metal default) or "classic" (the candle `silu(gate) * up` chain,
    /// `LAGUNA_ACT_CLASSIC`, which the Reference oracle also runs). Introduced at
    /// schema version 4 with grandfather "classic" — every earlier binary ran the
    /// candle chain, so a v1..v3 dump missing the field resolves to "classic";
    /// missing at v4+ is the stale-binary hard fail.
    act: Option<String>,
}

impl Provenance {
    /// Whether this dump ran the fused mm_id prefill path (the mm tier's premise):
    /// the fused runner, a prefill at/above the mm_id threshold, and mv_id not forced.
    fn mm_active(&self) -> bool {
        self.moe_impl == "fused" && self.seq_len >= self.mm_min_seq && !self.no_mm_id
    }
}

/// A per-layer intermediate tap (subset of the `logits-dump` tap schema that is
/// useful for a layer-by-layer diff): the whole-tensor `sum`, its `l2` norm,
/// and the full last-position feature row when the dump kept it.
struct Tap {
    name: String,
    sum: f64,
    l2: f64,
    last_row: Option<Vec<f32>>,
}

fn u32_array(v: &Value, key: &str) -> Result<Vec<u32>> {
    v[key]
        .as_array()
        .with_context(|| format!("`{key}` is not an array"))?
        .iter()
        .map(|x| {
            let n = x.as_u64().context("non-integer in array")?;
            u32::try_from(n).with_context(|| format!("value {n} exceeds u32 in `{key}`"))
        })
        .collect()
}

/// Parse the optional `provenance` object (current `logits-dump` writes it; older
/// dumps omit it entirely — `None`). Present-but-malformed is an error, not `None`,
/// so a truncated field can't silently downgrade to "legacy dump".
fn parse_provenance(v: &Value) -> Result<Option<Provenance>> {
    match v.get("provenance") {
        Some(p) if !p.is_null() => Ok(Some(Provenance {
            // Absent in dumps written before schema versioning: they are
            // version 1, the baseline field set — this default is what keeps
            // every pre-versioning reference dump valid. A version NEWER than
            // this test knows is a stale TEST binary — fail closed rather than
            // mis-resolve fields introduced after this build.
            schema_version: match p.get("schema_version") {
                None => 1,
                Some(sv) => {
                    let n = sv.as_u64().context("provenance `schema_version` is not an integer")?;
                    let n = u32::try_from(n)
                        .with_context(|| format!("provenance schema_version {n} exceeds u32"))?;
                    if n > laguna::parity_schema::PROVENANCE_SCHEMA_VERSION {
                        bail!(
                            "provenance schema_version {n} is newer than this gate's {} — rebuild \
                             the parity test binary",
                            laguna::parity_schema::PROVENANCE_SCHEMA_VERSION
                        );
                    }
                    n
                }
            },
            moe_impl: p["moe_impl"].as_str().context("provenance missing `moe_impl`")?.to_string(),
            seq_len: p["seq_len"].as_u64().context("provenance missing `seq_len`")? as usize,
            mm_variant: p["mm_variant"].as_str().context("provenance missing `mm_variant`")?.to_string(),
            no_mm_id: p["no_mm_id"].as_bool().context("provenance missing `no_mm_id`")?,
            mm_min_seq: p["mm_min_seq"].as_u64().context("provenance missing `mm_min_seq`")? as usize,
            // Absent in dumps from binaries predating the field (the gate hard-fails
            // on that later, per side); present-but-not-a-string is malformed.
            attn_dtype: match p.get("attn_dtype") {
                Some(d) => Some(d.as_str().context("provenance `attn_dtype` is not a string")?.to_string()),
                None => None,
            },
            // Absent in dumps predating the combine provenance field (the gate
            // hard-fails on that later, per side); present-but-not-a-string is malformed.
            combine: match p.get("combine") {
                Some(d) => Some(d.as_str().context("provenance `combine` is not a string")?.to_string()),
                None => None,
            },
            // Absent in dumps predating the attn_mm field (the gate hard-fails on
            // that later, per side); present-but-not-a-string is malformed.
            attn_mm: match p.get("attn_mm") {
                Some(d) => Some(d.as_str().context("provenance `attn_mm` is not a string")?.to_string()),
                None => None,
            },
            // Absent in dumps predating the attn_glue field (the gate hard-fails
            // on that later, per side); present-but-not-a-string is malformed.
            attn_glue: match p.get("attn_glue") {
                Some(d) => Some(d.as_str().context("provenance `attn_glue` is not a string")?.to_string()),
                None => None,
            },
            // Absent in schema-version-1 dumps (grandfathered to "f16" by the
            // version-aware check); present-but-not-a-string is malformed.
            sdpa: match p.get("sdpa") {
                Some(d) => Some(d.as_str().context("provenance `sdpa` is not a string")?.to_string()),
                None => None,
            },
            // Absent in schema-version-1/2 dumps (grandfathered to "classic" by
            // the version-aware check); present-but-not-a-string is malformed.
            flash: match p.get("flash") {
                Some(d) => Some(d.as_str().context("provenance `flash` is not a string")?.to_string()),
                None => None,
            },
            // Absent in schema-version-1..3 dumps (grandfathered to "classic" by
            // the version-aware check); present-but-not-a-string is malformed.
            act: match p.get("act") {
                Some(d) => Some(d.as_str().context("provenance `act` is not a string")?.to_string()),
                None => None,
            },
        })),
        _ => Ok(None),
    }
}

/// Top-`k` token ids by logit, descending — the same comparator `logits-dump`'s
/// `topk` uses, so re-deriving the recorded top1/top5 from the full logit vector
/// reproduces the writer's order exactly (deterministic tie-break). Used by
/// `load_dump` to reject a dump whose recorded ids were forged or desynced from
/// its logits.
fn topk_ids(logits: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    idx.into_iter().take(k).collect()
}

fn load_dump(path: &Path) -> Result<Dump> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let v: Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;

    let logits: Vec<f32> = v["logits"]
        .as_array()
        .context("`logits` is not an array")?
        .iter()
        .map(|x| x.as_f64().context("non-float logit").map(|f| f as f32))
        .collect::<Result<_>>()?;

    let top5: Vec<(u32, f32)> = v["top5"]
        .as_array()
        .context("`top5` is not an array")?
        .iter()
        .map(|pair| {
            let id = pair[0].as_u64().context("bad top5 token")? as u32;
            let logit = pair[1].as_f64().context("bad top5 logit")? as f32;
            Ok((id, logit))
        })
        .collect::<Result<_>>()?;

    let top1 = v["top1"].as_u64().context("`top1` missing")? as u32;

    // `is_near_tie` / top-5 overlap assume `top5` is sorted descending by logit
    // with `top5[0].0 == top1`. Dumps are external input, so validate at load with
    // a clear error (not a debug_assert).
    for w in top5.windows(2) {
        if w[0].1 < w[1].1 {
            bail!(
                "{}: top5 not sorted descending by logit: ({}, {}) precedes ({}, {})",
                path.display(),
                w[0].0,
                w[0].1,
                w[1].0,
                w[1].1
            );
        }
    }
    if let Some(&(id0, _)) = top5.first() {
        if id0 != top1 {
            bail!("{}: top5[0] token {id0} disagrees with top1 {top1}", path.display());
        }
    }

    // The recorded top1/top5 are external input: a doctored top-k (or a top-k
    // stale relative to an edited logit vector) would otherwise steer the argmax /
    // overlap gate independently of the logits it is supposedly summarizing.
    // Re-derive both from the full logits and require an exact, order-sensitive
    // match. (Same comparator as the writer, so a genuine dump reproduces bit-for-
    // bit — the logits round-trip f32→f64→f32 exactly.)
    let recorded_ids: Vec<u32> = top5.iter().map(|&(id, _)| id).collect();
    let want_ids = topk_ids(&logits, recorded_ids.len());
    if want_ids != recorded_ids {
        bail!(
            "{}: recorded top5 {recorded_ids:?} disagrees with the top-{} recomputed from the \
             logit vector {want_ids:?} (forged or stale top-k)",
            path.display(),
            recorded_ids.len()
        );
    }
    if let Some(&argmax) = want_ids.first() {
        if argmax != top1 {
            bail!(
                "{}: recorded top1 {top1} disagrees with the argmax {argmax} recomputed from the \
                 logit vector",
                path.display()
            );
        }
    }

    let provenance = parse_provenance(&v)?;

    // Taps are optional: a blessed regression dump may omit them, in which case
    // the layer-by-layer diff simply reports nothing.
    let taps: Vec<Tap> = match v["taps"].as_array() {
        None => Vec::new(),
        Some(arr) => arr
            .iter()
            .map(|t| {
                let name = t["name"].as_str().context("tap missing `name`")?.to_string();
                let sum = t["sum"].as_f64().context("tap missing `sum`")?;
                let l2 = t["l2"].as_f64().unwrap_or(0.0);
                let last_row = t["last_row"].as_array().map(|r| {
                    r.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect::<Vec<f32>>()
                });
                Ok(Tap { name, sum, l2, last_row })
            })
            .collect::<Result<_>>()?,
    };

    Ok(Dump {
        tokens: u32_array(&v, "tokens")?,
        logits,
        top1,
        top5,
        taps,
        provenance,
    })
}

/// L2 norm of a logit vector in f64, for the scale (norm-ratio) check.
fn l2_norm(a: &[f32]) -> f64 {
    a.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

/// Cosine over two feature rows of possibly-different length (compare the
/// shared prefix). Returns 1.0 for empty input so it never manufactures a
/// divergence.
fn row_cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 1.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        dot += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

/// Layer-by-layer tap diff, matched by tap name. Reports the max per-layer
/// relative deviation of the whole-tensor sum (the headline number asked for),
/// but the whole-tensor sum is fragile: a tensor whose elements cancel to a
/// near-zero sum inflates the relative error to meaninglessness. So we also
/// report two denominator-stable metrics — |Δsum| normalized by the reference
/// tensor's L2 norm, and the cosine of the full last-position feature row —
/// which reflect the actual per-layer agreement. Diagnostic only; the gate is
/// the full-logit cosine/top-1/top-5 in `compare`.
fn report_taps(candidate: &Dump, reference: &Dump) {
    if candidate.taps.is_empty() || reference.taps.is_empty() {
        eprintln!("taps: none to diff (one dump has no taps)");
        return;
    }
    let ref_by_name: std::collections::HashMap<&str, &Tap> =
        reference.taps.iter().map(|t| (t.name.as_str(), t)).collect();

    struct Row {
        name: String,
        sum_rel: f64,
        norm_dsum: f64,
        row_cos: Option<f64>,
    }
    let mut rows: Vec<Row> = Vec::new();
    for t in &candidate.taps {
        let Some(r) = ref_by_name.get(t.name.as_str()) else { continue };
        let sum_rel = (t.sum - r.sum).abs() / (r.sum.abs() + 1e-6);
        let norm_dsum = (t.sum - r.sum).abs() / (r.l2 + 1e-9);
        let row_cos = match (&t.last_row, &r.last_row) {
            (Some(a), Some(b)) => Some(row_cosine(a, b)),
            _ => None,
        };
        rows.push(Row { name: t.name.clone(), sum_rel, norm_dsum, row_cos });
    }
    if rows.is_empty() {
        eprintln!("taps: no shared tap names to diff");
        return;
    }

    let max_sum_rel = rows.iter().max_by(|a, b| a.sum_rel.total_cmp(&b.sum_rel)).unwrap();
    let max_norm = rows.iter().max_by(|a, b| a.norm_dsum.total_cmp(&b.norm_dsum)).unwrap();
    let min_cos = rows
        .iter()
        .filter_map(|r| r.row_cos.map(|c| (r.name.as_str(), c)))
        .min_by(|a, b| a.1.total_cmp(&b.1));

    eprintln!("--- layer-by-layer tap diff ({} shared taps) ---", rows.len());
    eprintln!(
        "  max per-layer sum rel deviation: {:.3e} @ {}  (raw; cancellation-fragile)",
        max_sum_rel.sum_rel, max_sum_rel.name
    );
    eprintln!(
        "  max |Δsum|/l2 (stable):          {:.3e} @ {}",
        max_norm.norm_dsum, max_norm.name
    );
    if let Some((name, cos)) = min_cos {
        eprintln!("  worst last-row cosine (stable):  {cos:.6} @ {name}");
    }
    // Show the offenders the raw metric flags, with the stable metrics beside
    // them so a cancellation artifact is visible as "huge sum_rel, tiny
    // |Δsum|/l2, cosine ~1".
    rows.sort_by(|a, b| b.sum_rel.total_cmp(&a.sum_rel));
    eprintln!("  top raw-sum-rel offenders:");
    for r in rows.iter().take(6) {
        let cos = r.row_cos.map(|c| format!("{c:.6}")).unwrap_or_else(|| "n/a".into());
        eprintln!(
            "    {:<16} sum_rel={:.3e}  |Δsum|/l2={:.3e}  row_cos={}",
            r.name, r.sum_rel, r.norm_dsum, cos
        );
    }
}

/// Whether a top-1 mismatch is an acceptable near-tie (mm tier only): the
/// candidate's chosen token must be the reference's top-1 or top-2, AND the
/// reference's top-1/top-2 logit margin must be below `NEAR_TIE_MARGIN` — i.e.
/// the reference itself barely separated the two, so which one wins is noise.
fn is_near_tie(candidate: &Dump, reference: &Dump) -> bool {
    if reference.top5.len() < 2 {
        return false;
    }
    let (ref_t1, l1) = reference.top5[0];
    let (ref_t2, l2) = reference.top5[1];
    let cand_is_contender = candidate.top1 == ref_t1 || candidate.top1 == ref_t2;
    let margin = (l1 - l2).abs() as f64;
    cand_is_contender && margin < NEAR_TIE_MARGIN
}

/// Version-aware enforcement of one string provenance field: present must
/// equal `want`; missing resolves through `laguna::parity_schema` — a dump
/// whose `schema_version` predates the field's introduction takes the field's
/// grandfather value (keeping pre-introduction references valid), while
/// missing at/after introduction is the hard "stale binary" fail (such a
/// dump is otherwise indistinguishable from a current one and the gate would
/// pass vacuously).
fn check_field(p: &Provenance, side: &str, name: &str, value: Option<&str>, want: &str) -> Result<()> {
    let effective = match value {
        Some(v) => Some(v),
        None => laguna::parity_schema::resolve_missing(name, p.schema_version),
    };
    match effective {
        Some(v) if v == want => Ok(()),
        Some(other) => bail!(
            "{side} provenance.{name} is {other:?}, expected {want:?}; regenerate the dump \
             under the environment its gate prescribes (see docs/parity.md §3)"
        ),
        None => bail!(
            "{side} provenance has no {name} (stale `logits-dump` binary predating the field; \
             required at schema_version {}); rebuild logits-dump and regenerate the dump",
            p.schema_version
        ),
    }
}

/// Enforce the attention weight dtype recorded in one side's provenance.
/// `want` is "f32" for the Reference-oracle side (reference dumps are produced
/// under `LAGUNA_ATTN_F32=1`) and for strict-tier candidates; "f16" for
/// candidates gating the shipped default path.
fn check_attn_dtype(p: &Provenance, side: &str, want: &str) -> Result<()> {
    check_field(p, side, "attn_dtype", p.attn_dtype.as_deref(), want)
}

/// Enforce the routed-expert combine path recorded in one side's provenance.
/// `want` is "reference" for the Reference-oracle side (which never dispatches
/// `ops::combine`), "classic" for strict-tier candidates (produced under
/// `LAGUNA_COMBINE_CLASSIC=1`, the candle chain the strict anchor was blessed
/// with), and "fused" for the mm/decode/ppl candidates that grade the shipped
/// vendored combine.
fn check_combine(p: &Provenance, side: &str, want: &str) -> Result<()> {
    check_field(p, side, "combine", p.combine.as_deref(), want)
}

/// Enforce the attention prefill gemm path recorded in one side's provenance.
/// `want` is "f32-bypass" for the Reference-oracle side and strict-tier candidates
/// (both run under `LAGUNA_ATTN_F32=1`, which bypasses the f16 library's mm
/// branch — the mm kill-switch is moot there), and "tensor" for the mm/decode/ppl
/// candidates that grade the shipped cooperative-tensor prefill kernel — except
/// when a gate run opts in to grading the `LAGUNA_ATTN_MM_CLASSIC` kill-switch
/// path via `LAGUNA_PARITY_EXPECT_ATTN_MM` (see `expected_attn_mm`).
fn check_attn_mm(p: &Provenance, side: &str, want: &str) -> Result<()> {
    check_field(p, side, "attn_mm", p.attn_mm.as_deref(), want)
}

/// Enforce the attention-glue path recorded in one side's provenance. `want` is
/// "classic" for the Reference-oracle side and strict-tier candidates (both are
/// produced under `LAGUNA_ATTN_GLUE_CLASSIC=1` — unlike `combine`, the oracle
/// EXECUTES the attention glue, so its anchor is the env pin, not a separate
/// code path), and "fused" for the mm/decode/ppl candidates that grade the
/// shipped vendored glue kernels.
fn check_attn_glue(p: &Provenance, side: &str, want: &str) -> Result<()> {
    check_field(p, side, "attn_glue", p.attn_glue.as_deref(), want)
}

/// Enforce the sdpa compute dtype recorded in one side's provenance. "f16" is
/// the shipped kernel and the value every reference and strict candidate must
/// carry (the blessed anchors all ran it); mm/decode/ppl candidates also
/// expect "f16" unless a gate run opts in via `LAGUNA_PARITY_EXPECT_SDPA`
/// (see `expected_sdpa`). Introduced at schema version 2: a version-1 dump
/// missing the field is grandfathered to "f16" by `check_field` — that is what
/// keeps the pre-versioning reference dumps valid.
fn check_sdpa(p: &Provenance, side: &str, want: &str) -> Result<()> {
    check_field(p, side, "sdpa", p.sdpa.as_deref(), want)
}

/// Experiment hook (docs/parity.md §3b): the sdpa dtype expected of mm/decode/
/// ppl CANDIDATES. Default "f16" (the shipped kernel); `LAGUNA_PARITY_EXPECT_SDPA`
/// overrides it so an experiment gate run can grade `LAGUNA_SDPA_F32` candidates
/// (parity-gate.ts `--sdpa-f32` sets both ends). References and strict
/// candidates are never overridable — they anchor the oracle.
fn expected_sdpa() -> String {
    std::env::var("LAGUNA_PARITY_EXPECT_SDPA").unwrap_or_else(|_| "f16".to_string())
}

/// Experiment hook (docs/parity.md §3b): the attention prefill gemm path
/// expected of mm/decode/ppl CANDIDATES. Default "tensor" (the shipped
/// cooperative-tensor kernel); `LAGUNA_PARITY_EXPECT_ATTN_MM` overrides it so
/// an A/B gate run can grade `LAGUNA_ATTN_MM_CLASSIC` candidates
/// (parity-gate.ts `--attn-mm-classic` sets both ends). References and strict
/// candidates stay pinned to "f32-bypass" (they run under `LAGUNA_ATTN_F32=1`,
/// so the f16 library's mm branch never runs at all).
fn expected_attn_mm() -> String {
    std::env::var("LAGUNA_PARITY_EXPECT_ATTN_MM").unwrap_or_else(|_| "tensor".to_string())
}

/// Enforce the prefill attention kernel recorded in one side's provenance.
/// "classic" (the candle sdpa chain, `LAGUNA_FLASH_CLASSIC=1`) is what every
/// reference and strict candidate must carry — the oracle and the strict
/// anchor are pinned off the flash kernel; mm/decode/ppl candidates expect
/// "fused" (the shipped vendored flash prefill) unless a gate run opts out via
/// `LAGUNA_PARITY_EXPECT_FLASH` (see `expected_flash`). Introduced at schema
/// version 3: a v1/v2 dump missing the field is grandfathered to "classic" by
/// `check_field` — that keeps the pre-flash reference dumps valid.
fn check_flash(p: &Provenance, side: &str, want: &str) -> Result<()> {
    check_field(p, side, "flash", p.flash.as_deref(), want)
}

/// Experiment hook (docs/parity.md §3b): the prefill attention kernel expected
/// of mm/decode/ppl CANDIDATES. Default "fused" (the shipped flash kernel);
/// `LAGUNA_PARITY_EXPECT_FLASH` overrides it so an A/B gate run can grade
/// `LAGUNA_FLASH_CLASSIC` candidates (parity-gate.ts `--flash-classic` sets
/// both ends). References and strict candidates are never overridable — they
/// anchor the oracle.
fn expected_flash() -> String {
    std::env::var("LAGUNA_PARITY_EXPECT_FLASH").unwrap_or_else(|_| "fused".to_string())
}

/// Enforce the routed-expert SwiGLU activation path recorded in one side's
/// provenance. `want` is "classic" for the Reference-oracle side (its
/// ReferenceExperts always runs the candle `silu(gate) * up` chain) and for
/// strict-tier candidates (produced under `LAGUNA_ACT_CLASSIC=1`, the candle chain
/// the strict anchor was blessed with), and "fused" for the mm/decode/ppl
/// candidates that grade the shipped vendored `ops::silu_mul` kernel. Introduced
/// at schema version 4: a v1..v3 dump missing the field is grandfathered to
/// "classic" by `check_field` — that keeps the pre-`act` reference dumps valid,
/// since the reference side expects "classic".
fn check_act(p: &Provenance, side: &str, want: &str) -> Result<()> {
    check_field(p, side, "act", p.act.as_deref(), want)
}

fn compare(candidate: &Dump, reference: &Dump, tier: Tier) -> Result<()> {
    let mut failures: Vec<String> = Vec::new();

    // The reference dump must be a genuine Reference-oracle run in EVERY tier: a
    // `reference.json` accidentally produced with `--moe-impl fused` would turn
    // every tier into a fused-vs-fused self-comparison that hides a regression.
    // Fail closed — a missing/wrong reference provenance is a hard fail asking for
    // a regenerate, not a legacy exception (the reference side is what we grade
    // against, so it must be trustworthy). See docs/parity.md §3.
    match &reference.provenance {
        // The oracle also pins f32 attention compute (`LAGUNA_ATTN_F32=1`, see
        // parity-gate.ts referenceEnv()); an f16 or field-less reference would
        // silently move the strict tier's anchor. The Reference runner never
        // dispatches `ops::combine`, so its combine provenance must be "reference".
        Some(p) if p.moe_impl == "reference" => {
            check_attn_dtype(p, "reference dump", "f32")?;
            check_combine(p, "reference dump", "reference")?;
            // f32 attention bypasses the f16 library's mm branch entirely.
            check_attn_mm(p, "reference dump", "f32-bypass")?;
            // The oracle runs the attention glue too, pinned to the candle
            // chains (LAGUNA_ATTN_GLUE_CLASSIC=1, parity-gate.ts referenceEnv()).
            check_attn_glue(p, "reference dump", "classic")?;
            // The oracle always runs the shipped f16 sdpa kernel (the
            // LAGUNA_SDPA_F32 experiment hook is candidate-only).
            check_sdpa(p, "reference dump", "f16")?;
            // The oracle's prefill is pinned off the flash kernel
            // (LAGUNA_FLASH_CLASSIC=1, parity-gate.ts referenceEnv()).
            check_flash(p, "reference dump", "classic")?;
            // The oracle's ReferenceExperts always runs the candle silu*mul chain.
            check_act(p, "reference dump", "classic")?;
        }
        Some(p) => bail!(
            "reference dump provenance.moe_impl is {:?}, expected \"reference\": the reference side \
             must be a Reference-oracle run. Regenerate reference.json with the current `logits-dump` \
             (--moe-impl reference) — see docs/parity.md §3.",
            p.moe_impl
        ),
        None => bail!(
            "reference dump has no provenance (predates the field). Regenerate reference.json with \
             the current `logits-dump` (--moe-impl reference) — see docs/parity.md §3."
        ),
    }

    if candidate.tokens != reference.tokens {
        failures.push(format!(
            "input tokens differ: candidate {:?} vs reference {:?}",
            candidate.tokens, reference.tokens
        ));
    }
    if candidate.logits.len() != reference.logits.len() {
        bail!(
            "logit vectors differ in length: {} vs {}",
            candidate.logits.len(),
            reference.logits.len()
        );
    }

    // mm tier: the CANDIDATE must carry provenance proving it actually ran the
    // mm_id prefill path. Grading a decode/mv_id (or reference) dump under the
    // looser mm gate would mask a regression the strict gate would catch, so a
    // missing or non-mm provenance is a hard fail (regenerate the dump). The
    // strict tier has its own candidate check below; decode adds none beyond the
    // attn_dtype pin.
    if tier == Tier::Mm {
        match &candidate.provenance {
            None => bail!(
                "mm tier requires candidate provenance, but the dump has none (predates the \
                 provenance field). Regenerate the candidate with the current `logits-dump` \
                 (fused runner, mm_id prefill) — see docs/parity.md §3b."
            ),
            Some(p) if !p.mm_active() => bail!(
                "mm tier: candidate provenance shows the mm_id path was NOT active \
                 (moe_impl={:?}, seq_len={} vs mm_min_seq={}, no_mm_id={}, variant={:?}). \
                 A candidate graded under the loose mm tier must have run the fused mm_id path; \
                 regenerate it (fused runner, prompt seq_len >= {}, LAGUNA_NO_MM_ID unset).",
                p.moe_impl, p.seq_len, p.mm_min_seq, p.no_mm_id, p.mm_variant, p.mm_min_seq
            ),
            Some(_) => {}
        }
    }

    // EVERY tier pins the CANDIDATE's attention weight dtype: strict gates the
    // legacy f32 attention path (candidate produced under LAGUNA_ATTN_F32=1),
    // while mm and decode gate the shipped f16-weight default. Candidate provenance is
    // therefore required in every tier — a dump missing it (or missing the
    // attn_dtype field) came from a stale `logits-dump` binary and must be
    // regenerated, not graded.
    let want_attn = match tier {
        Tier::Strict => "f32",
        Tier::Mm | Tier::Decode => "f16",
    };
    match &candidate.provenance {
        Some(p) => check_attn_dtype(p, "candidate dump", want_attn)?,
        None => bail!(
            "candidate dump has no provenance (predates the field). Regenerate the candidate with \
             the current `logits-dump` — see docs/parity.md §3b."
        ),
    }

    // EVERY tier also pins the CANDIDATE's attention prefill gemm path: strict runs
    // under LAGUNA_ATTN_F32 (f16 library bypassed → "f32-bypass"; the classic path
    // strict anchors never dispatches the mm branch), while mm and decode grade the
    // shipped cooperative-tensor prefill kernel by default — overridable per gate
    // run via LAGUNA_PARITY_EXPECT_ATTN_MM (experiment hook).
    // Candidate provenance is already proven Some by the attn_dtype check above.
    let want_attn_mm = match tier {
        Tier::Strict => "f32-bypass".to_string(),
        Tier::Mm | Tier::Decode => expected_attn_mm(),
    };
    if let Some(p) = &candidate.provenance {
        check_attn_mm(p, "candidate dump", &want_attn_mm)?;
    }

    // EVERY tier pins the CANDIDATE's sdpa compute dtype: "f16" (the shipped
    // kernel) everywhere by default; mm and decode may be overridden per gate
    // run via LAGUNA_PARITY_EXPECT_SDPA (experiment hook — grading a
    // LAGUNA_SDPA_F32 candidate). Strict stays pinned: its 0.999 anchor was
    // blessed on the f16 kernel.
    let want_sdpa = match tier {
        Tier::Strict => "f16".to_string(),
        Tier::Mm | Tier::Decode => expected_sdpa(),
    };
    if let Some(p) = &candidate.provenance {
        check_sdpa(p, "candidate dump", &want_sdpa)?;
    }

    // strict tier: the CANDIDATE must be a Fused-runner dump produced under
    // LAGUNA_NO_MM_ID=1 (the classic mv fallback path strict grades). Without
    // this, a copied reference dump (moe_impl "reference", attn_dtype "f32" —
    // exactly what strict's attn pin expects) would clear the strict thresholds
    // vacuously, and an mm_id-path dump would be graded against the wrong
    // envelope. Mirrors the mm tier's candidate check above.
    if tier == Tier::Strict {
        match &candidate.provenance {
            None => bail!(
                "strict tier requires candidate provenance, but the dump has none (predates the \
                 provenance field). Regenerate the candidate with the current `logits-dump` \
                 (fused runner, LAGUNA_NO_MM_ID=1 LAGUNA_MV_CLASSIC=1 LAGUNA_ATTN_F32=1) — see \
                 docs/parity.md §3b."
            ),
            Some(p) if p.moe_impl != "fused" => bail!(
                "strict tier: candidate provenance.moe_impl is {:?}, expected \"fused\" — a \
                 reference dump graded as its own candidate passes vacuously. Regenerate the \
                 candidate with the fused runner — see docs/parity.md §3b.",
                p.moe_impl
            ),
            Some(p) if !p.no_mm_id => bail!(
                "strict tier: candidate provenance shows no_mm_id={}, but strict grades the \
                 classic mv fallback path (candidate produced under LAGUNA_NO_MM_ID=1 \
                 LAGUNA_MV_CLASSIC=1 LAGUNA_ATTN_F32=1). Regenerate the candidate — see \
                 docs/parity.md §3b.",
                p.no_mm_id
            ),
            Some(_) => {}
        }
    }

    // EVERY tier also pins the CANDIDATE's routed-expert combine path (like
    // attn_dtype): strict grades the classic candle combine (candidate produced
    // under LAGUNA_COMBINE_CLASSIC=1 → "classic"), mm/decode grade the shipped
    // vendored combine ("fused"). Placed after the mm_active / strict-fused runner
    // checks above, so a reference dump misused as a candidate fails on moe_impl
    // there, not here. Candidate provenance is already proven Some by the
    // attn_dtype check above.
    let want_combine = match tier {
        Tier::Strict => "classic",
        Tier::Mm | Tier::Decode => "fused",
    };
    if let Some(p) = &candidate.provenance {
        check_combine(p, "candidate dump", want_combine)?;
    }

    // EVERY tier also pins the CANDIDATE's attention-glue path (like combine):
    // strict grades the candle glue chains (candidate produced under
    // LAGUNA_ATTN_GLUE_CLASSIC=1 → "classic"), mm/decode grade the shipped
    // fused glue kernels ("fused"). Candidate provenance is already proven Some
    // by the attn_dtype check above.
    let want_attn_glue = match tier {
        Tier::Strict => "classic",
        Tier::Mm | Tier::Decode => "fused",
    };
    if let Some(p) = &candidate.provenance {
        check_attn_glue(p, "candidate dump", want_attn_glue)?;
    }

    // EVERY tier also pins the CANDIDATE's prefill attention kernel (like
    // sdpa): strict grades the candle sdpa chain (candidate produced under
    // LAGUNA_FLASH_CLASSIC=1 → "classic"), mm/decode grade the shipped flash
    // kernel ("fused") by default — overridable per gate run via
    // LAGUNA_PARITY_EXPECT_FLASH (A/B hook, parity-gate.ts --flash-classic).
    let want_flash = match tier {
        Tier::Strict => "classic".to_string(),
        Tier::Mm | Tier::Decode => expected_flash(),
    };
    if let Some(p) = &candidate.provenance {
        check_flash(p, "candidate dump", &want_flash)?;
    }

    // EVERY tier also pins the CANDIDATE's routed-expert activation path (like
    // combine): strict grades the candle silu*mul chain (candidate produced under
    // LAGUNA_ACT_CLASSIC=1 → "classic"), mm/decode grade the shipped fused
    // ops::silu_mul kernel ("fused"). The fused kernel is bit-identical to the
    // chain, so — like combine — strict pins classic purely as the blessed-anchor
    // discipline. Candidate provenance is already proven Some by the attn_dtype
    // check above.
    let want_act = match tier {
        Tier::Strict => "classic",
        Tier::Mm | Tier::Decode => "fused",
    };
    if let Some(p) = &candidate.provenance {
        check_act(p, "candidate dump", want_act)?;
    }

    // Non-finite candidate logits are a hard failure in EVERY tier: a NaN/Inf
    // slips past the scale-invariant cosine/top-k checks (`NaN < cos_min` is
    // false, so it would "pass") and the greedy tier's argmax.
    let nonfinite = candidate.logits.iter().filter(|x| !x.is_finite()).count();
    if nonfinite > 0 {
        failures.push(format!("{nonfinite} non-finite candidate logits (NaN/Inf)"));
    }

    // Scale check (EVERY tier): cosine and top-k are scale-invariant, so a
    // uniformly rescaled candidate passes them at cosine 1.0. The L2-norm ratio
    // catches that (bound calibrated in docs/parity.md §3b). A non-finite ratio
    // (e.g. from non-finite logits) is itself a failure.
    let ref_norm = l2_norm(&reference.logits);
    let cand_norm = l2_norm(&candidate.logits);
    let ratio = cand_norm / ref_norm;
    if !ratio.is_finite() || ratio < 1.0 / NORM_RATIO_MAX || ratio > NORM_RATIO_MAX {
        failures.push(format!(
            "L2-norm ratio {ratio:.4} outside [{:.4}, {NORM_RATIO_MAX}] \
             (candidate norm {cand_norm:.2} vs reference {ref_norm:.2})",
            1.0 / NORM_RATIO_MAX
        ));
    }

    let cos = cosine(&candidate.logits, &reference.logits);
    if !cos.is_finite() {
        failures.push(format!("cosine metric is non-finite ({cos})"));
    }
    let overlap = candidate
        .top5
        .iter()
        .filter(|(t, _)| reference.top5.iter().any(|(r, _)| r == t))
        .count();

    // Decode tier: cosine/top-1/top-5 are diagnostics, not a gate (the greedy
    // agreement gate in `greedy_parity` is the real check). Hard fails still apply
    // — input token / logit-length mismatch, non-finite logits, and a scale
    // (norm-ratio) blowout — since a diagnostic over those is meaningless.
    if tier == Tier::Decode {
        report_taps(candidate, reference);
        eprintln!(
            "decode tier (DIAGNOSTIC ONLY — gate is greedy_parity): cosine={cos:.6}, \
             candidate top1={} vs reference top1={}, top5 overlap={overlap}/5",
            candidate.top1, reference.top1
        );
        if failures.is_empty() {
            return Ok(());
        }
        bail!(
            "decode tier hard-fail ({} criteria — inputs / finiteness / scale):\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }

    let cos_min = match tier {
        Tier::Strict => COS_MIN_STRICT,
        Tier::Mm => COS_MIN_MM,
        Tier::Decode => unreachable!("decode tier returns above"),
    };
    if cos < cos_min {
        failures.push(format!("cosine {cos:.6} < {cos_min}"));
    }

    // Strict tier: unconditional top-1 match. mm tier: match OR a reference
    // near-tie (candidate picked one of the reference's two contenders and the
    // reference barely separated them).
    if candidate.top1 != reference.top1 {
        let excused = tier == Tier::Mm && is_near_tie(candidate, reference);
        if !excused {
            let extra = if tier == Tier::Mm {
                let margin = if reference.top5.len() >= 2 {
                    (reference.top5[0].1 - reference.top5[1].1).abs()
                } else {
                    f32::INFINITY
                };
                format!(" (not a near-tie: reference top1/top2 margin {margin:.3} >= {NEAR_TIE_MARGIN} or candidate not a contender)")
            } else {
                String::new()
            };
            failures.push(format!(
                "top-1 differs: candidate {} vs reference {}{extra}",
                candidate.top1, reference.top1
            ));
        }
    }

    if overlap < TOP5_OVERLAP_MIN {
        failures.push(format!(
            "top-5 overlap {overlap}/5 < {TOP5_OVERLAP_MIN} (candidate {:?} vs reference {:?})",
            candidate.top5, reference.top5
        ));
    }

    report_taps(candidate, reference);

    let tier_name = match tier {
        Tier::Strict => "strict",
        Tier::Mm => "mm",
        Tier::Decode => unreachable!("decode tier returns above"),
    };
    if failures.is_empty() {
        eprintln!("parity PASS ({tier_name} tier): cosine={cos:.6}, top1={}, top5 overlap={overlap}/5", candidate.top1);
        Ok(())
    } else {
        bail!(
            "parity FAIL ({tier_name} tier, {} criteria):\n  - {}\n(cosine={cos:.6}, top5 overlap={overlap}/5)",
            failures.len(),
            failures.join("\n  - ")
        )
    }
}

fn parity_dir() -> Result<PathBuf> {
    let dir = std::env::var("LAGUNA_PARITY_DIR")
        .context("set LAGUNA_PARITY_DIR to a directory containing candidate.json and reference.json")?;
    Ok(PathBuf::from(dir))
}

#[test]
#[ignore = "needs real dumps; run with LAGUNA_PARITY_DIR=<dir> cargo test --test parity -- --ignored"]
fn logit_parity() -> Result<()> {
    let dir = parity_dir()?;
    let tier = Tier::from_env()?;
    let candidate = load_dump(&dir.join("candidate.json"))?;
    let reference = load_dump(&dir.join("reference.json"))?;
    compare(&candidate, &reference, tier)
}

/// One decode step of a `logits-dump` greedy/replay dump: the runner's own
/// top-1/top-2 at that position, plus (greedy) the token it produced or
/// (replay) the reference token it was forced along.
struct Step {
    /// Greedy dump only: the argmax token this position produced (== `top1.0`).
    token: Option<u32>,
    /// Replay dump only: the reference token that was teacher-forced here.
    forced_token: Option<u32>,
    top1: (u32, f32),
    top2: (u32, f32),
    /// L2 norm of this step's full logit vector, and the count of non-finite
    /// logits — the only scale/finiteness signal the decode gate has, since the
    /// dumps carry no full logits. Written by the current `logits-dump`.
    l2: f64,
    nonfinite: u64,
}

/// A greedy (`--greedy`) or replay (`--replay`) decode dump.
struct GreedyDump {
    kind: String,
    /// Prompt token ids (before any decode step).
    prompt: Vec<u32>,
    steps: Vec<Step>,
    /// How the dump was produced. The gate enforces the runner (reference vs
    /// fused) per side; a missing provenance is a hard fail (regenerate).
    provenance: Option<Provenance>,
}

fn pair(v: &Value, key: &str) -> Result<(u32, f32)> {
    let arr = v[key].as_array().with_context(|| format!("step missing `{key}`"))?;
    let id_raw = arr.first().and_then(Value::as_u64).with_context(|| format!("bad `{key}` id"))?;
    let id = u32::try_from(id_raw).with_context(|| format!("`{key}` id {id_raw} exceeds u32"))?;
    let logit = arr.get(1).and_then(Value::as_f64).with_context(|| format!("bad `{key}` logit"))? as f32;
    Ok((id, logit))
}

fn load_greedy_dump(path: &Path) -> Result<GreedyDump> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let v: Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;

    let kind = v["kind"].as_str().context("dump missing `kind`")?.to_string();
    let prompt = u32_array(&v, "tokens")?;
    let steps = v["steps"]
        .as_array()
        .context("dump missing `steps`")?
        .iter()
        .map(|s| {
            // `l2`/`nonfinite` are the decode gate's scale/finiteness signal; a dump
            // missing them predates the check and must be regenerated (hard fail),
            // not silently exempted.
            let l2 = s["l2"].as_f64().with_context(|| {
                format!("step missing `l2` (regenerate the dump with the current `logits-dump`) in {}", path.display())
            })?;
            let nonfinite = s["nonfinite"].as_u64().with_context(|| {
                format!("step missing `nonfinite` (regenerate the dump with the current `logits-dump`) in {}", path.display())
            })?;
            Ok(Step {
                token: s["token"].as_u64().map(|n| n as u32),
                forced_token: s["forced_token"].as_u64().map(|n| n as u32),
                top1: pair(s, "top1")?,
                top2: pair(s, "top2")?,
                l2,
                nonfinite,
            })
        })
        .collect::<Result<_>>()?;

    let provenance = parse_provenance(&v)?;
    Ok(GreedyDump { kind, prompt, steps, provenance })
}

/// Decode-kernel parity gate: greedy agreement vs the frozen Reference oracle,
/// under teacher forcing. The reference side is a `--greedy` dump from the
/// Reference runner; the candidate is a `--replay` dump from the Fused runner,
/// forced step-by-step along the reference's tokens. A step passes when the
/// candidate's argmax equals the reference token; a mismatch is excused only
/// when the reference itself barely separated its own top-1/top-2 (margin <
/// `NEAR_TIE_MARGIN`) AND the two sides' picks are mutual contenders (the
/// candidate's pick is in the reference's top-2, or the reference token is in
/// the candidate's top-2 with the candidate's own margin also sub-tie), i.e.
/// which token wins there is oracle noise. Total excuses are capped at
/// max(2, n/8) — see `greedy_compare`.
///
/// Forced replay (not free-run) is deliberate: a free-run comparison cascades at
/// the first near-tie — once the two engines pick different tokens the histories
/// diverge and every later step is incomparable (WP8's long-swa free-run split
/// at step 9 on a 0.079-logit near-tie). Forcing the reference token at every
/// step keeps all N positions comparable.
#[test]
#[ignore = "needs real dumps; run with LAGUNA_PARITY_DIR=<dir> LAGUNA_PARITY_TIER=decode cargo test --test parity -- --ignored"]
fn greedy_parity() -> Result<()> {
    let dir = parity_dir()?;
    let reference = load_greedy_dump(&dir.join("reference-greedy.json"))?;
    let candidate = load_greedy_dump(&dir.join("candidate-greedy.json"))?;
    greedy_compare(&reference, &candidate)
}

/// The decode-gate comparison itself, split out from disk I/O so every rejection
/// path is unit-testable. See `greedy_parity` for the semantics and rationale.
fn greedy_compare(reference: &GreedyDump, candidate: &GreedyDump) -> Result<()> {
    // Fail closed on anything that would make the comparison meaningless.
    if reference.kind != "greedy" {
        bail!("reference-greedy.json must be kind `greedy`, got `{}`", reference.kind);
    }
    if candidate.kind != "replay" {
        bail!("candidate-greedy.json must be kind `replay`, got `{}`", candidate.kind);
    }

    // Enforce the runner per side: the reference must be the Reference oracle and
    // the candidate (replay) the Fused runner. The `--moe-impl` default of
    // "reference" means a forgotten `--moe-impl fused` on the candidate side would
    // otherwise make the gate compare the reference oracle against itself — a
    // vacuous self-agreement pass. A missing provenance predates the field and is
    // a hard fail (regenerate), not a legacy exemption.
    match &reference.provenance {
        // The oracle side additionally pins f32 attention compute
        // (`LAGUNA_ATTN_F32=1`, parity-gate.ts referenceEnv()) and combine
        // "reference" (the Reference runner never dispatches ops::combine).
        Some(p) if p.moe_impl == "reference" => {
            check_attn_dtype(p, "reference-greedy.json", "f32")?;
            check_combine(p, "reference-greedy.json", "reference")?;
            check_attn_mm(p, "reference-greedy.json", "f32-bypass")?;
            check_attn_glue(p, "reference-greedy.json", "classic")?;
            check_sdpa(p, "reference-greedy.json", "f16")?;
            check_flash(p, "reference-greedy.json", "classic")?;
            check_act(p, "reference-greedy.json", "classic")?;
        }
        Some(p) => bail!(
            "reference-greedy.json provenance.moe_impl is {:?}, expected \"reference\"; regenerate \
             it with the current `logits-dump` (--moe-impl reference --greedy N)",
            p.moe_impl
        ),
        None => bail!(
            "reference-greedy.json has no provenance (predates the field); regenerate it with the \
             current `logits-dump` (--moe-impl reference --greedy N)"
        ),
    }
    match &candidate.provenance {
        // The decode gate grades the SHIPPED path, which computes attention in
        // f16 and uses the fused combine; an f32/classic (or field-less) candidate
        // ran some other build/env. The attn_mm and sdpa expectations are
        // overridable per gate run (LAGUNA_PARITY_EXPECT_* experiment hooks).
        Some(p) if p.moe_impl == "fused" => {
            check_attn_dtype(p, "candidate-greedy.json", "f16")?;
            check_combine(p, "candidate-greedy.json", "fused")?;
            check_attn_mm(p, "candidate-greedy.json", &expected_attn_mm())?;
            check_attn_glue(p, "candidate-greedy.json", "fused")?;
            check_sdpa(p, "candidate-greedy.json", &expected_sdpa())?;
            check_flash(p, "candidate-greedy.json", &expected_flash())?;
            check_act(p, "candidate-greedy.json", "fused")?;
        }
        Some(p) => bail!(
            "candidate-greedy.json (replay) provenance.moe_impl is {:?}, expected \"fused\"; the \
             candidate must be the Fused runner. Regenerate it with the current `logits-dump` \
             (--moe-impl fused --replay ...)",
            p.moe_impl
        ),
        None => bail!(
            "candidate-greedy.json has no provenance (predates the field); regenerate it with the \
             current `logits-dump` (--moe-impl fused --replay ...)"
        ),
    }

    if reference.prompt != candidate.prompt {
        bail!(
            "prompt tokens differ: reference {:?} vs candidate {:?}",
            reference.prompt, candidate.prompt
        );
    }
    if reference.steps.len() != candidate.steps.len() {
        bail!(
            "step counts differ: reference {} vs candidate {}",
            reference.steps.len(),
            candidate.steps.len()
        );
    }

    let n = reference.steps.len();
    if n == 0 {
        bail!("no decode steps to compare");
    }

    let mut agreements = 0usize;
    let mut near_ties = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for (i, (r, c)) in reference.steps.iter().zip(&candidate.steps).enumerate() {
        // Finiteness (both sides): the dumps carry no full logits, so a NaN/Inf in
        // the decode path would otherwise pass unseen — the argmax comparison and
        // the near-tie margin both swallow non-finite values silently.
        for (side, s) in [("reference", r), ("candidate", c)] {
            if s.nonfinite > 0 {
                failures.push(format!("step {i}: {side} has {} non-finite logits", s.nonfinite));
            }
            if !s.top1.1.is_finite() || !s.top2.1.is_finite() {
                failures.push(format!(
                    "step {i}: {side} recorded a non-finite top1/top2 logit (top1={:?}, top2={:?})",
                    s.top1, s.top2
                ));
            }
            if !s.l2.is_finite() {
                failures.push(format!("step {i}: {side} l2 is non-finite ({})", s.l2));
            }
        }
        // Scale (per step): the argmax comparison is scale-INVARIANT, so a decode
        // path that uniformly rescaled the logits would agree on every token yet be
        // wrong. Bound the candidate/reference L2-norm ratio, same as the
        // full-logit gate.
        let ratio = c.l2 / r.l2;
        if !ratio.is_finite() || ratio < 1.0 / NORM_RATIO_MAX || ratio > NORM_RATIO_MAX {
            failures.push(format!(
                "step {i}: candidate/reference l2 ratio {ratio:.4} outside [{:.4}, {NORM_RATIO_MAX}] \
                 (candidate l2 {:.3} vs reference {:.3})",
                1.0 / NORM_RATIO_MAX,
                c.l2,
                r.l2
            ));
        }

        // The reference token is the argmax it produced; it must match its own
        // top-1 (greedy). The candidate must have been forced along it.
        let ref_token = r.token.with_context(|| format!("reference step {i} missing `token`"))?;
        if ref_token != r.top1.0 {
            bail!("reference step {i} token {ref_token} != its top1 {} (not greedy?)", r.top1.0);
        }
        let forced = c.forced_token.with_context(|| format!("candidate step {i} missing `forced_token`"))?;
        if forced != ref_token {
            bail!("candidate step {i} was forced with {forced}, but reference produced {ref_token}");
        }

        let cand_token = c.top1.0;
        if cand_token == ref_token {
            agreements += 1;
            continue;
        }
        // Excuse a mismatch ONLY as a genuine reference near-tie: the oracle's own
        // top-1/top-2 margin is below NEAR_TIE_MARGIN AND the two argmaxes are
        // mutual contenders — the candidate's pick appears in the reference's
        // top-2, OR the reference's pick appears in the candidate's top-2 AND
        // the candidate's own top-1/top-2 margin is also below NEAR_TIE_MARGIN.
        // The second clause is fork-grounded: at small margins the oracle's
        // stored top-2 is not always the true contender set (the fork itself has
        // ranked a candidate's "non-contender" pick #2 at a 3-way tie), but only
        // when the candidate is itself flat there (the calibrated case had
        // candidate margin 0.068) — a candidate that CONFIDENTLY picks a token
        // outside the reference's contenders is never fork-class tie noise, so
        // that branch requires the tie to be flat on BOTH sides. An arbitrary
        // wrong token with no contender overlap is always a failure.
        // (`cand_token == r.top1.0` and `ref_token == c.top1.0` are impossible
        // here — the agreement case above already returned — but the symmetric
        // form keeps the rule readable.)
        let ref_margin = (r.top1.1 - r.top2.1).abs() as f64;
        let cand_margin = (c.top1.1 - c.top2.1).abs() as f64;
        let cand_in_ref_top2 = cand_token == r.top1.0 || cand_token == r.top2.0;
        let ref_in_cand_top2 = ref_token == c.top1.0 || ref_token == c.top2.0;
        let both_sides_flat = ref_in_cand_top2 && cand_margin < NEAR_TIE_MARGIN;
        if ref_margin < NEAR_TIE_MARGIN && (cand_in_ref_top2 || both_sides_flat) {
            near_ties += 1;
            eprintln!(
                "  step {i}: excused near-tie — candidate {cand_token} vs reference {ref_token}, \
                 reference top1/top2 margin {ref_margin:.4} < {NEAR_TIE_MARGIN}"
            );
            continue;
        }
        let reason = if ref_margin >= NEAR_TIE_MARGIN {
            format!("reference top1/top2 margin {ref_margin:.4} >= {NEAR_TIE_MARGIN}")
        } else if ref_in_cand_top2 {
            format!(
                "candidate confidently picked a non-contender: candidate {cand_token} not in \
                 reference top-2 (top1 {}, top2 {}) and candidate top1/top2 margin \
                 {cand_margin:.4} >= {NEAR_TIE_MARGIN} (the reference-in-candidate-top-2 excuse \
                 requires a near-tie on BOTH sides)",
                r.top1.0, r.top2.0
            )
        } else {
            format!(
                "no contender overlap: candidate {cand_token} not in reference top-2 \
                 (top1 {}, top2 {}) and reference {ref_token} not in candidate top-2 \
                 (top1 {}, top2 {})",
                r.top1.0, r.top2.0, c.top1.0, c.top2.0
            )
        };
        failures.push(format!(
            "step {i}: candidate {cand_token} vs reference {ref_token}, {reason}"
        ));
    }

    // Cap the total excused near-ties: each excuse is individually plausible, but
    // fork calibration measured <= 4/64 tie-flips per fixture and observed
    // legitimate runs top out at 6/64 (long-swa's 5 plus headroom) — a candidate
    // that needs to excuse more than 1-in-8 steps is drifting, not riding oracle
    // tie noise.
    let excused_cap = std::cmp::max(2, n / 8);
    if near_ties > excused_cap {
        failures.push(format!(
            "{near_ties} excused near-ties exceed the cap {excused_cap} (max(2, n/8) with n={n}): \
             this many tie-flips is systematic drift, not tie noise"
        ));
    }

    eprintln!(
        "greedy decode gate: {n} steps, {agreements} agreements, {near_ties} excused near-ties, \
         {} non-excused mismatches",
        failures.len()
    );
    if failures.is_empty() {
        Ok(())
    } else {
        bail!(
            "greedy decode gate FAIL ({} non-excused mismatches of {n} steps):\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        )
    }
}

// --- Perplexity-delta gate ----------------------------------------------------

/// A `kind:"ppl"` dump from `logits-dump --ppl`: the mean next-token NLL over the
/// frozen corpus plus enough to re-verify the two runners scored the identical
/// token stream. The reference side (`--moe-impl reference`) is a keepable frozen
/// artifact under `tests/fixtures/reference-ppl.json`.
struct PplDump {
    kind: String,
    /// Full scored token stream (leading BOS + corpus). Empty only if a stored
    /// reference dump ever trims it — the `token_hash` still gates alignment.
    tokens: Vec<u32>,
    n_tokens: usize,
    token_hash: String,
    mean_nll: f64,
    nonfinite: u64,
    provenance: Option<Provenance>,
}

fn load_ppl_dump(path: &Path) -> Result<PplDump> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let v: Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    Ok(PplDump {
        kind: v["kind"].as_str().context("ppl dump missing `kind`")?.to_string(),
        tokens: u32_array(&v, "tokens")?,
        n_tokens: v["n_tokens"].as_u64().context("ppl dump missing `n_tokens`")? as usize,
        token_hash: v["token_hash"].as_str().context("ppl dump missing `token_hash`")?.to_string(),
        mean_nll: v["mean_nll"].as_f64().context("ppl dump missing `mean_nll`")?,
        nonfinite: v["nonfinite"].as_u64().context("ppl dump missing `nonfinite`")?,
        provenance: parse_provenance(&v)?,
    })
}

/// Perplexity-parity gate comparison, split from disk I/O so every rejection path
/// is unit-testable (same pattern as `greedy_compare`). Enforces: both sides are
/// `kind:"ppl"`; the reference ran the Reference oracle and the candidate the
/// Fused runner (a forgotten `--moe-impl fused` would self-compare the oracle);
/// neither side hit a non-finite logprob; the two sides scored the identical
/// token stream (count, ids, and hash); and the mean-NLL delta is within
/// `PPL_NLL_DELTA_MAX`.
fn ppl_compare(reference: &PplDump, candidate: &PplDump) -> Result<()> {
    if reference.kind != "ppl" {
        bail!("reference-ppl.json must be kind `ppl`, got `{}`", reference.kind);
    }
    if candidate.kind != "ppl" {
        bail!("candidate-ppl.json must be kind `ppl`, got `{}`", candidate.kind);
    }

    match &reference.provenance {
        // The oracle side additionally pins f32 attention compute
        // (`LAGUNA_ATTN_F32=1`, parity-gate.ts referenceEnv()) and combine
        // "reference" (the Reference runner never dispatches ops::combine).
        Some(p) if p.moe_impl == "reference" => {
            check_attn_dtype(p, "reference-ppl.json", "f32")?;
            check_combine(p, "reference-ppl.json", "reference")?;
            check_attn_mm(p, "reference-ppl.json", "f32-bypass")?;
            check_attn_glue(p, "reference-ppl.json", "classic")?;
            check_sdpa(p, "reference-ppl.json", "f16")?;
            check_flash(p, "reference-ppl.json", "classic")?;
            check_act(p, "reference-ppl.json", "classic")?;
        }
        Some(p) => bail!(
            "reference-ppl.json provenance.moe_impl is {:?}, expected \"reference\"; regenerate it \
             with the current `logits-dump` (--moe-impl reference --ppl ...)",
            p.moe_impl
        ),
        None => bail!(
            "reference-ppl.json has no provenance (predates the field); regenerate it with the \
             current `logits-dump` (--moe-impl reference --ppl ...)"
        ),
    }
    match &candidate.provenance {
        // The ppl gate grades the SHIPPED path, which streams f16 attention weights
        // and uses the fused combine. The attn_mm and sdpa expectations are
        // overridable per gate run (LAGUNA_PARITY_EXPECT_* experiment hooks).
        Some(p) if p.moe_impl == "fused" => {
            check_attn_dtype(p, "candidate-ppl.json", "f16")?;
            check_combine(p, "candidate-ppl.json", "fused")?;
            check_attn_mm(p, "candidate-ppl.json", &expected_attn_mm())?;
            check_attn_glue(p, "candidate-ppl.json", "fused")?;
            check_sdpa(p, "candidate-ppl.json", &expected_sdpa())?;
            check_flash(p, "candidate-ppl.json", &expected_flash())?;
            check_act(p, "candidate-ppl.json", "fused")?;
        }
        Some(p) => bail!(
            "candidate-ppl.json provenance.moe_impl is {:?}, expected \"fused\"; the candidate must \
             be the Fused runner. Regenerate it with the current `logits-dump` (--moe-impl fused --ppl ...)",
            p.moe_impl
        ),
        None => bail!(
            "candidate-ppl.json has no provenance (predates the field); regenerate it with the \
             current `logits-dump` (--moe-impl fused --ppl ...)"
        ),
    }

    let mut failures: Vec<String> = Vec::new();

    if !reference.mean_nll.is_finite() {
        failures.push(format!("reference mean_nll is non-finite ({})", reference.mean_nll));
    }
    if !candidate.mean_nll.is_finite() {
        failures.push(format!("candidate mean_nll is non-finite ({})", candidate.mean_nll));
    }
    if reference.nonfinite > 0 {
        failures.push(format!("reference scored {} non-finite logprobs", reference.nonfinite));
    }
    if candidate.nonfinite > 0 {
        failures.push(format!("candidate scored {} non-finite logprobs", candidate.nonfinite));
    }

    // Both sides must have scored the identical token stream, or the delta is
    // comparing perplexities of different corpora. Count + hash always; the full
    // id vector when both dumps carry it (a trimmed reference falls back to hash).
    if reference.n_tokens != candidate.n_tokens {
        failures.push(format!(
            "token counts differ: reference {} vs candidate {}",
            reference.n_tokens, candidate.n_tokens
        ));
    }
    if reference.token_hash != candidate.token_hash {
        failures.push(format!(
            "token_hash differs: reference {} vs candidate {} (different token streams)",
            reference.token_hash, candidate.token_hash
        ));
    }
    if !reference.tokens.is_empty() && !candidate.tokens.is_empty() && reference.tokens != candidate.tokens {
        failures.push("token id streams differ (see token_hash)".to_string());
    }

    let delta = (candidate.mean_nll - reference.mean_nll).abs();
    if delta > PPL_NLL_DELTA_MAX {
        failures.push(format!(
            "mean-NLL delta {delta:.6} > {PPL_NLL_DELTA_MAX} \
             (fused {:.6} vs reference {:.6})",
            candidate.mean_nll, reference.mean_nll
        ));
    }

    if failures.is_empty() {
        eprintln!(
            "ppl parity PASS: |Δmean_nll| = {delta:.6} <= {PPL_NLL_DELTA_MAX} \
             (fused {:.6} vs reference {:.6}, {} tokens)",
            candidate.mean_nll, reference.mean_nll, reference.n_tokens
        );
        Ok(())
    } else {
        bail!(
            "ppl parity FAIL ({} criteria):\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        )
    }
}

/// Perplexity-delta gate: bound |mean_NLL(fused) − mean_NLL(reference)| over the
/// frozen corpus (docs/parity.md "Perplexity gate"). Reads `reference-ppl.json`
/// (keepable, `tests/fixtures/reference-ppl.json`) and `candidate-ppl.json` from
/// `LAGUNA_PARITY_DIR`.
#[test]
#[ignore = "needs real dumps; run with LAGUNA_PARITY_DIR=<dir> cargo test --test parity ppl_parity -- --ignored"]
fn ppl_parity() -> Result<()> {
    let dir = parity_dir()?;
    let reference = load_ppl_dump(&dir.join("reference-ppl.json"))?;
    let candidate = load_ppl_dump(&dir.join("candidate-ppl.json"))?;
    ppl_compare(&reference, &candidate)
}

// --- Rejection-path unit tests (non-ignored: no model, no real dumps) ---------
//
// These exercise the soundness guards added to the decode gate and the dump
// loaders with synthetic in-memory / temp-file dumps. They MUST run in the plain
// `cargo test --test parity` (no `--ignored`), unlike the model-fed gate tests.

/// A `provenance` object with the given runner and otherwise-inert fields. The
/// attention dtype and combine path match how the gate script produces each side:
/// the Reference oracle runs under `LAGUNA_ATTN_F32=1` (f32) and records combine
/// "reference"; fused candidates run the shipped f16 default with the fused
/// combine. Strict-tier candidate tests override `attn_dtype`/`combine` to the
/// f32/classic path strict grades. Tests perturb the fields to hit rejection paths.
fn prov(moe_impl: &str) -> Provenance {
    Provenance {
        schema_version: laguna::parity_schema::PROVENANCE_SCHEMA_VERSION,
        moe_impl: moe_impl.to_string(),
        seq_len: 2,
        mm_variant: "tensor".to_string(),
        no_mm_id: false,
        mm_min_seq: 32,
        attn_dtype: Some(if moe_impl == "reference" { "f32" } else { "f16" }.to_string()),
        combine: Some(if moe_impl == "reference" { "reference" } else { "fused" }.to_string()),
        // Reference runs under LAGUNA_ATTN_F32 (f16 library bypassed); fused
        // candidates run the shipped tensor prefill kernel. Strict-tier candidate
        // tests override this to "f32-bypass".
        attn_mm: Some(if moe_impl == "reference" { "f32-bypass" } else { "tensor" }.to_string()),
        // Reference runs under LAGUNA_ATTN_GLUE_CLASSIC (the oracle executes the
        // attention glue too, anchored to the candle chains by the env pin);
        // fused candidates run the shipped fused glue kernels. Strict-tier
        // candidate tests override this to "classic".
        attn_glue: Some(if moe_impl == "reference" { "classic" } else { "fused" }.to_string()),
        // Both runners execute the shipped f16 sdpa kernel (the LAGUNA_SDPA_F32
        // experiment hook is opt-in and never blessed).
        sdpa: Some("f16".to_string()),
        // Reference runs under LAGUNA_FLASH_CLASSIC (the oracle's prefill is
        // pinned off the flash kernel); fused candidates run the shipped flash
        // prefill. Strict-tier candidate tests override this to "classic".
        flash: Some(if moe_impl == "reference" { "classic" } else { "fused" }.to_string()),
        // Reference's ReferenceExperts always runs the candle silu*mul chain;
        // fused candidates run the shipped fused ops::silu_mul kernel. Strict-tier
        // candidate tests override this to "classic".
        act: Some(if moe_impl == "reference" { "classic" } else { "fused" }.to_string()),
    }
}

/// A greedy-side step: it produced `token` (== its own top-1), with the given
/// top-1/top-2 logits. Finite, unit scale.
fn ref_step(token: u32, top1: (u32, f32), top2: (u32, f32)) -> Step {
    Step { token: Some(token), forced_token: None, top1, top2, l2: 20.0, nonfinite: 0 }
}

/// A replay-side step: teacher-forced along `forced`, recording this runner's own
/// top-1/top-2. Finite, unit scale.
fn cand_step(forced: u32, top1: (u32, f32), top2: (u32, f32)) -> Step {
    Step { token: None, forced_token: Some(forced), top1, top2, l2: 20.0, nonfinite: 0 }
}

/// A reference(greedy)/candidate(replay) pair that passes the gate cleanly, for
/// each rejection test to minimally perturb.
fn valid_pair() -> (GreedyDump, GreedyDump) {
    let prompt = vec![2u32, 100];
    let reference = GreedyDump {
        kind: "greedy".to_string(),
        prompt: prompt.clone(),
        steps: vec![ref_step(10, (10, 5.0), (11, 3.0)), ref_step(12, (12, 5.0), (13, 3.0))],
        provenance: Some(prov("reference")),
    };
    let candidate = GreedyDump {
        kind: "replay".to_string(),
        prompt,
        steps: vec![cand_step(10, (10, 5.0), (11, 3.0)), cand_step(12, (12, 5.0), (13, 3.0))],
        provenance: Some(prov("fused")),
    };
    (reference, candidate)
}

#[test]
fn greedy_gate_accepts_valid_dumps() {
    let (r, c) = valid_pair();
    greedy_compare(&r, &c).expect("a clean reference/replay pair should pass the gate");
}

#[test]
fn greedy_gate_rejects_reference_runner_candidate() {
    // A candidate replay produced with the default `--moe-impl reference` (the
    // forgotten-flag case) would compare the oracle against itself.
    let (r, mut c) = valid_pair();
    c.provenance = Some(prov("reference"));
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("expected \"fused\""), "unexpected error: {err}");
}

#[test]
fn greedy_gate_rejects_missing_provenance() {
    let (r, mut c) = valid_pair();
    c.provenance = None;
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("no provenance"), "unexpected error: {err}");
}

#[test]
fn greedy_gate_rejects_wrong_attn_dtype() {
    // The decode gate grades the shipped f16 attention path; a candidate replay
    // produced under LAGUNA_ATTN_F32=1 ran the legacy f32 path instead.
    let (r, mut c) = valid_pair();
    c.provenance.as_mut().unwrap().attn_dtype = Some("f32".to_string());
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("attn_dtype"), "unexpected error: {err}");
    // The reference oracle pins f32 attention; an f16 reference moved the anchor.
    let (mut r, c) = valid_pair();
    r.provenance.as_mut().unwrap().attn_dtype = Some("f16".to_string());
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("attn_dtype"), "unexpected error: {err}");
}

#[test]
fn greedy_gate_rejects_missing_attn_dtype() {
    // A provenance without attn_dtype comes from a stale `logits-dump` binary
    // predating the field — such dumps are otherwise indistinguishable from
    // current ones, so both sides must fail closed.
    let (mut r, c) = valid_pair();
    r.provenance.as_mut().unwrap().attn_dtype = None;
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("no attn_dtype"), "unexpected error: {err}");
    let (r, mut c) = valid_pair();
    c.provenance.as_mut().unwrap().attn_dtype = None;
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("no attn_dtype"), "unexpected error: {err}");
}

#[test]
fn greedy_gate_rejects_wrong_or_missing_combine() {
    // The decode gate grades the shipped fused combine; a candidate replay
    // produced under LAGUNA_COMBINE_CLASSIC=1 ran the classic candle chain.
    let (r, mut c) = valid_pair();
    c.provenance.as_mut().unwrap().combine = Some("classic".to_string());
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("combine"), "unexpected error: {err}");
    // A candidate provenance missing the combine field predates it — hard fail.
    let (r, mut c) = valid_pair();
    c.provenance.as_mut().unwrap().combine = None;
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("no combine"), "unexpected error: {err}");
    // The Reference oracle must record combine "reference"; a stale/missing field
    // (the oracle never dispatches ops::combine) is a hard fail.
    let (mut r, c) = valid_pair();
    r.provenance.as_mut().unwrap().combine = None;
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("no combine"), "unexpected error: {err}");
}

#[test]
fn greedy_gate_rejects_wrong_or_missing_attn_mm() {
    // The decode gate grades the shipped tensor prefill kernel; a candidate replay
    // produced under LAGUNA_ATTN_MM_CLASSIC ran the classic simdgroup kernel
    // instead.
    let (r, mut c) = valid_pair();
    c.provenance.as_mut().unwrap().attn_mm = Some("classic".to_string());
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("attn_mm"), "unexpected error: {err}");
    // A candidate provenance missing the attn_mm field predates it — hard fail.
    let (r, mut c) = valid_pair();
    c.provenance.as_mut().unwrap().attn_mm = None;
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("no attn_mm"), "unexpected error: {err}");
    // The Reference oracle runs under LAGUNA_ATTN_F32 (f16 library bypassed); a
    // "classic" (or stale/missing) reference moved the anchor.
    let (mut r, c) = valid_pair();
    r.provenance.as_mut().unwrap().attn_mm = None;
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("no attn_mm"), "unexpected error: {err}");
}

#[test]
fn greedy_gate_rejects_wrong_or_missing_attn_glue() {
    // The decode gate grades the shipped fused attention glue; a candidate replay
    // produced under LAGUNA_ATTN_GLUE_CLASSIC=1 ran the candle chains instead.
    let (r, mut c) = valid_pair();
    c.provenance.as_mut().unwrap().attn_glue = Some("classic".to_string());
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("attn_glue"), "unexpected error: {err}");
    // A candidate provenance missing the attn_glue field predates it — hard fail.
    let (r, mut c) = valid_pair();
    c.provenance.as_mut().unwrap().attn_glue = None;
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("no attn_glue"), "unexpected error: {err}");
    // The Reference oracle is pinned to the classic glue chains
    // (LAGUNA_ATTN_GLUE_CLASSIC=1); a "fused" (or stale/missing) reference moved
    // the anchor.
    let (mut r, c) = valid_pair();
    r.provenance.as_mut().unwrap().attn_glue = Some("fused".to_string());
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("attn_glue"), "unexpected error: {err}");
    let (mut r, c) = valid_pair();
    r.provenance.as_mut().unwrap().attn_glue = None;
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("no attn_glue"), "unexpected error: {err}");
}

/// A one-step reference/candidate pair for the near-tie excuse tests: the
/// reference produced 10 with the given top-2 logit, the candidate (forced with
/// 10) recorded the given top-1/top-2.
fn one_step_pair(
    ref_top2: (u32, f32),
    cand_top1: (u32, f32),
    cand_top2: (u32, f32),
) -> Result<()> {
    let prompt = vec![2u32];
    let reference = GreedyDump {
        kind: "greedy".to_string(),
        prompt: prompt.clone(),
        steps: vec![ref_step(10, (10, 5.0), ref_top2)],
        provenance: Some(prov("reference")),
    };
    let candidate = GreedyDump {
        kind: "replay".to_string(),
        prompt,
        steps: vec![cand_step(10, cand_top1, cand_top2)],
        provenance: Some(prov("fused")),
    };
    greedy_compare(&reference, &candidate)
}

#[test]
fn greedy_near_tie_excuses_only_contender_overlap() {
    // Reference near-tie (top1/top2 margin 0.2 < 0.5); candidate landed on the
    // reference's OTHER contender (top-2 = 11): excused.
    one_step_pair((11, 4.8), (11, 4.9), (12, 4.7))
        .expect("a near-tie contender mismatch should be excused");
    // Candidate picked an arbitrary token at the same near-tie, with no overlap
    // in either direction (its own top-2 doesn't rank the reference token
    // either): must fail.
    let err = one_step_pair((11, 4.8), (99, 4.9), (98, 4.7)).unwrap_err().to_string();
    assert!(err.contains("no contender overlap"), "unexpected error: {err}");
}

#[test]
fn greedy_near_tie_excuses_reference_in_candidate_top2() {
    // The widened clause: the candidate's pick (99) is outside the reference's
    // stored top-2, but the candidate ranks the reference token #2 itself AND is
    // flat on its own top-1/top-2 (margin 0.2 < 0.5). At a small reference
    // margin the oracle's stored top-2 is not always the true contender set, so
    // a tie that is flat on BOTH sides is tie noise: excused.
    one_step_pair((11, 4.8), (99, 4.9), (10, 4.7))
        .expect("reference token in the candidate's top-2 at a both-sides-flat near-tie should be excused");
    // The same disagreement at a decisive reference margin (2.0 >= 0.5) is a
    // real mismatch regardless of the candidate's ranking.
    let err = one_step_pair((11, 3.0), (99, 4.9), (10, 4.7)).unwrap_err().to_string();
    assert!(err.contains("margin"), "unexpected error: {err}");
}

#[test]
fn greedy_near_tie_rejects_confident_wrong_pick() {
    // Reference near-tie (margin 0.49 < 0.5) and the candidate ranks the
    // reference token #2 — but its own top-1 is an unrelated token a decisive
    // 5.0 logits clear of it. The reference-in-candidate-top-2 excuse covers
    // ties that are flat on BOTH sides (the fork-calibrated case had candidate
    // margin 0.068); a candidate that confidently promotes a non-contender is
    // never fork-class tie noise and must fail.
    let err = one_step_pair((11, 4.51), (99, 9.7), (10, 4.7)).unwrap_err().to_string();
    assert!(err.contains("confidently picked a non-contender"), "unexpected error: {err}");
    // The same contender geometry with the candidate also flat (margin 0.2)
    // stays excused.
    one_step_pair((11, 4.51), (99, 4.9), (10, 4.7))
        .expect("a both-sides-flat near-tie should stay excused");
}

#[test]
fn greedy_gate_caps_excused_near_ties() {
    // Every mismatch here is individually excusable (reference margin 0.2 and
    // the candidate picks the reference's #2 contender), but 4 excused of 24
    // steps exceeds the max(2, n/8) = 3 cap: fork calibration measured <= 4/64
    // tie-flips per fixture, so an engine flipping more than 1-in-8 steps is
    // drifting, not riding oracle tie noise.
    let n = 24;
    let prompt = vec![2u32];
    let mut ref_steps = Vec::new();
    let mut cand_steps = Vec::new();
    for i in 0..n {
        ref_steps.push(ref_step(10, (10, 5.0), (11, 4.8)));
        if i < 4 {
            cand_steps.push(cand_step(10, (11, 4.9), (10, 4.7)));
        } else {
            cand_steps.push(cand_step(10, (10, 5.0), (11, 4.8)));
        }
    }
    let reference = GreedyDump {
        kind: "greedy".to_string(),
        prompt: prompt.clone(),
        steps: ref_steps,
        provenance: Some(prov("reference")),
    };
    let candidate = GreedyDump {
        kind: "replay".to_string(),
        prompt,
        steps: cand_steps,
        provenance: Some(prov("fused")),
    };
    let err = greedy_compare(&reference, &candidate).unwrap_err().to_string();
    assert!(err.contains("excused near-ties exceed the cap"), "unexpected error: {err}");
}

#[test]
fn greedy_gate_rejects_nonfinite() {
    let (r, mut c) = valid_pair();
    c.steps[0].nonfinite = 1;
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("non-finite"), "unexpected error: {err}");
}

#[test]
fn greedy_gate_rejects_scale_blowout() {
    // Candidate/reference L2-norm ratio 1.5 is outside the 1.18 bound.
    let (r, mut c) = valid_pair();
    c.steps[0].l2 = r.steps[0].l2 * 1.5;
    let err = greedy_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("l2 ratio"), "unexpected error: {err}");
}

fn scratch_dir(tag: &str) -> PathBuf {
    let mut d = std::env::temp_dir();
    d.push(format!("laguna-parity-ut-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn load_greedy_dump_requires_per_step_scale_fields() {
    // A step lacking `l2` predates the per-step scale/finiteness fields and must
    // be a hard load failure (regenerate), not a silent exemption.
    let dir = scratch_dir("missing-l2");
    let dump = json!({
        "kind": "greedy",
        "tokens": [2, 3],
        "provenance": { "moe_impl": "reference", "seq_len": 2, "mm_variant": "tensor", "no_mm_id": false, "mm_min_seq": 32, "attn_dtype": "f32" },
        "steps": [ { "token": 10, "top1": [10, 5.0], "top2": [11, 3.0], "nonfinite": 0 } ],
    });
    let p = dir.join("greedy.json");
    std::fs::write(&p, serde_json::to_string(&dump).unwrap()).unwrap();
    let err = load_greedy_dump(&p).err().expect("missing `l2` should fail to load").to_string();
    assert!(err.contains("`l2`"), "unexpected error: {err}");
}

#[test]
fn load_dump_rejects_forged_top5() {
    // top5 lists token 2 twice; the real top-5 recomputed from `logits` is
    // [0,1,2,3,4], so the forged/stale list must be rejected at load.
    let dir = scratch_dir("forged-top5");
    let dump = json!({
        "tokens": [2, 3],
        "logits": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        "top1": 0,
        "top5": [[0, 5.0], [1, 4.0], [2, 3.0], [2, 2.0], [4, 1.0]],
    });
    let p = dir.join("forged.json");
    std::fs::write(&p, serde_json::to_string(&dump).unwrap()).unwrap();
    let err = load_dump(&p).err().expect("forged top5 should fail to load").to_string();
    assert!(err.contains("forged or stale"), "unexpected error: {err}");
}

/// A minimal full-logit dump carrying the given provenance (top-k consistent with
/// `logits`, so it passes the load-time recomputation).
fn tiny_dump(provenance: Option<Provenance>) -> Dump {
    Dump {
        tokens: vec![1],
        logits: vec![1.0, 0.5],
        top1: 0,
        top5: vec![(0, 1.0), (1, 0.5)],
        taps: vec![],
        provenance,
    }
}

#[test]
fn compare_requires_reference_provenance() {
    let candidate = tiny_dump(None);
    // Missing reference provenance (legacy reference.json) → hard fail, every tier.
    let err = compare(&candidate, &tiny_dump(None), Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("no provenance"), "unexpected error: {err}");
    // Reference accidentally produced with the fused runner → hard fail (a
    // fused-vs-fused self-comparison would hide a regression).
    let err = compare(&candidate, &tiny_dump(Some(prov("fused"))), Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("expected \"reference\""), "unexpected error: {err}");
}

#[test]
fn compare_requires_reference_attn_dtype() {
    let candidate = tiny_dump(Some(prov("fused")));
    // Reference regenerated WITHOUT LAGUNA_ATTN_F32 (f16 attention) → hard fail
    // in every tier: the oracle pins f32 attention compute.
    let mut f16_ref = prov("reference");
    f16_ref.attn_dtype = Some("f16".to_string());
    let err = compare(&candidate, &tiny_dump(Some(f16_ref)), Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("attn_dtype"), "unexpected error: {err}");
    // Reference written by a stale `logits-dump` binary predating the field —
    // otherwise indistinguishable from a current dump — must also hard-fail.
    let mut stale_ref = prov("reference");
    stale_ref.attn_dtype = None;
    let err = compare(&candidate, &tiny_dump(Some(stale_ref)), Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("no attn_dtype"), "unexpected error: {err}");
}

#[test]
fn compare_pins_candidate_attn_dtype_per_tier() {
    let reference = tiny_dump(Some(prov("reference")));

    // strict gates the legacy f32 attention path: an f16 candidate (the shipped
    // default, i.e. LAGUNA_ATTN_F32 forgotten) must be rejected.
    let err = compare(&tiny_dump(Some(prov("fused"))), &reference, Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("attn_dtype"), "unexpected error: {err}");
    // strict also requires candidate provenance at all (stale-binary dump).
    let err = compare(&tiny_dump(None), &reference, Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("no provenance"), "unexpected error: {err}");
    // A candidate provenance missing only the attn_dtype field is equally stale.
    let mut stale = prov("fused");
    stale.attn_dtype = None;
    let err = compare(&tiny_dump(Some(stale)), &reference, Tier::Decode).unwrap_err().to_string();
    assert!(err.contains("no attn_dtype"), "unexpected error: {err}");

    // mm and decode gate the shipped f16 default: an f32 candidate ran the
    // wrong path. (The mm candidate must be mm-active to reach the dtype check.)
    let mut p = prov("fused");
    p.seq_len = p.mm_min_seq;
    p.attn_dtype = Some("f32".to_string());
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Mm).unwrap_err().to_string();
    assert!(err.contains("attn_dtype"), "unexpected error: {err}");
    let mut p = prov("fused");
    p.attn_dtype = Some("f32".to_string());
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Decode).unwrap_err().to_string();
    assert!(err.contains("attn_dtype"), "unexpected error: {err}");
}

#[test]
fn compare_requires_reference_attn_mm() {
    let candidate = tiny_dump(Some(prov("fused")));
    // Reference regenerated WITHOUT LAGUNA_ATTN_F32 (its mm branch ran the shipped
    // tensor kernel instead of being bypassed) → hard fail every tier.
    let mut tensor_ref = prov("reference");
    tensor_ref.attn_mm = Some("tensor".to_string());
    let err = compare(&candidate, &tiny_dump(Some(tensor_ref)), Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("attn_mm"), "unexpected error: {err}");
    // Reference from a binary predating the attn_mm field must also hard-fail.
    let mut stale_ref = prov("reference");
    stale_ref.attn_mm = None;
    let err = compare(&candidate, &tiny_dump(Some(stale_ref)), Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("no attn_mm"), "unexpected error: {err}");
}

#[test]
fn compare_pins_candidate_attn_mm_per_tier() {
    let reference = tiny_dump(Some(prov("reference")));

    // strict runs under LAGUNA_ATTN_F32 (f16 library bypassed): a "tensor"
    // candidate (the shipped default, i.e. LAGUNA_ATTN_F32 forgotten) must be
    // rejected. It first clears strict's f32 attn_dtype pin to reach the attn_mm check.
    let mut p = prov("fused");
    p.attn_dtype = Some("f32".to_string());
    // attn_mm defaults to "tensor" — wrong for strict, which pins "f32-bypass".
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("attn_mm"), "unexpected error: {err}");

    // mm and decode grade the shipped tensor prefill kernel: a "classic"
    // (LAGUNA_ATTN_MM_CLASSIC) or "f32-bypass" candidate ran the wrong path. (The
    // mm candidate must be mm-active to reach the attn_mm check.)
    let mut p = prov("fused");
    p.seq_len = p.mm_min_seq;
    p.attn_mm = Some("classic".to_string());
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Mm).unwrap_err().to_string();
    assert!(err.contains("attn_mm"), "unexpected error: {err}");
    let mut p = prov("fused");
    p.attn_mm = Some("f32-bypass".to_string());
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Decode).unwrap_err().to_string();
    assert!(err.contains("attn_mm"), "unexpected error: {err}");

    // A candidate provenance missing only the attn_mm field is stale (predates it).
    let mut p = prov("fused");
    p.attn_mm = None;
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Decode).unwrap_err().to_string();
    assert!(err.contains("no attn_mm"), "unexpected error: {err}");
}

#[test]
fn compare_requires_reference_attn_glue() {
    let candidate = tiny_dump(Some(prov("fused")));
    // A reference regenerated WITHOUT LAGUNA_ATTN_GLUE_CLASSIC ran the fused glue
    // kernels instead of the candle chains the oracle is anchored to → hard fail
    // every tier.
    let mut fused_ref = prov("reference");
    fused_ref.attn_glue = Some("fused".to_string());
    let err = compare(&candidate, &tiny_dump(Some(fused_ref)), Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("attn_glue"), "unexpected error: {err}");
    // A reference from a binary predating the attn_glue field must also hard-fail.
    let mut stale_ref = prov("reference");
    stale_ref.attn_glue = None;
    let err = compare(&candidate, &tiny_dump(Some(stale_ref)), Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("no attn_glue"), "unexpected error: {err}");
}

#[test]
fn compare_pins_candidate_attn_glue_per_tier() {
    let reference = tiny_dump(Some(prov("reference")));

    // strict grades the candle glue chains: a candidate on the default fused glue
    // (LAGUNA_ATTN_GLUE_CLASSIC forgotten) must be rejected. It must first clear
    // the strict fused/no_mm_id + f32-attn + classic-combine checks to reach the
    // attn_glue pin.
    let mut p = prov("fused");
    p.attn_dtype = Some("f32".to_string());
    p.attn_mm = Some("f32-bypass".to_string());
    p.no_mm_id = true;
    p.combine = Some("classic".to_string());
    // attn_glue defaults to "fused" — wrong for strict.
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("attn_glue"), "unexpected error: {err}");

    // mm/decode grade the fused glue: a "classic" candidate ran the wrong path.
    let mut p = prov("fused");
    p.seq_len = p.mm_min_seq; // mm-active
    p.attn_glue = Some("classic".to_string());
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Mm).unwrap_err().to_string();
    assert!(err.contains("attn_glue"), "unexpected error: {err}");
    let mut p = prov("fused");
    p.attn_glue = Some("classic".to_string());
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Decode).unwrap_err().to_string();
    assert!(err.contains("attn_glue"), "unexpected error: {err}");

    // A candidate provenance missing only the attn_glue field is stale (predates it).
    let mut p = prov("fused");
    p.attn_glue = None;
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Decode).unwrap_err().to_string();
    assert!(err.contains("no attn_glue"), "unexpected error: {err}");
}

#[test]
fn compare_pins_candidate_sdpa_per_tier() {
    let reference = tiny_dump(Some(prov("reference")));

    // Every tier expects the shipped f16 sdpa kernel by default; a candidate
    // produced under LAGUNA_SDPA_F32 (the experiment hook) must be rejected
    // unless the gate run opts in via LAGUNA_PARITY_EXPECT_SDPA.
    let mut p = prov("fused");
    p.sdpa = Some("f32".to_string());
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Decode).unwrap_err().to_string();
    assert!(err.contains("sdpa"), "unexpected error: {err}");

    // The reference oracle always runs the f16 sdpa kernel; an f32 reference
    // moved the anchor.
    let mut r = prov("reference");
    r.sdpa = Some("f32".to_string());
    let err = compare(&tiny_dump(Some(prov("fused"))), &tiny_dump(Some(r)), Tier::Decode)
        .unwrap_err()
        .to_string();
    assert!(err.contains("sdpa"), "unexpected error: {err}");
}

#[test]
fn sdpa_missing_at_current_version_hard_fails() {
    // A dump claiming the current schema version but missing sdpa came from a
    // stale/doctored binary: hard fail, both sides.
    let reference = tiny_dump(Some(prov("reference")));
    let mut p = prov("fused");
    p.sdpa = None;
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Decode).unwrap_err().to_string();
    assert!(err.contains("no sdpa"), "unexpected error: {err}");

    let mut r = prov("reference");
    r.sdpa = None;
    let err = compare(&tiny_dump(Some(prov("fused"))), &tiny_dump(Some(r)), Tier::Decode)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no sdpa"), "unexpected error: {err}");
}

#[test]
fn sdpa_missing_grandfathered_at_schema_version_1() {
    // A version-1 dump (schema_version absent = 1) predates the sdpa field:
    // it resolves to the grandfather value "f16" and PASSES when f16 is
    // expected. This is the property that keeps every pre-versioning
    // reference/candidate dump valid across the field's introduction.
    let mut r = prov("reference");
    r.schema_version = 1;
    r.sdpa = None;
    let mut c = prov("fused");
    c.schema_version = 1;
    c.sdpa = None;
    compare(&tiny_dump(Some(c)), &tiny_dump(Some(r)), Tier::Decode)
        .expect("v1 dumps without sdpa must be grandfathered to f16 and pass");
}

#[test]
fn compare_pins_candidate_flash_per_tier() {
    let reference = tiny_dump(Some(prov("reference")));

    // strict grades the candle sdpa prefill: a candidate on the default flash
    // kernel (LAGUNA_FLASH_CLASSIC forgotten) must be rejected. It must first
    // clear every other strict pin to reach the flash check (the last one).
    let mut p = prov("fused");
    p.attn_dtype = Some("f32".to_string());
    p.attn_mm = Some("f32-bypass".to_string());
    p.no_mm_id = true;
    p.combine = Some("classic".to_string());
    p.attn_glue = Some("classic".to_string());
    // flash defaults to "fused" — wrong for strict.
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("flash"), "unexpected error: {err}");

    // mm/decode grade the fused flash prefill: a "classic" candidate ran the
    // wrong path (unless the gate run opts out via LAGUNA_PARITY_EXPECT_FLASH).
    let mut p = prov("fused");
    p.seq_len = p.mm_min_seq; // mm-active
    p.flash = Some("classic".to_string());
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Mm).unwrap_err().to_string();
    assert!(err.contains("flash"), "unexpected error: {err}");
    let mut p = prov("fused");
    p.flash = Some("classic".to_string());
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Decode).unwrap_err().to_string();
    assert!(err.contains("flash"), "unexpected error: {err}");

    // A reference that ran the flash kernel moved the oracle's anchor.
    let mut r = prov("reference");
    r.flash = Some("fused".to_string());
    let err = compare(&tiny_dump(Some(prov("fused"))), &tiny_dump(Some(r)), Tier::Decode)
        .unwrap_err()
        .to_string();
    assert!(err.contains("flash"), "unexpected error: {err}");
}

#[test]
fn flash_missing_at_current_version_hard_fails() {
    // A dump claiming the current schema version but missing flash came from a
    // stale/doctored binary: hard fail, both sides.
    let reference = tiny_dump(Some(prov("reference")));
    let mut p = prov("fused");
    p.flash = None;
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Decode).unwrap_err().to_string();
    assert!(err.contains("no flash"), "unexpected error: {err}");

    let mut r = prov("reference");
    r.flash = None;
    let err = compare(&tiny_dump(Some(prov("fused"))), &tiny_dump(Some(r)), Tier::Decode)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no flash"), "unexpected error: {err}");
}

#[test]
fn flash_missing_grandfathered_at_schema_version_2() {
    // A v2 REFERENCE dump (written before the flash field existed) resolves the
    // missing field to the grandfather "classic" — exactly what the reference
    // pin expects, so pre-flash reference dumps stay valid. The current-build
    // candidate carries flash "fused" and passes the decode expectation.
    let mut r = prov("reference");
    r.schema_version = 2;
    r.flash = None;
    compare(&tiny_dump(Some(prov("fused"))), &tiny_dump(Some(r)), Tier::Decode)
        .expect("v2 references without flash must be grandfathered to classic and pass");
}

#[test]
fn compare_pins_candidate_act_per_tier() {
    let reference = tiny_dump(Some(prov("reference")));

    // strict grades the candle silu*mul chain: a candidate on the default fused
    // activation kernel (LAGUNA_ACT_CLASSIC forgotten) must be rejected. It must
    // first clear every other strict pin to reach the act check (the last one).
    let mut p = prov("fused");
    p.attn_dtype = Some("f32".to_string());
    p.attn_mm = Some("f32-bypass".to_string());
    p.no_mm_id = true;
    p.combine = Some("classic".to_string());
    p.attn_glue = Some("classic".to_string());
    p.flash = Some("classic".to_string());
    // act defaults to "fused" — wrong for strict.
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("act"), "unexpected error: {err}");

    // mm/decode grade the fused activation kernel: a "classic" candidate ran the
    // candle chain instead.
    let mut p = prov("fused");
    p.seq_len = p.mm_min_seq; // mm-active
    p.act = Some("classic".to_string());
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Mm).unwrap_err().to_string();
    assert!(err.contains("act"), "unexpected error: {err}");
    let mut p = prov("fused");
    p.act = Some("classic".to_string());
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Decode).unwrap_err().to_string();
    assert!(err.contains("act"), "unexpected error: {err}");

    // A reference that recorded the fused activation moved the oracle's anchor
    // (the Reference runner must always be the candle chain → "classic").
    let mut r = prov("reference");
    r.act = Some("fused".to_string());
    let err = compare(&tiny_dump(Some(prov("fused"))), &tiny_dump(Some(r)), Tier::Decode)
        .unwrap_err()
        .to_string();
    assert!(err.contains("act"), "unexpected error: {err}");
}

#[test]
fn act_missing_at_current_version_hard_fails() {
    // A dump claiming the current schema version but missing act came from a
    // stale/doctored binary: hard fail, both sides.
    let reference = tiny_dump(Some(prov("reference")));
    let mut p = prov("fused");
    p.act = None;
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Decode).unwrap_err().to_string();
    assert!(err.contains("no act"), "unexpected error: {err}");

    let mut r = prov("reference");
    r.act = None;
    let err = compare(&tiny_dump(Some(prov("fused"))), &tiny_dump(Some(r)), Tier::Decode)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no act"), "unexpected error: {err}");
}

#[test]
fn act_missing_grandfathered_at_schema_version_3() {
    // A v3 REFERENCE dump (written before the act field existed) resolves the
    // missing field to the grandfather "classic" — exactly what the reference
    // pin expects, so pre-act reference dumps stay valid. The current-build
    // candidate carries act "fused" and passes the decode expectation.
    let mut r = prov("reference");
    r.schema_version = 3;
    r.act = None;
    compare(&tiny_dump(Some(prov("fused"))), &tiny_dump(Some(r)), Tier::Decode)
        .expect("v3 references without act must be grandfathered to classic and pass");
}

#[test]
fn v1_baseline_fields_hard_fail_missing_at_any_version() {
    // The version-1 baseline fields have no grandfather: missing is a hard
    // fail even from a dump that (correctly) claims schema version 1 — the
    // just-regenerated references all carry them, so a v1 dump without one is
    // genuinely stale, not merely old.
    let candidate = tiny_dump(Some(prov("fused")));
    let mut r = prov("reference");
    r.schema_version = 1;
    r.attn_dtype = None;
    let err = compare(&candidate, &tiny_dump(Some(r)), Tier::Decode).unwrap_err().to_string();
    assert!(err.contains("no attn_dtype"), "unexpected error: {err}");

    let mut r = prov("reference");
    r.schema_version = 1;
    r.attn_glue = None;
    let err = compare(&candidate, &tiny_dump(Some(r)), Tier::Decode).unwrap_err().to_string();
    assert!(err.contains("no attn_glue"), "unexpected error: {err}");
}

#[test]
fn load_dump_defaults_missing_schema_version_to_v1() {
    // A dump written before schema versioning: full v1 provenance, no
    // schema_version, no sdpa. It must load as version 1 and clear the
    // reference-side sdpa pin via the grandfather value — the exact shape of
    // the cached /tmp/laguna-parity references and the committed ppl fixture.
    let dir = scratch_dir("v1-provenance");
    let dump = json!({
        "tokens": [2, 3],
        "logits": [5.0, 4.0],
        "top1": 0,
        "top5": [[0, 5.0], [1, 4.0]],
        "provenance": {
            "moe_impl": "reference", "seq_len": 2, "mm_variant": "tensor",
            "no_mm_id": false, "mm_min_seq": 32, "attn_dtype": "f32",
            "combine": "reference", "attn_mm": "f32-bypass", "attn_glue": "classic",
        },
    });
    let p = dir.join("v1.json");
    std::fs::write(&p, serde_json::to_string(&dump).unwrap()).unwrap();
    let d = load_dump(&p).expect("a v1-shaped dump must still load");
    let prov = d.provenance.expect("provenance must parse");
    assert_eq!(prov.schema_version, 1);
    assert_eq!(prov.sdpa, None);
    check_sdpa(&prov, "reference dump", "f16")
        .expect("missing sdpa at v1 must grandfather to f16");
    assert_eq!(prov.flash, None);
    check_flash(&prov, "reference dump", "classic")
        .expect("missing flash at v1 must grandfather to classic");
    assert_eq!(prov.act, None);
    check_act(&prov, "reference dump", "classic")
        .expect("missing act at v1 must grandfather to classic");
}

#[test]
fn load_dump_rejects_future_schema_version() {
    // A dump claiming a schema version newer than this test binary knows
    // cannot have its missing fields resolved soundly: fail closed.
    let dir = scratch_dir("future-schema");
    let dump = json!({
        "tokens": [2],
        "logits": [5.0, 4.0],
        "top1": 0,
        "top5": [[0, 5.0], [1, 4.0]],
        "provenance": {
            "schema_version": laguna::parity_schema::PROVENANCE_SCHEMA_VERSION + 1,
            "moe_impl": "reference", "seq_len": 1, "mm_variant": "tensor",
            "no_mm_id": false, "mm_min_seq": 32, "attn_dtype": "f32",
            "combine": "reference", "attn_mm": "f32-bypass", "attn_glue": "classic",
        },
    });
    let p = dir.join("future.json");
    std::fs::write(&p, serde_json::to_string(&dump).unwrap()).unwrap();
    let err = load_dump(&p).err().expect("future schema_version must fail").to_string();
    assert!(err.contains("newer than this gate"), "unexpected error: {err}");
}

/// The committed ppl reference fixture must stay valid under the current
/// checks without regeneration — the whole point of schema versioning. This
/// runs the exact reference-side pins `ppl_compare` applies (it is v1-shaped:
/// no schema_version, no sdpa → grandfathered "f16", no flash → grandfathered
/// "classic").
#[test]
fn committed_ppl_reference_fixture_stays_valid() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/reference-ppl.json");
    let d = load_ppl_dump(&fixture).expect("committed fixture must load");
    assert_eq!(d.kind, "ppl");
    let p = d.provenance.expect("committed fixture carries provenance");
    assert_eq!(p.moe_impl, "reference");
    check_attn_dtype(&p, "reference-ppl.json", "f32").unwrap();
    check_combine(&p, "reference-ppl.json", "reference").unwrap();
    check_attn_mm(&p, "reference-ppl.json", "f32-bypass").unwrap();
    check_attn_glue(&p, "reference-ppl.json", "classic").unwrap();
    check_sdpa(&p, "reference-ppl.json", "f16").unwrap();
    check_flash(&p, "reference-ppl.json", "classic").unwrap();
}

#[test]
fn compare_requires_reference_combine() {
    let candidate = tiny_dump(Some(prov("fused")));
    // A reference whose combine is not "reference" (the Reference runner never
    // dispatches ops::combine) came from a wrong build/env → hard fail every tier.
    let mut wrong = prov("reference");
    wrong.combine = Some("fused".to_string());
    let err = compare(&candidate, &tiny_dump(Some(wrong)), Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("combine"), "unexpected error: {err}");
    // A reference from a binary predating the combine field must also hard-fail.
    let mut stale = prov("reference");
    stale.combine = None;
    let err = compare(&candidate, &tiny_dump(Some(stale)), Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("no combine"), "unexpected error: {err}");
}

#[test]
fn compare_pins_candidate_combine_per_tier() {
    let reference = tiny_dump(Some(prov("reference")));

    // strict grades the classic candle combine: a fused candidate on the default
    // fused combine (LAGUNA_COMBINE_CLASSIC forgotten) must be rejected. It must
    // first clear the strict fused/no_mm_id + f32-attn checks to reach the combine pin.
    let mut p = prov("fused");
    p.attn_dtype = Some("f32".to_string());
    p.attn_mm = Some("f32-bypass".to_string());
    p.no_mm_id = true;
    // combine defaults to "fused" — wrong for strict.
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("combine"), "unexpected error: {err}");

    // mm/decode grade the fused combine: a "classic" candidate ran the wrong path.
    let mut p = prov("fused");
    p.seq_len = p.mm_min_seq; // mm-active
    p.combine = Some("classic".to_string());
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Mm).unwrap_err().to_string();
    assert!(err.contains("combine"), "unexpected error: {err}");
    let mut p = prov("fused");
    p.combine = Some("classic".to_string());
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Decode).unwrap_err().to_string();
    assert!(err.contains("combine"), "unexpected error: {err}");

    // A candidate provenance missing only the combine field is stale (predates it).
    let mut p = prov("fused");
    p.combine = None;
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Decode).unwrap_err().to_string();
    assert!(err.contains("no combine"), "unexpected error: {err}");
}

#[test]
fn compare_strict_requires_fused_no_mm_id_candidate() {
    let reference = tiny_dump(Some(prov("reference")));

    // A copied reference dump graded as its own candidate (moe_impl
    // "reference", attn_dtype "f32" — exactly what strict's attn pin expects)
    // would clear the strict thresholds vacuously: rejected.
    let err = compare(&tiny_dump(Some(prov("reference"))), &reference, Tier::Strict)
        .unwrap_err()
        .to_string();
    assert!(err.contains("expected \"fused\""), "unexpected error: {err}");

    // A fused candidate that ran WITHOUT LAGUNA_NO_MM_ID (no_mm_id false) took
    // the mm_id path and belongs under the mm tier, not strict.
    let mut p = prov("fused");
    p.attn_dtype = Some("f32".to_string());
    p.attn_mm = Some("f32-bypass".to_string());
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Strict).unwrap_err().to_string();
    assert!(err.contains("no_mm_id"), "unexpected error: {err}");
}

#[test]
fn compare_mm_requires_mm_active_candidate() {
    let reference = tiny_dump(Some(prov("reference")));

    // No candidate provenance at all (stale `logits-dump` binary): rejected.
    let err = compare(&tiny_dump(None), &reference, Tier::Mm).unwrap_err().to_string();
    assert!(err.contains("requires candidate provenance"), "unexpected error: {err}");

    // Fused candidate with LAGUNA_NO_MM_ID set (mm_id force-disabled — the
    // classic fallback path): not an mm_id run, rejected.
    let mut p = prov("fused");
    p.seq_len = p.mm_min_seq;
    p.no_mm_id = true;
    let err = compare(&tiny_dump(Some(p)), &reference, Tier::Mm).unwrap_err().to_string();
    assert!(err.contains("NOT active"), "unexpected error: {err}");

    // Fused candidate whose prompt was below the mm_id seq threshold (a
    // decode-style dump that ran mv_id): rejected. prov() defaults seq_len 2
    // against mm_min_seq 32.
    let err = compare(&tiny_dump(Some(prov("fused"))), &reference, Tier::Mm)
        .unwrap_err()
        .to_string();
    assert!(err.contains("NOT active"), "unexpected error: {err}");
}

// --- Perplexity-gate rejection-path unit tests --------------------------------

/// A ppl dump; `token_hash` is derived from the ids so equal ids ⇒ equal hash,
/// matching the invariant the binary's real FNV hash preserves.
fn ppl_dump(kind: &str, moe_impl: &str, mean_nll: f64, tokens: Vec<u32>, nonfinite: u64) -> PplDump {
    let token_hash = format!("{tokens:?}");
    PplDump {
        kind: kind.to_string(),
        n_tokens: tokens.len(),
        tokens,
        token_hash,
        mean_nll,
        nonfinite,
        provenance: Some(prov(moe_impl)),
    }
}

/// A reference(ppl)/candidate(ppl) pair that passes the gate cleanly.
fn valid_ppl_pair() -> (PplDump, PplDump) {
    let tokens = vec![2u32, 100, 200, 300];
    let reference = ppl_dump("ppl", "reference", 2.5, tokens.clone(), 0);
    let candidate = ppl_dump("ppl", "fused", 2.5, tokens, 0);
    (reference, candidate)
}

#[test]
fn ppl_gate_accepts_valid_dumps() {
    let (r, c) = valid_ppl_pair();
    ppl_compare(&r, &c).expect("a clean reference/candidate ppl pair should pass");
}

#[test]
fn ppl_gate_accepts_delta_within_bound() {
    let (r, mut c) = valid_ppl_pair();
    // Half the frozen bound: comfortably inside.
    c.mean_nll = r.mean_nll + PPL_NLL_DELTA_MAX * 0.5;
    ppl_compare(&r, &c).expect("a sub-bound delta should pass");
}

#[test]
fn ppl_gate_rejects_delta_over_bound() {
    let (r, mut c) = valid_ppl_pair();
    // Twice the frozen bound: must fail whatever the frozen value is.
    c.mean_nll = r.mean_nll + PPL_NLL_DELTA_MAX * 2.0;
    let err = ppl_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("mean-NLL delta"), "unexpected error: {err}");
}

#[test]
fn ppl_gate_rejects_reference_runner_candidate() {
    // Candidate produced with the default `--moe-impl reference` self-compares.
    let (r, mut c) = valid_ppl_pair();
    c.provenance = Some(prov("reference"));
    let err = ppl_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("expected \"fused\""), "unexpected error: {err}");
}

#[test]
fn ppl_gate_rejects_missing_provenance() {
    let (r, mut c) = valid_ppl_pair();
    c.provenance = None;
    let err = ppl_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("no provenance"), "unexpected error: {err}");
}

#[test]
fn ppl_gate_rejects_wrong_or_missing_attn_dtype() {
    // The ppl gate grades the shipped f16 attention path; a candidate produced
    // under LAGUNA_ATTN_F32=1 ran the legacy f32 path instead.
    let (r, mut c) = valid_ppl_pair();
    c.provenance.as_mut().unwrap().attn_dtype = Some("f32".to_string());
    let err = ppl_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("attn_dtype"), "unexpected error: {err}");
    // A reference from a stale `logits-dump` binary predating the field must
    // hard-fail (the oracle pins f32 attention; the field proves it).
    let (mut r, c) = valid_ppl_pair();
    r.provenance.as_mut().unwrap().attn_dtype = None;
    let err = ppl_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("no attn_dtype"), "unexpected error: {err}");
}

#[test]
fn ppl_gate_rejects_wrong_or_missing_combine() {
    // The ppl gate grades the shipped fused combine; a candidate produced under
    // LAGUNA_COMBINE_CLASSIC=1 ran the classic candle chain instead.
    let (r, mut c) = valid_ppl_pair();
    c.provenance.as_mut().unwrap().combine = Some("classic".to_string());
    let err = ppl_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("combine"), "unexpected error: {err}");
    // A reference from a binary predating the combine field must hard-fail (the
    // oracle records combine "reference"; the field proves it).
    let (mut r, c) = valid_ppl_pair();
    r.provenance.as_mut().unwrap().combine = None;
    let err = ppl_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("no combine"), "unexpected error: {err}");
}

#[test]
fn ppl_gate_rejects_wrong_or_missing_act() {
    // The ppl gate grades the shipped fused activation; a candidate produced under
    // LAGUNA_ACT_CLASSIC=1 ran the candle silu*mul chain instead.
    let (r, mut c) = valid_ppl_pair();
    c.provenance.as_mut().unwrap().act = Some("classic".to_string());
    let err = ppl_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("act"), "unexpected error: {err}");
    // A reference from a binary at the current schema version but missing the act
    // field is stale/doctored and must hard-fail (the oracle records act "classic";
    // the field proves it).
    let (mut r, c) = valid_ppl_pair();
    r.provenance.as_mut().unwrap().act = None;
    let err = ppl_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("no act"), "unexpected error: {err}");
}

#[test]
fn ppl_gate_rejects_wrong_or_missing_attn_mm() {
    // The ppl gate grades the shipped tensor prefill kernel; a candidate produced
    // under LAGUNA_ATTN_MM_CLASSIC ran the classic simdgroup kernel instead.
    let (r, mut c) = valid_ppl_pair();
    c.provenance.as_mut().unwrap().attn_mm = Some("classic".to_string());
    let err = ppl_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("attn_mm"), "unexpected error: {err}");
    // A reference from a binary predating the attn_mm field must hard-fail (the
    // oracle runs under LAGUNA_ATTN_F32, recording "f32-bypass"; the field proves it).
    let (mut r, c) = valid_ppl_pair();
    r.provenance.as_mut().unwrap().attn_mm = None;
    let err = ppl_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("no attn_mm"), "unexpected error: {err}");
}

#[test]
fn ppl_gate_rejects_nonfinite() {
    let (r, mut c) = valid_ppl_pair();
    c.nonfinite = 1;
    let err = ppl_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("non-finite"), "unexpected error: {err}");
}

#[test]
fn ppl_gate_rejects_token_stream_mismatch() {
    // Different scored token stream ⇒ comparing perplexities of different corpora.
    let (r, mut c) = valid_ppl_pair();
    c.tokens = vec![2u32, 100, 200, 999];
    c.n_tokens = c.tokens.len();
    c.token_hash = format!("{:?}", c.tokens);
    let err = ppl_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("token_hash differs") || err.contains("token id streams differ"), "unexpected error: {err}");
}

#[test]
fn ppl_gate_rejects_wrong_kind() {
    let (mut r, c) = valid_ppl_pair();
    r.kind = "greedy".to_string();
    let err = ppl_compare(&r, &c).unwrap_err().to_string();
    assert!(err.contains("must be kind `ppl`"), "unexpected error: {err}");
}
