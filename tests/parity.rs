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
use serde_json::Value;

const COS_MIN_STRICT: f64 = 0.999;
const COS_MIN_MM: f64 = 0.995;
const NEAR_TIE_MARGIN: f64 = 0.5;
const TOP5_OVERLAP_MIN: usize = 4;

#[derive(Clone, Copy, PartialEq)]
enum Tier {
    Strict,
    Mm,
}

impl Tier {
    fn from_env() -> Result<Tier> {
        match std::env::var("LAGUNA_PARITY_TIER").as_deref() {
            Ok("mm") => Ok(Tier::Mm),
            Ok("strict") | Err(_) => Ok(Tier::Strict),
            Ok(other) => bail!("LAGUNA_PARITY_TIER must be `strict` or `mm`, got {other:?}"),
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
        .map(|x| x.as_u64().map(|n| n as u32).context("non-integer in array"))
        .collect()
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
        top1: v["top1"].as_u64().context("`top1` missing")? as u32,
        top5,
        taps,
    })
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

fn compare(candidate: &Dump, reference: &Dump, tier: Tier) -> Result<()> {
    let mut failures: Vec<String> = Vec::new();

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

    let cos_min = match tier {
        Tier::Strict => COS_MIN_STRICT,
        Tier::Mm => COS_MIN_MM,
    };
    let cos = cosine(&candidate.logits, &reference.logits);
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

    let overlap = candidate
        .top5
        .iter()
        .filter(|(t, _)| reference.top5.iter().any(|(r, _)| r == t))
        .count();
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
