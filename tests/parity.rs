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
/// actually exercised the mm_id prefill path it is being graded against.
struct Provenance {
    moe_impl: String,
    seq_len: usize,
    mm_variant: String,
    no_mm_id: bool,
    mm_min_seq: usize,
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
            moe_impl: p["moe_impl"].as_str().context("provenance missing `moe_impl`")?.to_string(),
            seq_len: p["seq_len"].as_u64().context("provenance missing `seq_len`")? as usize,
            mm_variant: p["mm_variant"].as_str().context("provenance missing `mm_variant`")?.to_string(),
            no_mm_id: p["no_mm_id"].as_bool().context("provenance missing `no_mm_id`")?,
            mm_min_seq: p["mm_min_seq"].as_u64().context("provenance missing `mm_min_seq`")? as usize,
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

fn compare(candidate: &Dump, reference: &Dump, tier: Tier) -> Result<()> {
    let mut failures: Vec<String> = Vec::new();

    // The reference dump must be a genuine Reference-oracle run in EVERY tier: a
    // `reference.json` accidentally produced with `--moe-impl fused` would turn
    // every tier into a fused-vs-fused self-comparison that hides a regression.
    // Fail closed — a missing/wrong reference provenance is a hard fail asking for
    // a regenerate, not a legacy exception (the reference side is what we grade
    // against, so it must be trustworthy). See docs/parity.md §3.
    match &reference.provenance {
        Some(p) if p.moe_impl == "reference" => {}
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
    // missing or non-mm provenance is a hard fail (regenerate the dump). Strict
    // and decode tiers add no requirement: the strict gate over-rejects mm output,
    // which is safe.
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
/// `NEAR_TIE_MARGIN`), i.e. which token wins there is oracle noise.
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
        Some(p) if p.moe_impl == "reference" => {}
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
        Some(p) if p.moe_impl == "fused" => {}
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
        // top-1/top-2 margin is below NEAR_TIE_MARGIN AND the candidate picked one
        // of the reference's two contenders (its top-1 or top-2). An arbitrary
        // wrong token at a reference near-tie is still a failure — the near-tie
        // only excuses picking the OTHER contender, not any token. (Since the
        // agreement case above already handled `cand_token == ref_token == top1`,
        // "contender" here means the candidate landed on the reference's top-2.)
        let ref_margin = (r.top1.1 - r.top2.1).abs() as f64;
        let cand_is_contender = cand_token == r.top1.0 || cand_token == r.top2.0;
        if ref_margin < NEAR_TIE_MARGIN && cand_is_contender {
            near_ties += 1;
            eprintln!(
                "  step {i}: excused near-tie — candidate {cand_token} vs reference {ref_token}, \
                 reference top1/top2 margin {ref_margin:.4} < {NEAR_TIE_MARGIN}"
            );
            continue;
        }
        let reason = if !cand_is_contender {
            format!("candidate {cand_token} is not a reference contender (top1 {}, top2 {})", r.top1.0, r.top2.0)
        } else {
            format!("reference top1/top2 margin {ref_margin:.4} >= {NEAR_TIE_MARGIN}")
        };
        failures.push(format!(
            "step {i}: candidate {cand_token} vs reference {ref_token}, {reason}"
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

// --- Rejection-path unit tests (non-ignored: no model, no real dumps) ---------
//
// These exercise the soundness guards added to the decode gate and the dump
// loaders with synthetic in-memory / temp-file dumps. They MUST run in the plain
// `cargo test --test parity` (no `--ignored`), unlike the model-fed gate tests.

/// A `provenance` object with the given runner and otherwise-inert fields.
fn prov(moe_impl: &str) -> Provenance {
    Provenance {
        moe_impl: moe_impl.to_string(),
        seq_len: 2,
        mm_variant: "tensor".to_string(),
        no_mm_id: false,
        mm_min_seq: 32,
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
fn greedy_near_tie_excuses_only_a_contender() {
    // One reference step is a genuine near-tie (top1/top2 margin 0.2 < 0.5).
    let build = |cand_top1: (u32, f32)| {
        let prompt = vec![2u32];
        let reference = GreedyDump {
            kind: "greedy".to_string(),
            prompt: prompt.clone(),
            steps: vec![ref_step(10, (10, 5.0), (11, 4.8))],
            provenance: Some(prov("reference")),
        };
        let candidate = GreedyDump {
            kind: "replay".to_string(),
            prompt,
            steps: vec![cand_step(10, cand_top1, (10, 4.7))],
            provenance: Some(prov("fused")),
        };
        greedy_compare(&reference, &candidate)
    };
    // Candidate landed on the reference's OTHER contender (top-2 = 11): excused.
    build((11, 4.9)).expect("a near-tie contender mismatch should be excused");
    // Candidate landed on an arbitrary token (99) at the same near-tie: must fail.
    let err = build((99, 4.9)).unwrap_err().to_string();
    assert!(err.contains("not a reference contender"), "unexpected error: {err}");
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
        "provenance": { "moe_impl": "reference", "seq_len": 2, "mm_variant": "tensor", "no_mm_id": false, "mm_min_seq": 32 },
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
