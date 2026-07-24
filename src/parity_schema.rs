//! Single source of truth for the parity-dump `provenance` schema shared by the
//! writer (`src/bin/logits-dump.rs`) and the readers (`tests/parity.rs`,
//! `scripts/parity-gate.ts` mirrors it in TypeScript).
//!
//! Why versioning exists: the parity gate treats a missing provenance field as
//! a hard "stale binary" failure — the only sound default, since a dump from a
//! binary predating a field is otherwise indistinguishable from a current one.
//! But without versioning, ADDING a field retroactively invalidates every
//! cached/committed reference dump (each regeneration is ~40 min of GPU time).
//! The `schema_version` stamped into each dump resolves the ambiguity: a field
//! missing from a dump whose version PREDATES the field's introduction takes
//! the field's grandfather value (the only value any binary of that era could
//! have produced), while a field missing at/after its introduction version is
//! still the stale-binary hard failure.
//!
//! A dump with no `schema_version` at all is version 1: the baseline field set
//! written before versioning existed. Every version-1 field is REQUIRED with no
//! grandfather — the references regenerated at that era carry all of them, and
//! grandfathering any would let a genuinely stale dump pass.

/// The schema version the current `logits-dump` writes.
pub const PROVENANCE_SCHEMA_VERSION: u32 = 5;

/// One provenance field's introduction record.
pub struct ProvenanceField {
    pub name: &'static str,
    /// First schema version whose dumps carry this field.
    pub introduced: u32,
    /// Value a dump from a pre-`introduced` schema version is resolved to when
    /// the field is missing (the only value binaries of that era could produce).
    /// `None` for the version-1 baseline fields: they are required outright.
    pub grandfather: Option<&'static str>,
}

/// Field-introduction table, one entry per provenance field.
/// Version 1 (baseline, all required): moe_impl, seq_len, mm_variant,
/// mm_min_seq, no_mm_id, attn_dtype, combine, attn_mm, attn_glue.
/// Version 2: sdpa (the sdpa compute dtype; every earlier binary ran the f16
/// sdpa kernel unconditionally, hence grandfather "f16").
/// Version 3: flash (the prefill attention kernel; every earlier binary ran
/// the candle sdpa prefill — today's classic path — hence grandfather
/// "classic").
/// Version 4: act (the routed-expert SwiGLU activation kernel; every earlier
/// binary ran candle's `silu(gate) * up` chain — today's classic path — hence
/// grandfather "classic").
/// Version 5: attn_decode (the attention DECODE-projection path: "q8" for a
/// q8_0-quantized checkpoint's vendored decode gemv, "f16" for the dense f16
/// gemv, "f32-bypass" under LAGUNA_ATTN_F32). Grandfather "f32-bypass": like every
/// post-baseline field, the grandfather is the value the Reference oracle carries
/// (the gate enforces attn_decode on the reference side), and the oracle runs
/// under LAGUNA_ATTN_F32 — the whole attention block is the dequant-f32 QMatMul —
/// so every pre-v5 reference dump ran the f32-bypass decode path. Candidate dumps
/// are always freshly generated at the current version and never rely on the
/// grandfather; only the cached (committed) reference dumps do, and those are all
/// f32-bypass, so grandfathering to it keeps them valid.
pub const PROVENANCE_FIELDS: &[ProvenanceField] = &[
    ProvenanceField { name: "moe_impl", introduced: 1, grandfather: None },
    ProvenanceField { name: "seq_len", introduced: 1, grandfather: None },
    ProvenanceField { name: "mm_variant", introduced: 1, grandfather: None },
    ProvenanceField { name: "mm_min_seq", introduced: 1, grandfather: None },
    ProvenanceField { name: "no_mm_id", introduced: 1, grandfather: None },
    ProvenanceField { name: "attn_dtype", introduced: 1, grandfather: None },
    ProvenanceField { name: "combine", introduced: 1, grandfather: None },
    ProvenanceField { name: "attn_mm", introduced: 1, grandfather: None },
    ProvenanceField { name: "attn_glue", introduced: 1, grandfather: None },
    ProvenanceField { name: "sdpa", introduced: 2, grandfather: Some("f16") },
    ProvenanceField { name: "flash", introduced: 3, grandfather: Some("classic") },
    ProvenanceField { name: "act", introduced: 4, grandfather: Some("classic") },
    ProvenanceField { name: "attn_decode", introduced: 5, grandfather: Some("f32-bypass") },
];

/// Look up a field's introduction record by name.
pub fn field(name: &str) -> Option<&'static ProvenanceField> {
    PROVENANCE_FIELDS.iter().find(|f| f.name == name)
}

/// Resolve a field MISSING from a dump of the given `schema_version`:
/// `Some(grandfather)` when the dump predates the field's introduction (the
/// dump stays valid), `None` when the dump's version says the field should be
/// present (stale binary — the caller hard-fails). Unknown field names resolve
/// to `None` (fail closed).
pub fn resolve_missing(name: &str, schema_version: u32) -> Option<&'static str> {
    let f = field(name)?;
    if schema_version < f.introduced { f.grandfather } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Table invariants the resolution logic relies on: unique names, every
    /// introduction within the current version, versions starting at 1, and a
    /// grandfather value for every field introduced after the baseline (a
    /// post-baseline field without one could never be grandfathered, silently
    /// reintroducing the regen tax this module exists to kill).
    #[test]
    fn table_is_well_formed() {
        for (i, f) in PROVENANCE_FIELDS.iter().enumerate() {
            assert!(f.introduced >= 1, "{}: introduced {} < 1", f.name, f.introduced);
            assert!(
                f.introduced <= PROVENANCE_SCHEMA_VERSION,
                "{}: introduced {} > current version {PROVENANCE_SCHEMA_VERSION}",
                f.name,
                f.introduced
            );
            if f.introduced > 1 {
                assert!(f.grandfather.is_some(), "{}: post-baseline field needs a grandfather", f.name);
            }
            for other in &PROVENANCE_FIELDS[..i] {
                assert_ne!(f.name, other.name, "duplicate field {}", f.name);
            }
        }
    }

    #[test]
    fn resolve_missing_grandfathers_only_predating_versions() {
        // sdpa introduced at v2: a v1 dump missing it resolves to "f16" ...
        assert_eq!(resolve_missing("sdpa", 1), Some("f16"));
        // ... while missing at/after v2 stays a hard failure.
        assert_eq!(resolve_missing("sdpa", 2), None);
        // flash introduced at v3: v1/v2 dumps missing it resolve to "classic"
        // (their binaries ran the candle sdpa prefill); missing at v3 fails.
        assert_eq!(resolve_missing("flash", 1), Some("classic"));
        assert_eq!(resolve_missing("flash", 2), Some("classic"));
        assert_eq!(resolve_missing("flash", 3), None);
        // act introduced at v4: v1..v3 dumps missing it resolve to "classic"
        // (their binaries ran candle's silu*mul chain); missing at v4 fails.
        assert_eq!(resolve_missing("act", 1), Some("classic"));
        assert_eq!(resolve_missing("act", 3), Some("classic"));
        assert_eq!(resolve_missing("act", 4), None);
        // attn_decode introduced at v5: v1..v4 dumps missing it resolve to
        // "f32-bypass" (the Reference oracle's LAGUNA_ATTN_F32 decode path — the
        // value every cached pre-v5 reference dump carries); missing at v5 fails.
        assert_eq!(resolve_missing("attn_decode", 1), Some("f32-bypass"));
        assert_eq!(resolve_missing("attn_decode", 4), Some("f32-bypass"));
        assert_eq!(resolve_missing("attn_decode", 5), None);
        // Baseline fields are required at every version.
        assert_eq!(resolve_missing("attn_dtype", 1), None);
        assert_eq!(resolve_missing("attn_dtype", 2), None);
        // Unknown fields fail closed.
        assert_eq!(resolve_missing("not_a_field", 1), None);
    }
}
