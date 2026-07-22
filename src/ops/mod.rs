mod dispatch;
pub mod mm_id;
pub mod mv_id;
mod pipelines;

pub use dispatch::mv_vendored_supported;
pub use mm_id::mul_mm_id;
pub use mv_id::{mul_mv, mul_mv_id, mv_classic};

pub use crate::gguf::ExpertStack;

use std::sync::OnceLock;

/// Prefill token-count threshold at/above which the fused MoE switches from
/// per-token mv_id to the mm_id two-pass matmul (ggml's mm_id break-even point).
/// Single source of truth: `moe` gates the prefill branch on it, and
/// `logits-dump` records it in each dump's provenance so the parity gate can
/// tell whether a dump actually exercised the mm_id path (do NOT re-hardcode 32).
pub const MM_ID_MIN_SEQ: usize = 32;

/// `LAGUNA_NO_MM_ID=1` forces the per-token mv_id path everywhere (prefill
/// included), as a fallback / parity-debug switch. Read once and cached — it is
/// consulted per MoE layer on the hot path.
///
/// PRESENCE-BASED, like the `LAGUNA_MM_ID_*` variant toggles below: any value
/// (even `LAGUNA_NO_MM_ID=0`) enables it — only unset disables it.
pub(crate) fn no_mm_id() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("LAGUNA_NO_MM_ID").is_some())
}

/// Public view of the cached `LAGUNA_NO_MM_ID` switch, for dump provenance.
pub fn no_mm_id_forced() -> bool {
    no_mm_id()
}

/// The active mm_id kernel variant's provenance name (e.g. `"tensor"`), for
/// dump provenance. Reflects the same cached `mm_id_variant()` the hot path uses.
pub fn active_mm_variant_name() -> &'static str {
    mm_id_variant().name()
}

/// The mm_id prefill kernel family. Runtime-selectable via env; the single
/// source of truth (both the kernel selection in dispatch and the rescale
/// decision in moe read the cached value here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MmVariant {
    /// The fork's cooperative-tensor path (`_t`): Metal-4 `matmul2d`, f16 operand
    /// tiles. Default. Casts the activation to f16, so it needs the L2 rescale
    /// guard, and its parity sits in the fork-equivalent mm tier.
    Tensor,
    /// Cooperative-tensor path with f32 operand tiles (`_t_hp`): `matmul2d` on
    /// float cooperative tensors. No f16 cast (no rescale); still tiled, so mm
    /// tier. Only instantiated for q4_K/q6_K (the production experts).
    TensorHp,
    /// Classic simdgroup tiles in f32 (`_hp`): no f16 cast, no rescale.
    ClassicHp,
    /// Classic simdgroup tiles in f16 (base name): f16 operand cast, needs rescale.
    ClassicF16,
}

impl MmVariant {
    /// Kernel host-name suffix appended to `kernel_mul_mm_id_<dtype>_f32`.
    pub(crate) fn suffix(self) -> &'static str {
        match self {
            MmVariant::Tensor => "_t",
            MmVariant::TensorHp => "_t_hp",
            MmVariant::ClassicHp => "_hp",
            MmVariant::ClassicF16 => "",
        }
    }

    /// Threadgroup tile bytes: the f32-tile variants (`_hp`, `_t_hp`) need 12288,
    /// the half-tile variants 8192 (sa+sb+float store-back tile).
    pub(crate) fn tile_smem(self) -> usize {
        match self {
            MmVariant::ClassicHp | MmVariant::TensorHp => 12288,
            MmVariant::Tensor | MmVariant::ClassicF16 => 8192,
        }
    }

    /// Whether this variant casts the down-projection activation to f16 (so the
    /// L2 rescale guard is required). The f32-tile variants (`_hp`, `_t_hp`) do not.
    pub(crate) fn casts_activation_f16(self) -> bool {
        matches!(self, MmVariant::Tensor | MmVariant::ClassicF16)
    }

    /// Stable provenance name (distinct from `suffix`, which is a kernel-name
    /// fragment and empty for `ClassicF16`).
    pub(crate) fn name(self) -> &'static str {
        match self {
            MmVariant::Tensor => "tensor",
            MmVariant::TensorHp => "tensor_hp",
            MmVariant::ClassicHp => "classic_hp",
            MmVariant::ClassicF16 => "classic_f16",
        }
    }
}

/// Which mm_id variant to run, cached (read once). Precedence: `LAGUNA_MM_ID_F16`
/// → classic f16 tiles; `LAGUNA_MM_ID_CLASSIC` → classic f32 (`_hp`) tiles;
/// `LAGUNA_MM_ID_TENSOR_HP` → f32 tensor tiles (`_t_hp`); else the f16 tensor
/// path (default). The tensor kernels compile on this device (the mm_id.metal
/// probe test gates that); the other variants remain for A/B.
///
/// PRESENCE-BASED toggles: each is enabled by the env var merely being SET, whatever
/// its value — `LAGUNA_MM_ID_F16=0` still selects the f16 classic tiles. To disable a
/// variant, UNSET its var (do not set it to `0`/`false`). First set var in the
/// precedence order above wins.
pub(crate) fn mm_id_variant() -> MmVariant {
    static V: OnceLock<MmVariant> = OnceLock::new();
    *V.get_or_init(|| {
        if std::env::var_os("LAGUNA_MM_ID_F16").is_some() {
            MmVariant::ClassicF16
        } else if std::env::var_os("LAGUNA_MM_ID_CLASSIC").is_some() {
            MmVariant::ClassicHp
        } else if std::env::var_os("LAGUNA_MM_ID_TENSOR_HP").is_some() {
            MmVariant::TensorHp
        } else {
            MmVariant::Tensor
        }
    })
}

/// Which expert-FFN implementation a model is built with.
/// Fused dispatches candle's kernel_mul_mv_id_*/mm_id_* Metal kernels over the
/// stacked quantized tensors (ids stay on GPU); Reference slices the stack into
/// per-expert QTensors with a CPU id readback — slow, but the correctness oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertRunner {
    Fused,
    Reference,
}
