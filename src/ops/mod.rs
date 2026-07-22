mod dispatch;
pub mod mm_id;
pub mod mv_id;
mod pipelines;

pub use mm_id::mul_mm_id;
pub use mv_id::mul_mv_id;

pub use crate::gguf::ExpertStack;

use std::sync::OnceLock;

/// `LAGUNA_NO_MM_ID=1` forces the per-token mv_id path everywhere (prefill
/// included), as a fallback / parity-debug switch. Read once and cached — it is
/// consulted per MoE layer on the hot path.
pub(crate) fn no_mm_id() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("LAGUNA_NO_MM_ID").is_some())
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
}

/// Which mm_id variant to run, cached (read once). Precedence: `LAGUNA_MM_ID_F16`
/// → classic f16 tiles; `LAGUNA_MM_ID_CLASSIC` → classic f32 (`_hp`) tiles;
/// `LAGUNA_MM_ID_TENSOR_HP` → f32 tensor tiles (`_t_hp`); else the f16 tensor
/// path (default). The tensor kernels compile on this device (the mm_id.metal
/// probe test gates that); the other variants remain for A/B.
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
