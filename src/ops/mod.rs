mod dispatch;
pub mod mm_id;
pub mod mv_id;

pub use mm_id::mul_mm_id;
pub use mv_id::mul_mv_id;

pub use crate::gguf::ExpertStack;

/// Which expert-FFN implementation a model is built with.
/// Fused dispatches candle's kernel_mul_mv_id_*/mm_id_* Metal kernels over the
/// stacked quantized tensors (ids stay on GPU); Reference slices the stack into
/// per-expert QTensors with a CPU id readback — slow, but the correctness oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertRunner {
    Fused,
    Reference,
}
