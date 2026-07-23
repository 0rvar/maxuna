use std::sync::Arc;

use anyhow::Result;
use candle_core::quantized::QTensor;
use candle_core::{DType, Tensor};
use candle_nn::ops::{sigmoid, silu};

use crate::config::LagunaConfig;
use crate::gguf::{ExpertStack, QLinear, Weights};
use crate::ops::{ExpertRunner, mm_id, mv_id};

/// F16's smallest positive normal, the denominator floor llama.cpp clamps the
/// routing-weight sum to before normalizing (`ffn_moe_weights_sum_clamped`).
const WEIGHTS_SUM_FLOOR: f64 = 6.103_515_625e-5;

/// Runs the selected experts' SwiGLU FFNs.
/// x: [seq, hidden] f32; ids: [seq, top_k] u32 on-device; weights: [seq, top_k] f32.
/// Returns [seq, hidden] f32 (already weight-combined).
pub trait ExpertFfn: Send {
    fn forward(&self, x: &Tensor, ids: &Tensor, weights: &Tensor) -> Result<Tensor>;
}

/// One MoE FFN block (layers >= dense_layers). Routing math (fork build_moe_ffn):
/// probs = sigmoid(router(x)); selection = probs + exp_probs_b (bias affects
/// selection ONLY); top-k by selection; weights = gather(probs, ids) unbiased;
/// weights /= max(sum, 6.1e-5); weights *= expert_weights_scale; plus an
/// unscaled shared-expert SwiGLU added in parallel.
pub struct MoeBlock {
    /// Router weight, pre-transposed to `[hidden, n_expert]` and made contiguous
    /// once at load so the per-forward routing matmul needs no transpose/copy.
    router_t: Tensor,
    exp_probs_b: Tensor,
    experts: Box<dyn ExpertFfn>,
    shared: SharedExpert,
    n_expert_used: usize,
    weights_scale: f64,
    weights_norm: bool,
}

impl MoeBlock {
    /// `w` positioned at the block prefix. `runner` picks Fused vs Reference experts.
    pub fn new(w: &Weights, cfg: &LagunaConfig, runner: ExpertRunner) -> Result<Self> {
        let router_t = w.dense_f32("ffn_gate_inp")?.t()?.contiguous()?;
        let exp_probs_b = w.dense_f32_biasable("exp_probs_b")?;
        let experts: Box<dyn ExpertFfn> = match runner {
            ExpertRunner::Reference => Box::new(ReferenceExperts::new(w)?),
            ExpertRunner::Fused => Box::new(FusedExperts::new(w)?),
        };
        let shared = SharedExpert::new(w)?;
        Ok(Self {
            router_t,
            exp_probs_b,
            experts,
            shared,
            n_expert_used: cfg.n_expert_used,
            weights_scale: cfg.expert_weights_scale,
            weights_norm: cfg.expert_weights_norm,
        })
    }

    /// Returns (ids [seq, top_k] u32 on-device, weights [seq, top_k] f32).
    pub fn route(&self, x_normed: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = route_logits(&self.router_t, x_normed)?;
        route_from_logits(
            &logits,
            &self.exp_probs_b,
            self.n_expert_used,
            self.weights_scale,
            self.weights_norm,
        )
    }

    pub fn forward(&self, x_normed: &Tensor) -> Result<Tensor> {
        let (ids, weights) = self.route(x_normed)?;
        let routed = self.experts.forward(x_normed, &ids, &weights)?;
        let shared = self.shared.forward(x_normed)?;
        Ok((routed + shared)?)
    }
}

/// Router projection: `logits[t, e] = <x[t], router[e]>`, computed in f32.
/// `router_t` is the `[hidden, n_expert]` transpose of the ggml `ffn_gate_inp`
/// weight, precomputed once at load; result is `[seq, n_expert]`.
fn route_logits(router_t: &Tensor, x: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    Ok(x.matmul(router_t)?)
}

/// The routing decision from pre-sigmoid logits, kept separate from the router
/// matmul so it can be exercised with hand-built logits. `bias` (`exp_probs_b`)
/// perturbs the top-k *selection* only; the returned weights are gathered from
/// the unbiased sigmoid probabilities, sum-normalized (with the F16 floor) when
/// `norm` is set, then multiplied by `scale`.
fn route_from_logits(
    logits: &Tensor,
    bias: &Tensor,
    n_expert_used: usize,
    scale: f64,
    norm: bool,
) -> Result<(Tensor, Tensor)> {
    let probs = sigmoid(&logits.to_dtype(DType::F32)?)?; // [seq, n_expert], unbiased
    let selection = probs.broadcast_add(bias)?; // bias affects selection only

    // Descending arg-sort is stable on CPU (std slice sort), so ties resolve to
    // the lower expert index first — matching ggml_argsort_top_k.
    let order = selection.contiguous()?.arg_sort_last_dim(false)?;
    let ids = order.narrow(1, 0, n_expert_used)?.contiguous()?; // [seq, top_k] u32

    let mut weights = probs.gather(&ids, 1)?; // gather from UNBIASED probs
    if norm {
        let sum = weights.sum_keepdim(1)?;
        let sum = sum.clamp(WEIGHTS_SUM_FLOOR as f32, f32::INFINITY)?;
        weights = weights.broadcast_div(&sum)?;
    }
    if scale != 0.0 && scale != 1.0 {
        weights = (weights * scale)?;
    }
    Ok((ids, weights))
}

/// Correctness oracle: keeps each routed expert's gate/up/down as an individual
/// quantized weight and evaluates only the experts a token actually selected.
/// Reads the selection ids back to the CPU to bucket tokens per expert — slow,
/// but exercises the same dequantized weights the fused kernels consume.
struct ReferenceExperts {
    gate: Vec<Arc<QTensor>>, // each [expert_ff, hidden]
    up: Vec<Arc<QTensor>>,   // each [expert_ff, hidden]
    down: Vec<Arc<QTensor>>, // each [hidden, expert_ff]
}

impl ReferenceExperts {
    fn new(w: &Weights) -> Result<Self> {
        Ok(Self {
            gate: w.expert_qtensors("ffn_gate_exps")?,
            up: w.expert_qtensors("ffn_up_exps")?,
            down: w.expert_qtensors("ffn_down_exps")?,
        })
    }
}

impl ExpertFfn for ReferenceExperts {
    fn forward(&self, x: &Tensor, ids: &Tensor, weights: &Tensor) -> Result<Tensor> {
        let (seq, hidden) = x.dims2()?;
        let top_k = ids.dim(1)?;
        let device = x.device();
        let x = x.to_dtype(DType::F32)?;

        let ids_v: Vec<u32> = ids.flatten_all()?.to_vec1()?;
        let w_v: Vec<f32> = weights.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;

        // Bucket (token, weight) pairs per expert so each selected expert runs
        // one batched matmul over its assigned tokens.
        let n_expert = self.gate.len();
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); n_expert];
        let mut wts: Vec<Vec<f32>> = vec![Vec::new(); n_expert];
        for t in 0..seq {
            for k in 0..top_k {
                let e = ids_v[t * top_k + k] as usize;
                rows[e].push(t as u32);
                wts[e].push(w_v[t * top_k + k]);
            }
        }

        let mut out = Tensor::zeros((seq, hidden), DType::F32, device)?;
        for e in 0..n_expert {
            if rows[e].is_empty() {
                continue;
            }
            let m = rows[e].len();
            let idx = Tensor::from_vec(rows[e].clone(), m, device)?;
            let xe = x.index_select(&idx, 0)?; // [m, hidden]

            let gate_w = self.gate[e].dequantize(device)?.to_dtype(DType::F32)?;
            let up_w = self.up[e].dequantize(device)?.to_dtype(DType::F32)?;
            let down_w = self.down[e].dequantize(device)?.to_dtype(DType::F32)?;

            let g = silu(&xe.matmul(&gate_w.t()?)?)?; // [m, expert_ff]
            let u = xe.matmul(&up_w.t()?)?; // [m, expert_ff]
            let h = (&g * &u)?;
            let d = h.matmul(&down_w.t()?)?; // [m, hidden]

            let we = Tensor::from_vec(wts[e].clone(), (m, 1), device)?;
            let d = d.broadcast_mul(&we)?;
            out = out.index_add(&idx, &d, 0)?;
        }
        Ok(out)
    }
}

/// Fused path: the routed experts stay stacked in their quantized layout and the
/// `ops::mv_id` gather-matvec kernel indexes them by id on-device (no readback).
/// Mirrors the fork's build_moe_ffn expert evaluation, including the per-column
/// L2 rescale that keeps the down-projection input inside f16 range.
struct FusedExperts {
    gate: ExpertStack,
    up: ExpertStack,
    down: ExpertStack,
}

impl FusedExperts {
    fn new(w: &Weights) -> Result<Self> {
        let gate = w.expert_stack("ffn_gate_exps")?;
        let up = w.expert_stack("ffn_up_exps")?;
        let down = w.expert_stack("ffn_down_exps")?;
        // The shared map0 pass is built once from `gate.n_expert` and reused for
        // all three projections; a stack disagreeing on n_expert would read the
        // ids-map at the wrong offset. (run_mm_shared re-validates per dispatch,
        // but a mismatch here is a malformed checkpoint — fail at load.)
        anyhow::ensure!(
            gate.n_expert == up.n_expert && gate.n_expert == down.n_expert,
            "MoE expert stacks disagree on n_expert: gate={}, up={}, down={}",
            gate.n_expert,
            up.n_expert,
            down.n_expert
        );
        Ok(Self { gate, up, down })
    }
}

impl ExpertFfn for FusedExperts {
    fn forward(&self, x: &Tensor, ids: &Tensor, weights: &Tensor) -> Result<Tensor> {
        let (seq, hidden) = x.dims2()?;
        let top_k = ids.dim(1)?;
        let x = x.to_dtype(DType::F32)?;

        // Prefill (many tokens) uses the two-pass token-grouped matmul so each
        // expert's rows are dequantized once for all the tokens routed to it;
        // decode (batch=1, and short chunks) stays on the per-token matvec, which
        // wins at low token counts. MM_ID_MIN_SEQ is ggml's mm_id break-even point.
        // `LAGUNA_NO_MM_ID` forces mv_id everywhere as a fallback; mm_id is also
        // skipped (mv_id fallback) for any dtype/top_k/variant the vendored kernels
        // are not instantiated for, so other checkpoints still run. Note mm_id's
        // tiled f32 accumulation drifts a little further from the per-row f32
        // reference oracle than mv_id does (fork-equivalent tiled behavior; see
        // docs/parity.md §3b), so mv_id is the reference for the strict gate.
        let variant = crate::ops::mm_id_variant();
        let mm_supported = mm_id::supported(self.gate.dtype, top_k, variant)
            && mm_id::supported(self.up.dtype, top_k, variant)
            && mm_id::supported(self.down.dtype, top_k, variant);
        let use_mm = seq >= crate::ops::mm_id_min_seq() && !crate::ops::no_mm_id() && mm_supported;

        // One map0 pass (per-expert token-slot map) shared by gate/up/down: the
        // three projections route by the SAME ids, so their maps are byte-identical.
        // Built once here and held for the whole forward — the down projection reads
        // it after silu/rescale — replacing two of the three per-projection passes.
        let map0 = if use_mm {
            Some(mm_id::prepare_map0(self.gate.n_expert, ids)?)
        } else {
            None
        };
        let matmul = |stack: &ExpertStack, x: &Tensor, ids: &Tensor| -> Result<Tensor> {
            match &map0 {
                Some(scratch) => mm_id::mul_mm_id_shared(stack, x, ids, scratch),
                None => mv_id::mul_mv_id(stack, x, ids),
            }
        };

        // gate/up share one activation per token: x_per_row = 1.
        let x_g = x.reshape((seq, 1, hidden))?;
        let gate = matmul(&self.gate, &x_g, ids)?; // [seq, top_k, expert_ff]
        let up = matmul(&self.up, &x_g, ids)?; // [seq, top_k, expert_ff]

        // The down projection consumes the SwiGLU activation. Some mm_id variants
        // stage that activation as f16, where a large value would overflow, and
        // need the L2 rescale guard; others read it as f32 and do not:
        //   - decode / mv_id: candle's kernel_mul_mv_q{4,6}_K_f32_impl reads src1
        //     as f32 and accumulates in f32 (quantized.metal:4889/4930, 5188/5225);
        //   - prefill / mm_id-hp (classic f32 tiles): stages src1 as float — no cast;
        //   - prefill / mm_id tensor default + classic-f16: stage src1 as half —
        //     f16 cast, so the rescale is required.
        let needs_rescale = use_mm && crate::ops::mm_id_variant().casts_activation_f16();

        // silu(gate)*up via candle ops. Vendored fused kernels (a full
        // silu/mul/L2/rescale one, and a plain elementwise silu*mul) were tried
        // and abandoned: even ~1e-6 differences in how the activation is computed
        // cascade through the MoE router (near-tie expert flips downstream) to
        // ~1e-3 final-logit divergence, below the strict gate. Any reimplementation
        // of the activation cascades, so it stays on candle's ops, which the f32
        // reference oracle also uses. See docs/parity.md §3b and TODO.md.
        let act = (silu(&gate)? * up)?; // [seq, top_k, expert_ff]

        // Routing weights, f32 [seq, top_k]; the fused combine takes this shape,
        // the classic candle chain reshapes to [seq, top_k, 1] for broadcasting.
        let w = weights.to_dtype(DType::F32)?;

        if needs_rescale {
            // Per-column L2 rescale keeps the f16-tile down cast in range; the
            // factor divides back out afterwards (a per-column identity). 32768
            // (f16's safe headroom) is only meaningful on this f16-tile branch.
            let f16_safe = 32768.0_f64;
            let col_l2 = act
                .sqr()?
                .sum_keepdim(2)?
                .sqrt()?
                .clamp(1e-8_f32, 1e30_f32)?; // [seq, top_k, 1]
            let act_s = (&act * f16_safe)?.broadcast_div(&col_l2)?;
            let down = matmul(&self.down, &act_s, ids)?; // [seq, top_k, hidden]
            // Undo the L2 scale, apply routing weights, sum over top_k. The fused
            // kernel does all three in one pass over `down`, bit-identically to the
            // candle chain; LAGUNA_COMBINE_CLASSIC keeps the candle chain.
            if crate::ops::combine_classic() {
                let down = (down.broadcast_mul(&col_l2)? * (1.0 / f16_safe))?;
                let w = w.reshape((seq, top_k, 1))?;
                return Ok(down.broadcast_mul(&w)?.sum(1)?);
            }
            return crate::ops::combine(&down, Some(&col_l2), &w);
        }

        // Default: f32 down projection (mv_id or mm_id-hp) — no f16 cast, so the
        // activation feeds the down matmul directly, no rescale needed.
        let down = matmul(&self.down, &act, ids)?; // [seq, top_k, hidden]
        if crate::ops::combine_classic() {
            let w = w.reshape((seq, top_k, 1))?;
            return Ok(down.broadcast_mul(&w)?.sum(1)?);
        }
        crate::ops::combine(&down, None, &w)
    }
}

/// The always-on shared expert: a plain SwiGLU with intermediate `shared_expert_ff`.
pub struct SharedExpert {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

impl SharedExpert {
    pub fn new(w: &Weights) -> Result<Self> {
        Ok(Self {
            gate: w.qlinear("ffn_gate_shexp")?,
            up: w.qlinear("ffn_up_shexp")?,
            down: w.qlinear("ffn_down_shexp")?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        swiglu(&self.gate, &self.up, &self.down, x)
    }
}

/// Layer 0's dense SwiGLU MLP (intermediate dense_ff = 12288).
pub struct DenseMlp {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

impl DenseMlp {
    pub fn new(w: &Weights) -> Result<Self> {
        Ok(Self {
            gate: w.qlinear("ffn_gate")?,
            up: w.qlinear("ffn_up")?,
            down: w.qlinear("ffn_down")?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        swiglu(&self.gate, &self.up, &self.down, x)
    }
}

/// SwiGLU FFN: `down(silu(gate(x)) * up(x))`.
fn swiglu(gate: &QLinear, up: &QLinear, down: &QLinear, x: &Tensor) -> Result<Tensor> {
    let g = silu(&gate.forward(x)?)?;
    let u = up.forward(x)?;
    let h = (&g * &u)?;
    Ok(down.forward(&h)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, Tensor};

    // Sigmoid of the fixed logit vector [2, -2, 0, 1, -1, 3, 0.5, -0.5], to full
    // f64 precision; the routing fixtures reason about these probabilities.
    const S2: f64 = 0.880_797_077_977_882_3; // sigmoid(2.0)
    const S1: f64 = 0.731_058_578_630_004_9; // sigmoid(1.0)
    const S3: f64 = 0.952_574_126_822_433_4; // sigmoid(3.0)
    const S05: f64 = 0.622_459_331_201_854_6; // sigmoid(0.5)

    fn logits_row() -> Tensor {
        Tensor::from_vec(
            vec![2.0_f32, -2.0, 0.0, 1.0, -1.0, 3.0, 0.5, -0.5],
            (1, 8),
            &Device::Cpu,
        )
        .unwrap()
    }

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected {b}, got {a} (tol {tol})");
    }

    /// A positive bias on a low-probability expert must pull it into the top-k,
    /// displacing a higher-probability one — yet the selected weights still come
    /// from the unbiased probabilities.
    #[test]
    fn routing_positive_bias_displaces() {
        // Unbiased top-3 by prob is {5, 0, 3}. Bias +0.2 on expert 6 lifts its
        // selection score (0.6225 -> 0.8225) above expert 3 (0.7311).
        let bias = Tensor::from_vec(
            vec![0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
            8,
            &Device::Cpu,
        )
        .unwrap();
        let (ids, weights) =
            route_from_logits(&logits_row(), &bias, 3, 2.5, true).unwrap();

        let ids_v: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(ids_v, vec![5, 0, 6]);

        // Weights are the unbiased probs of {5, 0, 6}, sum-normalized then *2.5.
        // Expert 6 contributes its unbiased 0.6225, NOT the biased 0.8225.
        let sel = [S3, S2, S05];
        let sum: f64 = sel.iter().sum();
        let expected: Vec<f64> = sel.iter().map(|p| p / sum * 2.5).collect();
        let got: Vec<f32> = weights.flatten_all().unwrap().to_vec1().unwrap();
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_close(*g as f64, *e, 1e-5);
        }
    }

    /// A strong negative bias drops an otherwise top-ranked expert out of the
    /// selection.
    #[test]
    fn routing_negative_bias_excludes() {
        // Expert 5 has the highest prob (0.9526); a -1.0 bias sinks its selection
        // score to -0.047, so the top-3 becomes {0, 3, 6}.
        let bias = Tensor::from_vec(
            vec![0.0_f32, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            8,
            &Device::Cpu,
        )
        .unwrap();
        let (ids, weights) =
            route_from_logits(&logits_row(), &bias, 3, 2.5, true).unwrap();

        let ids_v: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(ids_v, vec![0, 3, 6]);

        let sel = [S2, S1, S05];
        let sum: f64 = sel.iter().sum();
        let expected: Vec<f64> = sel.iter().map(|p| p / sum * 2.5).collect();
        let got: Vec<f32> = weights.flatten_all().unwrap().to_vec1().unwrap();
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_close(*g as f64, *e, 1e-5);
        }
    }

    /// Equal selection scores resolve to the lowest expert indices first.
    #[test]
    fn routing_ties_pick_lowest_indices() {
        let logits = Tensor::zeros((1, 8), DType::F32, &Device::Cpu).unwrap();
        let bias = Tensor::zeros(8, DType::F32, &Device::Cpu).unwrap();
        let (ids, _) = route_from_logits(&logits, &bias, 3, 2.5, true).unwrap();
        let ids_v: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(ids_v, vec![0, 1, 2]);
    }

    /// When every probability is ~0 the weight sum underflows and must clamp to
    /// the F16 floor, so weights become `prob / floor * scale`, not `prob / sum`.
    #[test]
    fn routing_clamps_tiny_denominator() {
        let logits = Tensor::from_vec(vec![-30.0_f32; 8], (1, 8), &Device::Cpu).unwrap();
        let bias = Tensor::zeros(8, DType::F32, &Device::Cpu).unwrap();
        let (_, weights) = route_from_logits(&logits, &bias, 3, 2.5, true).unwrap();

        let p = 1.0 / (1.0 + 30.0_f64.exp()); // sigmoid(-30)
        // Sum of three such probs (~2.8e-13) is far below the floor, so it clamps.
        let expected = p / WEIGHTS_SUM_FLOOR * 2.5;
        let got: Vec<f32> = weights.flatten_all().unwrap().to_vec1().unwrap();
        for g in got {
            assert_close(g as f64, expected, expected * 1e-3);
        }
    }

    /// The router matmul followed by the routing decision agrees with feeding
    /// the same logits in directly.
    #[test]
    fn router_matmul_matches_direct_logits() {
        // hidden == n_expert with an identity router, so logits == x.
        let router = Tensor::eye(8, DType::F32, &Device::Cpu).unwrap();
        let x = logits_row();

        let logits = route_logits(&router, &x).unwrap();
        let direct: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        let viamm: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in viamm.iter().zip(direct.iter()) {
            assert_close(*a as f64, *b as f64, 1e-6);
        }
    }

    /// Deterministic pseudo-random f32 tensor in roughly `[-scale, scale]`, so
    /// the quantization-vs-dequantization comparison is reproducible run to run.
    fn det_tensor(dims: &[usize], seed: u64, scale: f32) -> Tensor {
        let n: usize = dims.iter().product();
        let mut s = seed;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 40) as f32) / ((1u64 << 24) as f32); // [0, 1)
            v.push((u - 0.5) * 2.0 * scale);
        }
        Tensor::from_vec(v, dims, &Device::Cpu).unwrap()
    }

    /// A quantized f32 stack, sliced per-expert, must reproduce a
    /// dequantize-everything-then-matmul oracle. Both sides dequantize the same
    /// bytes, so the only gap is f32 accumulation order: the implementation
    /// batches each expert's assigned tokens into one gemm while the oracle runs
    /// a per-token gemv, and cancellation in the 256-wide dot products lifts that
    /// gap to a few 1e-4 — well below any wiring/weighting bug.
    fn check_reference_experts(dtype: GgmlDType) {
        let dev = Device::Cpu;
        let n_expert = 4usize;
        let n_ff = 256usize;
        let hidden = 256usize; // multiple of the 256-element K-quant block
        let seq = 3usize;
        let top_k = 2usize;

        let gate_stack = Arc::new(
            QTensor::quantize(&det_tensor(&[n_expert, n_ff, hidden], 1, 0.5), dtype).unwrap(),
        );
        let up_stack = Arc::new(
            QTensor::quantize(&det_tensor(&[n_expert, n_ff, hidden], 2, 0.5), dtype).unwrap(),
        );
        let down_stack = Arc::new(
            QTensor::quantize(&det_tensor(&[n_expert, hidden, n_ff], 3, 0.5), dtype).unwrap(),
        );

        let re = ReferenceExperts {
            gate: crate::gguf::split_expert_stack(&gate_stack, n_expert, n_ff, hidden, &dev)
                .unwrap(),
            up: crate::gguf::split_expert_stack(&up_stack, n_expert, n_ff, hidden, &dev).unwrap(),
            down: crate::gguf::split_expert_stack(&down_stack, n_expert, hidden, n_ff, &dev)
                .unwrap(),
        };

        let x = det_tensor(&[seq, hidden], 4, 1.0);
        let ids_v: Vec<u32> = vec![0, 3, 1, 2, 3, 0]; // [seq, top_k]
        let w_v: Vec<f32> = vec![0.5, 0.25, 1.5, 0.75, 0.1, 0.9];
        let ids = Tensor::from_vec(ids_v.clone(), (seq, top_k), &dev).unwrap();
        let weights = Tensor::from_vec(w_v.clone(), (seq, top_k), &dev).unwrap();

        let got = re.forward(&x, &ids, &weights).unwrap();

        // Oracle: dequantize the whole stack, evaluate each token's selected
        // experts with plain f32 matmuls.
        let gate_d = gate_stack.dequantize(&dev).unwrap();
        let up_d = up_stack.dequantize(&dev).unwrap();
        let down_d = down_stack.dequantize(&dev).unwrap();
        let mut expected_rows: Vec<Tensor> = Vec::with_capacity(seq);
        for t in 0..seq {
            let xt = x.narrow(0, t, 1).unwrap(); // [1, hidden]
            let mut acc = Tensor::zeros((1, hidden), DType::F32, &dev).unwrap();
            for k in 0..top_k {
                let e = ids_v[t * top_k + k] as usize;
                let gw = gate_d.narrow(0, e, 1).unwrap().squeeze(0).unwrap();
                let uw = up_d.narrow(0, e, 1).unwrap().squeeze(0).unwrap();
                let dw = down_d.narrow(0, e, 1).unwrap().squeeze(0).unwrap();
                let g = silu(&xt.matmul(&gw.t().unwrap()).unwrap()).unwrap();
                let u = xt.matmul(&uw.t().unwrap()).unwrap();
                let h = (&g * &u).unwrap();
                let d = h.matmul(&dw.t().unwrap()).unwrap();
                let contrib = (d * w_v[t * top_k + k] as f64).unwrap();
                acc = (acc + contrib).unwrap();
            }
            expected_rows.push(acc);
        }
        let expected = Tensor::cat(&expected_rows, 0).unwrap();

        let got_v: Vec<Vec<f32>> = got.to_vec2().unwrap();
        let exp_v: Vec<Vec<f32>> = expected.to_vec2().unwrap();
        for (gr, er) in got_v.iter().zip(exp_v.iter()) {
            for (g, e) in gr.iter().zip(er.iter()) {
                let tol = 1e-3 * (e.abs() as f64).max(1.0);
                assert_close(*g as f64, *e as f64, tol);
            }
        }
    }

    #[test]
    fn reference_experts_q8_0() {
        check_reference_experts(GgmlDType::Q8_0);
    }

    #[test]
    fn reference_experts_q4k() {
        check_reference_experts(GgmlDType::Q4K);
    }

    #[test]
    fn reference_experts_q5k() {
        check_reference_experts(GgmlDType::Q5K);
    }

    #[test]
    fn reference_experts_q6k() {
        check_reference_experts(GgmlDType::Q6K);
    }

    /// SwiGLU FFN matches a hand-rolled `down(silu(gate(x)) * up(x))` with dense
    /// f32 weights. F32 QTensors round-trip losslessly, so agreement is tight.
    fn check_swiglu(intermediate: usize) {
        let hidden = 16usize;
        let seq = 4usize;

        let gate_w = det_tensor(&[intermediate, hidden], 11, 0.5);
        let up_w = det_tensor(&[intermediate, hidden], 12, 0.5);
        let down_w = det_tensor(&[hidden, intermediate], 13, 0.5);

        let ql = |t: &Tensor| {
            QLinear::from_qtensor(Arc::new(QTensor::quantize(t, GgmlDType::F32).unwrap())).unwrap()
        };
        let ffn = SharedExpert { gate: ql(&gate_w), up: ql(&up_w), down: ql(&down_w) };

        let x = det_tensor(&[seq, hidden], 14, 1.0);
        let got = ffn.forward(&x).unwrap();

        let g = silu(&x.matmul(&gate_w.t().unwrap()).unwrap()).unwrap();
        let u = x.matmul(&up_w.t().unwrap()).unwrap();
        let h = (&g * &u).unwrap();
        let expected = h.matmul(&down_w.t().unwrap()).unwrap();

        let got_v: Vec<Vec<f32>> = got.to_vec2().unwrap();
        let exp_v: Vec<Vec<f32>> = expected.to_vec2().unwrap();
        for (gr, er) in got_v.iter().zip(exp_v.iter()) {
            for (a, b) in gr.iter().zip(er.iter()) {
                assert_close(*a as f64, *b as f64, 1e-4 * (b.abs() as f64).max(1.0));
            }
        }
    }

    #[test]
    fn shared_expert_swiglu() {
        check_swiglu(24);
    }

    #[test]
    fn dense_mlp_swiglu() {
        // Same SwiGLU wiring DenseMlp uses, with a larger dense intermediate.
        check_swiglu(48);
    }

    /// Decode MoE-FFN and per-token-overhead perf benches (phase 0 of the
    /// decode budget work; the attention half lives in attention.rs
    /// tests::decode_bench, whose conventions these copy). Synthetic weights at
    /// production geometry — never loads a model file. All are `#[ignore]`d;
    /// run one at a time with e.g.
    /// `cargo test --release moe_decode_ffn_bench -- --ignored --nocapture`.
    /// Iteration counts: LAGUNA_BENCH_WARMUP (default 10) / LAGUNA_BENCH_ITERS
    /// (default 50). Each iter ends in one small CPU readback so it measures
    /// end-to-end latency including the command-buffer flush.
    mod decode_bench {
        use super::*;
        use candle_core::Module;
        use candle_core::quantized::{QMatMul, QStorage};
        use candle_nn::RmsNorm;
        use std::borrow::Cow;
        use std::time::Instant;

        use crate::sampler::{Sampler, SamplerOptions};

        const HIDDEN: usize = 3072;
        /// GGUF `laguna.expert_feed_forward_length` (and
        /// `expert_shared_feed_forward_length` — both 1024 in the Q4_K_M file).
        const EXPERT_FF: usize = 1024;
        const SHARED_FF: usize = 1024;
        const N_EXPERT: usize = 256;
        const TOP_K: usize = 10;
        const VOCAB: usize = 100352;
        /// Layers 1..47 of the real model are MoE; results are scaled to this.
        const MOE_LAYERS: usize = 47;
        /// Memory constraint: 47 distinct synthetic layers would be ~50GB of
        /// expert stacks, so build 4 distinct layers (~5.5GB) and loop them 12
        /// times per iter — 48 layer-evals, scaled by 47/48 when printed.
        const N_DISTINCT: usize = 4;
        const PASSES: usize = 12;

        fn metal() -> Device {
            Device::new_metal(0).expect("decode benches require the Metal device")
        }

        fn iter_counts() -> (usize, usize) {
            let get = |k: &str, d: usize| {
                std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
            };
            (get("LAGUNA_BENCH_WARMUP", 10), get("LAGUNA_BENCH_ITERS", 50))
        }

        /// Small readback forcing command-buffer completion (the per-iter sync).
        fn read_scalar(t: &Tensor) -> f32 {
            let t = if t.dtype() == DType::F32 { t.clone() } else { t.to_dtype(DType::F32).unwrap() };
            t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]
        }

        /// Warm-up + timed loop; returns the mean ms/iter.
        fn bench(name: &str, mut f: impl FnMut() -> f32) -> f64 {
            let (warm, iters) = iter_counts();
            let mut sink = 0f32;
            for _ in 0..warm {
                sink += f();
            }
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                sink += f();
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
            eprintln!(
                "{name}: mean {mean:.3} ms/iter, min {min:.3} ms/iter ({iters} iters, sink {sink:.1})"
            );
            mean
        }

        /// A quantized `[n_expert, n_out, k]` ExpertStack whose bytes are 256
        /// tiled copies of ONE quantized expert. Weight VALUES are
        /// timing-irrelevant (the mv_id gather reads each selected expert's
        /// slice by address, so per-pass DRAM traffic is unchanged), and tiling
        /// cuts the synthetic quantization work 256x. The buffer/QTensor
        /// construction mirrors `ops::dispatch::testutil::build_stack` (and
        /// production `gguf::expert_stack`): one `QStorage::from_data` upload,
        /// buffer retained BEFORE the storage moves into the QTensor, so the
        /// fused kernels and the QTensor share a single resident allocation.
        fn tiled_stack(dev: &Device, n_out: usize, k: usize, seed: u64) -> ExpertStack {
            let one = det_tensor(&[n_out, k], seed, 0.5);
            let qt = QTensor::quantize(&one, GgmlDType::Q4K).unwrap();
            let bytes = qt.data().unwrap();
            let mut all = Vec::with_capacity(bytes.len() * N_EXPERT);
            for _ in 0..N_EXPERT {
                all.extend_from_slice(&bytes);
            }
            let storage = QStorage::from_data(Cow::Owned(all), dev, GgmlDType::Q4K).unwrap();
            let buffer = match &storage {
                QStorage::Metal(qms) => Some(Arc::new(qms.buffer().clone())),
                _ => None,
            };
            let qtensor = Arc::new(QTensor::new(storage, (N_EXPERT, n_out, k)).unwrap());
            ExpertStack {
                qtensor: Some(qtensor),
                buffer,
                base_off: 0,
                mmap: None,
                dtype: GgmlDType::Q4K,
                n_expert: N_EXPERT,
                n_out,
                k,
            }
        }

        /// RMSNorm weights near 1.0, as in the attention decode benches.
        fn norm(dim: usize, seed: u64, dev: &Device) -> RmsNorm {
            let w = det_tensor(&[dim], seed, 0.1).affine(1.0, 1.0).unwrap();
            RmsNorm::new(w.to_device(dev).unwrap(), 1e-6)
        }

        /// A residual-stream-shaped input: uniform with RMS ~1, like the output
        /// of a well-conditioned ffn_norm.
        fn normed_input(seed: u64, dev: &Device) -> Tensor {
            det_tensor(&[1, HIDDEN], seed, 1.7).to_device(dev).unwrap()
        }

        /// One production MoeBlock (Fused experts) plus its ffn_norm, at real
        /// geometry. Router weights are scaled small so the sigmoid probs stay
        /// spread out and top-10 selection varies with the input.
        fn build_moe_layer(il: usize, dev: &Device) -> (RmsNorm, MoeBlock) {
            let s = 0x5000 + il as u64 * 64;
            let router_t = det_tensor(&[N_EXPERT, HIDDEN], s + 10, 0.02)
                .to_device(dev)
                .unwrap()
                .t()
                .unwrap()
                .contiguous()
                .unwrap();
            let exp_probs_b = det_tensor(&[N_EXPERT], s + 11, 0.02).to_device(dev).unwrap();
            let experts = FusedExperts {
                gate: tiled_stack(dev, EXPERT_FF, HIDDEN, s + 1),
                up: tiled_stack(dev, EXPERT_FF, HIDDEN, s + 2),
                down: tiled_stack(dev, HIDDEN, EXPERT_FF, s + 3),
            };
            let ql = |t: &Tensor| {
                let qt = QTensor::quantize_onto(t, GgmlDType::Q4K, dev).unwrap();
                QLinear::from_qtensor(Arc::new(qt)).unwrap()
            };
            let shared = SharedExpert {
                gate: ql(&det_tensor(&[SHARED_FF, HIDDEN], s + 20, 0.5)),
                up: ql(&det_tensor(&[SHARED_FF, HIDDEN], s + 21, 0.5)),
                down: ql(&det_tensor(&[HIDDEN, SHARED_FF], s + 22, 0.5)),
            };
            let block = MoeBlock {
                router_t,
                exp_probs_b,
                experts: Box::new(experts),
                shared,
                n_expert_used: TOP_K,
                weights_scale: 2.5,
                weights_norm: true,
            };
            (norm(HIDDEN, s + 30, dev), block)
        }

        /// The f16 token-embedding table `[VOCAB, HIDDEN]` production keeps on
        /// Metal, as 98 tiled row-blocks (lookup timing is value-independent).
        fn build_embed(dev: &Device) -> Tensor {
            let block = det_tensor(&[1024, HIDDEN], 0x111, 1.0)
                .to_device(dev)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            let tiles: Vec<Tensor> = vec![block; VOCAB / 1024];
            let embed = Tensor::cat(&tiles.iter().collect::<Vec<_>>(), 0).unwrap();
            assert_eq!(embed.dims(), &[VOCAB, HIDDEN]);
            embed
        }

        /// A synthetic q6_K lm_head `[VOCAB, HIDDEN]` from 98 tiled quantized
        /// row-blocks (mv timing is value-independent). Same zero-copy
        /// construction as `gguf::qlinear_with_buffer`: the buffer is retained
        /// before the storage moves into the QTensor, and the QTensor must be
        /// kept alive so the shared allocation stays resident.
        fn build_lm_head(
            dev: &Device,
        ) -> (Arc<candle_metal_kernels::metal::Buffer>, Arc<QTensor>) {
            let one = det_tensor(&[1024, HIDDEN], 0x333, 0.5);
            let qt = QTensor::quantize(&one, GgmlDType::Q6K).unwrap();
            let bytes = qt.data().unwrap();
            let mut all = Vec::with_capacity(bytes.len() * (VOCAB / 1024));
            for _ in 0..VOCAB / 1024 {
                all.extend_from_slice(&bytes);
            }
            let storage = QStorage::from_data(Cow::Owned(all), dev, GgmlDType::Q6K).unwrap();
            let buffer = match &storage {
                QStorage::Metal(qms) => Arc::new(qms.buffer().clone()),
                _ => panic!("lm_head bench weights require Metal storage"),
            };
            let qtensor = Arc::new(QTensor::new(storage, (VOCAB, HIDDEN)).unwrap());
            (buffer, qtensor)
        }

        /// One token through 48 MoE layer-evals (4 distinct layers x 12
        /// passes), mirroring the model.rs per-layer FFN half exactly:
        /// ffn_norm -> MoeBlock::forward (route + mv_id experts + shared) ->
        /// residual add. The residual evolves across layer-evals, so the
        /// routing input (and thus the selected experts) differs every pass.
        fn moe_chain(layers: &[(RmsNorm, MoeBlock)], x0: &Tensor) -> Tensor {
            let mut x = x0.clone();
            for _ in 0..PASSES {
                for (ffn_norm, moe) in layers {
                    let normed = ffn_norm.forward(&x).unwrap();
                    let out = moe.forward(&normed).unwrap();
                    x = (&x + &out).unwrap();
                }
            }
            x
        }

        /// Headline: the MoE-FFN half of decode at production geometry through
        /// the PRODUCTION `MoeBlock::forward` path (on-GPU routing + vendored
        /// mv_id gather + silu*mul + weighted combine + shared expert), each
        /// layer wrapped with its ffn_norm and residual add as in model.rs.
        /// Also times routing-only, shared-expert-only and norm+residual-only
        /// variants to split the budget.
        #[test]
        #[ignore = "perf bench"]
        fn moe_decode_ffn_bench() {
            let dev = metal();
            eprintln!(
                "building {N_DISTINCT} synthetic q4_K MoE layers (~5.5GB device memory; \
                 expert stacks are {N_EXPERT} tiled copies of one quantized expert)..."
            );
            let t0 = Instant::now();
            let layers: Vec<(RmsNorm, MoeBlock)> =
                (0..N_DISTINCT).map(|il| build_moe_layer(il, &dev)).collect();
            eprintln!("build+quantize+upload time: {:.1}s", t0.elapsed().as_secs_f64());

            let x0 = normed_input(0x4242, &dev);
            let evals = N_DISTINCT * PASSES;

            let full = bench(
                &format!("moe ffn chain x{evals} (ffn_norm + route + mv_id experts + shared + residual)"),
                || read_scalar(&moe_chain(&layers, &x0)),
            );

            // Isolation variants run over pre-built normed inputs (one per
            // layer-eval) so their inputs vary the same way the chain's do.
            let inputs: Vec<Tensor> =
                (0..evals).map(|i| normed_input(0x9000 + i as u64, &dev)).collect();

            let routing = bench(
                &format!("routing only x{evals} (router matmul + sigmoid + bias + top-k + gather/normalize)"),
                || {
                    let mut last = None;
                    for (i, x) in inputs.iter().enumerate() {
                        let (_ids, w) = layers[i % N_DISTINCT].1.route(x).unwrap();
                        last = Some(w);
                    }
                    read_scalar(&last.unwrap())
                },
            );
            let shared = bench(&format!("shared expert only x{evals}"), || {
                let mut last = None;
                for (i, x) in inputs.iter().enumerate() {
                    last = Some(layers[i % N_DISTINCT].1.shared.forward(x).unwrap());
                }
                read_scalar(&last.unwrap())
            });
            let normres = bench(&format!("ffn_norm + residual add only x{evals}"), || {
                let mut x = x0.clone();
                for i in 0..evals {
                    let normed = layers[i % N_DISTINCT].0.forward(&x).unwrap();
                    x = (&x + &normed).unwrap();
                }
                read_scalar(&x)
            });

            let scale47 = |ms: f64| ms * MOE_LAYERS as f64 / evals as f64;
            eprintln!("{MOE_LAYERS}-layer-equivalent per token:");
            eprintln!("  full MoE FFN half:   {:.3} ms", scale47(full));
            eprintln!("  routing:             {:.3} ms", scale47(routing));
            eprintln!("  shared expert:       {:.3} ms", scale47(shared));
            eprintln!("  ffn_norm + residual: {:.3} ms", scale47(normres));
            eprintln!(
                "  derived mv_id gather + silu*mul + weighted combine: {:.3} ms",
                scale47(full - routing - shared - normres)
            );
            eprintln!(
                "caveat: {N_DISTINCT} distinct layers looped {PASSES}x — repeated expert-stack \
                 reads may be SLC-cached more than 47 distinct layers would be; treat \
                 expert-read numbers as a lower bound"
            );
        }

        /// Per-token sampler + token-feedback overhead, mirroring generate.rs's
        /// decode loop: Sampler::sample (the per-token GPU->CPU logits readback
        /// + CPU top-k/top-p sampling at temp 1.0 / top_k 20) and the follow-on
        /// Tensor::new of the sampled id + f16 embedding row gather + f32
        /// upcast that model.rs's next forward starts with.
        #[test]
        #[ignore = "perf bench"]
        fn sampler_decode_bench() {
            let dev = metal();
            eprintln!(
                "building synthetic f16 embedding [{VOCAB}, {HIDDEN}] (~0.6GB; 98 tiled \
                 row-blocks — lookup timing is value-independent) + logits pool..."
            );
            let embed = build_embed(&dev);

            const POOL: usize = 8;
            let pool: Vec<Tensor> = (0..POOL)
                .map(|i| det_tensor(&[VOCAB], 0x222 + i as u64, 4.0).to_device(&dev).unwrap())
                .collect();
            let mut sampler = Sampler::new(SamplerOptions::default(), vec![]);

            let mut i = 0usize;
            let sample_only = bench("sampler only (logits readback + CPU top-k sample)", || {
                let t = sampler.sample(&pool[i % POOL]).unwrap();
                i += 1;
                t as f32
            });
            let mut j = 0usize;
            let full = bench("sampler + token upload + f16 embed gather + f32 upcast", || {
                let t = sampler.sample(&pool[j % POOL]).unwrap();
                j += 1;
                let input = Tensor::new(&[t], &dev).unwrap();
                let x = embed
                    .index_select(&input.to_dtype(DType::U32).unwrap(), 0)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                read_scalar(&x)
            });
            eprintln!(
                "derived token-feedback (Tensor::new + index_select + upcast): {:.3} ms",
                full - sample_only
            );
            eprintln!(
                "caveat: the derived value includes one extra GPU sync absent in production \
                 (the embed gather is normally absorbed by the next token's layer chain)"
            );
        }

        /// The per-token tail after the layer stack, mirroring model.rs's
        /// decode path: final RMSNorm on [1, hidden] + the q6_K lm_head mat-vec
        /// (vendored `ops::mul_mv` over a retained buffer, QMatMul under
        /// LAGUNA_MV_CLASSIC) + full-logits readback. Cross-checks the earlier
        /// isolated lm_head number (ops::mv_id plain_mv_lmhead_bench, ~0.7 ms).
        #[test]
        #[ignore = "perf bench"]
        fn token_tail_bench() {
            let dev = metal();
            eprintln!(
                "building synthetic q6_K lm_head [{VOCAB}, {HIDDEN}] (~0.3GB; 98 tiled \
                 quantized row-blocks — mv timing is value-independent)..."
            );
            let (buffer, qtensor) = build_lm_head(&dev);
            let qmm = QMatMul::from_arc(qtensor.clone()).unwrap();
            let use_vendored =
                !crate::ops::mv_classic() && crate::ops::mv_vendored_supported(GgmlDType::Q6K);
            eprintln!("lm_head path: {}", if use_vendored { "vendored mul_mv" } else { "QMatMul (classic)" });

            let output_norm = norm(HIDDEN, 0x444, &dev);
            const POOL: usize = 8;
            let pool: Vec<Tensor> =
                (0..POOL).map(|i| normed_input(0x555 + i as u64, &dev)).collect();

            let mut i = 0usize;
            bench("token tail (final RMSNorm + q6_K lm_head mv + full logits readback)", || {
                let x = &pool[i % POOL];
                i += 1;
                let normed = output_norm.forward(x).unwrap();
                let last = normed.narrow(0, 0, 1).unwrap().contiguous().unwrap();
                let logits = if use_vendored {
                    crate::ops::mul_mv(&buffer, GgmlDType::Q6K, VOCAB, HIDDEN, &last).unwrap()
                } else {
                    qmm.forward(&last).unwrap()
                };
                logits.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]
            });
            eprintln!(
                "caveat: the full-logits readback here is the same per-token readback the \
                 sampler bench times; count it once when summing the budget"
            );
        }

        /// The discriminating experiment for the decode budget gap (isolation
        /// stages sum to ~39.7 ms vs 78.7 ms real decode): one iter is one FULL
        /// synthetic decoded token mirroring model.rs + generate.rs end to end —
        /// embed gather (f16, f32 upcast) → 48 x [attn_norm → AttnBlock (12
        /// full / 36 SWA, production geometry) → residual → ffn_norm → MoE
        /// block → residual] → final norm → q6_K lm_head → Sampler::sample
        /// (whose logits readback is the per-iter sync), with the sampled token
        /// fed back as the next iter's input exactly as generate.rs does. So
        /// attention/MoE interaction over the full working set is included, and
        /// the per-decile time series exposes sustained-clock degradation that
        /// a short isolation run's burst clocks would hide. Defaults to 100
        /// iters (LAGUNA_BENCH_ITERS still overrides).
        ///
        /// Approximation: layer 0's dense SwiGLU (ff 12288) is replaced by a
        /// 48th MoE eval — same wrapping norm/residual, different FFN math.
        #[test]
        #[ignore = "perf bench"]
        fn full_stack_decode_bench() {
            use crate::attention::tests::decode_bench as attn_bench;
            use crate::kv_cache::LayerCache;

            /// Sum of the isolated stage measurements (attention chain 25.8 +
            /// MoE FFN 12.5 + tail 0.8 + sampler/feedback ~0.6), 2026-07-22.
            const SUM_OF_PARTS_MS: f64 = 39.7;
            /// Real end-to-end decode ms/token on the Q4_K_M model, 2026-07-22.
            const REAL_DECODE_MS: f64 = 78.7;

            let dev = metal();
            eprintln!(
                "building 48 synthetic attention layers (~22GB) + {N_DISTINCT} q4_K MoE layers \
                 (~5.5GB) + f16 embedding + q6_K lm_head..."
            );
            let t0 = Instant::now();
            let attn_layers = attn_bench::build_layers(&dev, true);
            let cfg = attn_bench::prod_cfg();
            let mut caches: Vec<LayerCache> = attn_bench::build_caches(&cfg, &dev);
            let moe_layers: Vec<(RmsNorm, MoeBlock)> =
                (0..N_DISTINCT).map(|il| build_moe_layer(il, &dev)).collect();
            let embed = build_embed(&dev);
            let (lm_buffer, lm_qtensor) = build_lm_head(&dev);
            let qmm = QMatMul::from_arc(lm_qtensor.clone()).unwrap();
            let use_vendored =
                !crate::ops::mv_classic() && crate::ops::mv_vendored_supported(GgmlDType::Q6K);
            let output_norm = norm(HIDDEN, 0x666, &dev);
            let mut sampler = Sampler::new(SamplerOptions::default(), vec![]);
            eprintln!("build time: {:.1}s", t0.elapsed().as_secs_f64());

            let n_layer = attn_layers.len();
            let mut step = |tok: u32| -> u32 {
                attn_bench::reset_caches(&mut caches);
                let input = Tensor::new(&[tok], &dev).unwrap();
                let mut x = embed
                    .index_select(&input.to_dtype(DType::U32).unwrap(), 0)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                for il in 0..n_layer {
                    x = attn_bench::attn_step(&attn_layers[il], &x, &mut caches[il]);
                    let (ffn_norm, moe) = &moe_layers[il % N_DISTINCT];
                    let normed = ffn_norm.forward(&x).unwrap();
                    let out = moe.forward(&normed).unwrap();
                    x = (&x + &out).unwrap();
                }
                let normed = output_norm.forward(&x).unwrap();
                let last = normed.narrow(0, 0, 1).unwrap().contiguous().unwrap();
                let logits = if use_vendored {
                    crate::ops::mul_mv(&lm_buffer, GgmlDType::Q6K, VOCAB, HIDDEN, &last).unwrap()
                } else {
                    qmm.forward(&last).unwrap()
                };
                sampler.sample(&logits).unwrap()
            };

            // This bench defaults to 100 iters so the decile series can show
            // sustained-vs-burst clock behavior; the env vars still override.
            let get = |k: &str, d: usize| {
                std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
            };
            let (warm, iters) = (get("LAGUNA_BENCH_WARMUP", 10), get("LAGUNA_BENCH_ITERS", 100));

            let mut tok = 42u32;
            for _ in 0..warm {
                tok = step(tok);
            }
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                tok = step(tok);
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }

            let mean = times.iter().sum::<f64>() / times.len() as f64;
            let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
            eprintln!(
                "full-stack decode: mean {mean:.3} ms/token, min {min:.3} ms/token \
                 ({iters} iters, last token {tok})"
            );
            let chunk = (iters / 10).max(1);
            let deciles: Vec<String> = times
                .chunks(chunk)
                .map(|c| format!("{:.2}", c.iter().sum::<f64>() / c.len() as f64))
                .collect();
            eprintln!("time series (means of {chunk}-iter groups, ms): [{}]", deciles.join(", "));
            eprintln!(
                "reference: sum-of-parts from isolation benches ~{SUM_OF_PARTS_MS} ms; \
                 real end-to-end decode {REAL_DECODE_MS} ms/token"
            );
            eprintln!(
                "caveats: layer 0's dense SwiGLU (ff 12288) approximated by a MoE eval; \
                 {N_DISTINCT} distinct MoE layers looped {PASSES}x (SLC caveat as in \
                 moe_decode_ffn_bench)"
            );
        }

        // --- prefill-isolation benches (seq=512) ---------------------------
        //
        // Each isolates one dispatch group of the production MoE block at
        // prefill width so the per-layer prefill budget can be attributed to a
        // category (routing glue / mm_id expert matmuls / silu+rescale glue /
        // weighted combine / shared expert / norm+residual). Synthetic weights
        // at production geometry — never load a model file.
        // `prefill_moe_block_bench` runs the whole production `MoeBlock::forward`
        // (mm_id prefill path) and is the sum-check for the isolated parts.
        // Run one at a time, e.g.
        // `cargo test --release prefill_route_bench -- --ignored --nocapture`.

        /// Prefill chunk length the isolation benches share, so their numbers
        /// are directly comparable (same seq everywhere, top_k 10, hidden 3072,
        /// expert_ff 1024, 256 experts).
        const PREFILL_SEQ: usize = 512;

        /// A [seq, hidden] prefill activation with residual-stream conditioning
        /// (RMS ~1, like a well-behaved ffn_norm output).
        fn prefill_input(seed: u64, dev: &Device) -> Tensor {
            det_tensor(&[PREFILL_SEQ, HIDDEN], seed, 1.7).to_device(dev).unwrap()
        }

        /// Routing glue alone (`MoeBlock::route`): the router matmul plus the
        /// ~10-dispatch decision chain (sigmoid, bias add, top-k argsort,
        /// narrow/contiguous, gather, sum, clamp, div, scale) over x=[512,3072].
        #[test]
        #[ignore = "perf bench"]
        fn prefill_route_bench() {
            let dev = metal();
            let (_ffn_norm, moe) = build_moe_layer(0, &dev);
            let x = prefill_input(0x4242, &dev);
            bench(
                &format!(
                    "prefill routing glue, seq={PREFILL_SEQ} \
                     (router matmul + sigmoid + bias + top-k argsort + gather/normalize/scale)"
                ),
                || {
                    let (_ids, w) = moe.route(&x).unwrap();
                    read_scalar(&w)
                },
            );
        }

        /// The three expert mm_id dispatches (gate/up/down) at prefill width,
        /// each timed on its own. gate/up read one shared activation per token
        /// (x=[512,1,3072] -> [512,10,1024]); down reads a per-slot activation
        /// (act=[512,10,1024] -> [512,10,3072]). Real top-10 selections from the
        /// production router drive the per-expert row grouping, so the
        /// token-to-expert distribution matches a real chunk. Default `_t`
        /// tensor variant (seq >= MM_ID_MIN_SEQ).
        #[test]
        #[ignore = "perf bench"]
        fn prefill_mm_id_bench() {
            let dev = metal();
            let (_ffn_norm, moe) = build_moe_layer(0, &dev);
            let x = prefill_input(0x4242, &dev);
            let (ids, _w) = moe.route(&x).unwrap();

            let gate = tiled_stack(&dev, EXPERT_FF, HIDDEN, 0x9001);
            let up = tiled_stack(&dev, EXPERT_FF, HIDDEN, 0x9002);
            let down = tiled_stack(&dev, HIDDEN, EXPERT_FF, 0x9003);
            let x_g = x.reshape((PREFILL_SEQ, 1, HIDDEN)).unwrap();
            let act = det_tensor(&[PREFILL_SEQ, TOP_K, EXPERT_FF], 0x9004, 0.5)
                .to_device(&dev)
                .unwrap();

            // gate/up/down are identical-cost matmuls, but timing them in three
            // SEPARATE bench() calls let the first (gate) catch the DVFS boost clock
            // and read low while the later ones ran on the clamped sustained clock.
            // One shared warmup then a single interleaved loop with per-projection
            // timers keeps all three on the same (sustained) clock state; each
            // read_scalar still flushes so the per-projection numbers stay isolated.
            let (warm, iters) = iter_counts();
            let mut sink = 0f32;
            for _ in 0..warm {
                sink += read_scalar(&crate::ops::mul_mm_id(&gate, &x_g, &ids).unwrap());
                sink += read_scalar(&crate::ops::mul_mm_id(&up, &x_g, &ids).unwrap());
                sink += read_scalar(&crate::ops::mul_mm_id(&down, &act, &ids).unwrap());
            }
            let mut gate_t = Vec::with_capacity(iters);
            let mut up_t = Vec::with_capacity(iters);
            let mut down_t = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                sink += read_scalar(&crate::ops::mul_mm_id(&gate, &x_g, &ids).unwrap());
                gate_t.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                sink += read_scalar(&crate::ops::mul_mm_id(&up, &x_g, &ids).unwrap());
                up_t.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                sink += read_scalar(&crate::ops::mul_mm_id(&down, &act, &ids).unwrap());
                down_t.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let report = |name: &str, times: &[f64]| {
                let mean = times.iter().sum::<f64>() / times.len() as f64;
                let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
                eprintln!(
                    "{name}: mean {mean:.3} ms/iter, min {min:.3} ms/iter ({iters} iters, sink {sink:.1})"
                );
            };
            report(
                &format!("prefill mm_id gate matmul, seq={PREFILL_SEQ} (x[512,1,3072] -> [512,10,1024])"),
                &gate_t,
            );
            report(
                &format!("prefill mm_id up matmul, seq={PREFILL_SEQ} (x[512,1,3072] -> [512,10,1024])"),
                &up_t,
            );
            report(
                &format!("prefill mm_id down matmul, seq={PREFILL_SEQ} (act[512,10,1024] -> [512,10,3072])"),
                &down_t,
            );
        }

        /// The silu*mul + per-column L2 rescale glue over [512,10,1024], the
        /// default `_t` tensor variant's activation path: it stages the down
        /// input as f16, so it runs the L2 guard (moe.rs FusedExperts::forward).
        /// Mirrors that exact chain — silu(gate)*up, then column L2, scale up by
        /// f16's safe headroom and divide out — with synthetic gate/up inputs.
        #[test]
        #[ignore = "perf bench"]
        fn prefill_silu_rescale_bench() {
            let dev = metal();
            let gate = det_tensor(&[PREFILL_SEQ, TOP_K, EXPERT_FF], 0xA001, 0.5)
                .to_device(&dev)
                .unwrap();
            let up = det_tensor(&[PREFILL_SEQ, TOP_K, EXPERT_FF], 0xA002, 0.5)
                .to_device(&dev)
                .unwrap();
            let f16_safe = 32768.0_f64;
            bench(
                &format!("prefill silu*mul + L2 rescale glue, seq={PREFILL_SEQ} over [512,10,1024]"),
                || {
                    let act = (silu(&gate).unwrap() * up.clone()).unwrap();
                    let col_l2 = act
                        .sqr()
                        .unwrap()
                        .sum_keepdim(2)
                        .unwrap()
                        .sqrt()
                        .unwrap()
                        .clamp(1e-8_f32, 1e30_f32)
                        .unwrap();
                    let act_s = (&act * f16_safe).unwrap().broadcast_div(&col_l2).unwrap();
                    read_scalar(&act_s)
                },
            );
        }

        /// The weighted-combine tail of the rescale branch over [512,10,3072]:
        /// undo the per-column L2 scale, apply the routing weights, sum over the
        /// top-k experts to [512,3072] — the production `ops::combine` fused kernel
        /// (rescale variant), which reads `down` once. Synthetic down / col_l2 /
        /// weights at production shapes.
        #[test]
        #[ignore = "perf bench"]
        fn prefill_combine_bench() {
            let dev = metal();
            let down = det_tensor(&[PREFILL_SEQ, TOP_K, HIDDEN], 0xB001, 0.5)
                .to_device(&dev)
                .unwrap();
            // col_l2 is a positive per-column norm; keep it strictly > 0.
            let col_l2 = det_tensor(&[PREFILL_SEQ, TOP_K, 1], 0xB002, 0.5)
                .abs()
                .unwrap()
                .affine(1.0, 0.1)
                .unwrap()
                .to_device(&dev)
                .unwrap();
            let weights = det_tensor(&[PREFILL_SEQ, TOP_K], 0xB003, 0.25)
                .to_device(&dev)
                .unwrap();
            bench(
                &format!("prefill weighted combine (fused), seq={PREFILL_SEQ} over [512,10,3072] -> [512,3072]"),
                || read_scalar(&crate::ops::combine(&down, Some(&col_l2), &weights).unwrap()),
            );
        }

        /// The always-on shared-expert SwiGLU (q4_K QLinear gate/up/down) at
        /// prefill width, through the production `SharedExpert::forward`.
        #[test]
        #[ignore = "perf bench"]
        fn prefill_shared_expert_bench() {
            let dev = metal();
            let (_ffn_norm, moe) = build_moe_layer(0, &dev);
            let x = prefill_input(0x4242, &dev);
            bench(
                &format!("prefill shared expert SwiGLU (q4_K), seq={PREFILL_SEQ}"),
                || read_scalar(&moe.shared.forward(&x).unwrap()),
            );
        }

        /// The per-layer norm/residual glue model.rs wraps each block with:
        /// attn_norm -> +attn, ffn_norm -> +moe — 2 rms_norm + 2 adds at
        /// [512,3072]. Synthetic block outputs stand in for the attention and
        /// MoE contributions; only the norm+add dispatch cost is being priced.
        #[test]
        #[ignore = "perf bench"]
        fn prefill_norm_residual_bench() {
            let dev = metal();
            let attn_norm = norm(HIDDEN, 0xC001, &dev);
            let ffn_norm = norm(HIDDEN, 0xC002, &dev);
            let x0 = prefill_input(0x4242, &dev);
            let attn_out = prefill_input(0xC003, &dev);
            let moe_out = prefill_input(0xC004, &dev);
            bench(
                &format!("prefill norm+residual glue, seq={PREFILL_SEQ} (2 rms_norm + 2 adds at [512,3072])"),
                || {
                    let _attn_normed = attn_norm.forward(&x0).unwrap();
                    let x = (&x0 + &attn_out).unwrap();
                    let _ffn_normed = ffn_norm.forward(&x).unwrap();
                    let x = (&x + &moe_out).unwrap();
                    read_scalar(&x)
                },
            );
        }

        /// Whole synthetic MoE block at prefill width through the production
        /// `MoeBlock::forward`: ffn_norm -> route -> mm_id experts (default
        /// tensor variant, seq >= MM_ID_MIN_SEQ) -> silu*mul + rescale ->
        /// weighted combine -> +shared expert. The sum-check for the isolated
        /// parts above; prints a 10-iter-group time series so the boost ->
        /// sustained plateau is visible.
        #[test]
        #[ignore = "perf bench"]
        fn prefill_moe_block_bench() {
            let dev = metal();
            let (ffn_norm, moe) = build_moe_layer(0, &dev);
            let x0 = prefill_input(0x4242, &dev);
            let (warm, iters) = iter_counts();
            let step = || -> f32 {
                let normed = ffn_norm.forward(&x0).unwrap();
                read_scalar(&moe.forward(&normed).unwrap())
            };
            let mut sink = 0f32;
            for _ in 0..warm {
                sink += step();
            }
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                sink += step();
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
            eprintln!(
                "prefill MoE block (ffn_norm + route + mm_id experts + silu/rescale + combine + \
                 shared), seq={PREFILL_SEQ}: mean {mean:.3} ms/iter, min {min:.3} ms/iter \
                 ({iters} iters, sink {sink:.1})"
            );
            let chunk = (iters / 10).max(1);
            let series: Vec<String> = times
                .chunks(chunk)
                .map(|c| format!("{:.2}", c.iter().sum::<f64>() / c.len() as f64))
                .collect();
            eprintln!("time series (means of {chunk}-iter groups, ms): [{}]", series.join(", "));
        }
    }
}
