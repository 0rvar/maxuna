use std::sync::Arc;

use anyhow::Result;
use candle_core::quantized::QTensor;
use candle_core::{DType, Tensor};
use candle_nn::ops::{sigmoid, silu};

use crate::config::LagunaConfig;
use crate::gguf::{ExpertStack, QLinear, Weights};
use crate::ops::{ExpertRunner, mv_id};

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
        Ok(Self {
            gate: w.expert_stack("ffn_gate_exps")?,
            up: w.expert_stack("ffn_up_exps")?,
            down: w.expert_stack("ffn_down_exps")?,
        })
    }
}

impl ExpertFfn for FusedExperts {
    fn forward(&self, x: &Tensor, ids: &Tensor, weights: &Tensor) -> Result<Tensor> {
        let (seq, hidden) = x.dims2()?;
        let top_k = ids.dim(1)?;
        let x = x.to_dtype(DType::F32)?;

        // Both prefill and decode dispatch through mul_mv_id. The mm_id kernel
        // (token-grouped matmul) was measured slower here: candle's
        // kernel_mul_mm_id re-scans the whole ids buffer in every one of its
        // ~n_expert * (n_out/64) threadgroups, and for a 256-expert model that
        // redundant scan costs more than the expert-weight-reuse it buys. See
        // the WP9 report; a proper fix needs the ggml two-pass row-map kernel.

        // gate/up share one activation per token: x_per_row = 1.
        let x_g = x.reshape((seq, 1, hidden))?;
        let gate = mv_id::mul_mv_id(&self.gate, &x_g, ids)?; // [seq, top_k, expert_ff]
        let up = mv_id::mul_mv_id(&self.up, &x_g, ids)?; // [seq, top_k, expert_ff]
        let act = (silu(&gate)? * up)?; // [seq, top_k, expert_ff]

        // Rescale each (token, expert) column by its L2 norm before the down
        // projection so the kernel's f16 cast of the activation cannot overflow;
        // the factor is divided back out afterwards (a per-column identity).
        let f16_safe = 32768.0_f64;
        let col_l2 = act
            .sqr()?
            .sum_keepdim(2)?
            .sqrt()?
            .clamp(1e-8_f32, 1e30_f32)?; // [seq, top_k, 1]
        let act_s = (act * f16_safe)?.broadcast_div(&col_l2)?;

        let down = mv_id::mul_mv_id(&self.down, &act_s, ids)?; // [seq, top_k, hidden]
        let down = (down.broadcast_mul(&col_l2)? * (1.0 / f16_safe))?;

        // Weight each expert's output (unbiased, scaled weights) and sum.
        let w = weights.to_dtype(DType::F32)?.reshape((seq, top_k, 1))?;
        Ok(down.broadcast_mul(&w)?.sum(1)?)
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
}
