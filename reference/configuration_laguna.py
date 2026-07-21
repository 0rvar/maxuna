# ruff: noqa
# Copyright 2025 Poolside and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available


class LagunaConfig(PreTrainedConfig):
    r"""
    Configuration class for Laguna model.

    Laguna is Poolside's MoE architecture with:
    - Attention output gating (softplus gate)
    - Sigmoid routing instead of softmax
    - No QKV bias
    - Explicit head_dim parameter

    Args:
        head_dim (`int`, *optional*, defaults to 128):
            Dimension of attention heads. Laguna uses explicit head_dim rather than
            computing it from hidden_size // num_attention_heads.
        qkv_bias (`bool`, *optional*, defaults to `False`):
            Whether to add bias to QKV projections. Laguna uses no QKV bias.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to add bias to attention output projection. Laguna uses no attention bias.
        gating (`bool` or `str`, *optional*, defaults to `True`):
            Attention output gating mode. When ``True`` or ``"per-element"`` a g_proj
            linear layer with output size ``num_attention_heads * head_dim`` is added
            and ``attn_output = attn_output * softplus(g_proj(x))``. When ``"per-head"``
            g_proj has output size ``num_attention_heads`` and the gate broadcasts across
            ``head_dim``. When ``False`` no gating is applied.
        partial_rotary_factor (`float`, *optional*):
            Fraction of head_dim to apply rotary embeddings to. When set, this value is
            injected into ``rope_parameters`` (and ``swa_rope_parameters``) if not already
            specified there. When ``None`` the default behaviour of the rope implementation
            is used (typically full rotary).
        num_attention_heads_per_layer (`list[int]`, *optional*):
            Optional per-layer override for ``num_attention_heads``. When provided the list
            length must equal ``num_hidden_layers`` and each entry is the head count used by
            that layer. When ``None`` every layer uses ``num_attention_heads``.
        vocab_size (`int`, *optional*, defaults to 100352):
            Vocabulary size of the Laguna model.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimension of the MLP representations for dense layers.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value heads for GQA.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            Maximum sequence length.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMSNorm layers.
        sliding_window (`int`, *optional*):
            Sliding window attention size. Used by layers whose type in ``layer_types``
            is ``"sliding_attention"``. When ``None``, all layers use full attention.
        layer_types (`list[str]`, *optional*):
            Per-layer attention type. Each element should be ``"sliding_attention"`` or
            ``"full_attention"``. Length must equal ``num_hidden_layers``. When ``None``,
            all layers default to global attention.
        swa_attention_sink_enabled (`bool`, *optional*, defaults to `False`):
            Whether to enable learnable attention sinks on sliding-window attention layers.
            When enabled, a per-head bias parameter is added that allows the model to attend
            to position 0 even when it falls outside the sliding window.
        swa_rope_parameters (`RopeParameters`, *optional*):
            Separate RoPE configuration for sliding-window attention layers. When ``None``,
            SWA layers use the same RoPE as global attention layers.
        num_experts (`int`, *optional*, defaults to 256):
            Number of routed experts.
        num_experts_per_tok (`int`, *optional*, defaults to 16):
            Number of experts selected per token (top-k).
        moe_intermediate_size (`int`, *optional*, defaults to 1024):
            Intermediate size of routed experts.
        shared_expert_intermediate_size (`int`, *optional*, defaults to 1024):
            Intermediate size of the shared expert.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize top-k routing probabilities.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            Frequency of MoE layers (1 = every layer is MoE after mlp_only_layers).
        mlp_only_layers (`list[int]`, *optional*, defaults to `[0]`):
            Layer indices that use dense MLP instead of MoE.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            Auxiliary loss coefficient for load balancing.
        moe_routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scalar multiplier applied to the routed-expert output before combining with the
            shared-expert output.
        moe_apply_router_weight_on_input (`bool`, *optional*, defaults to `False`):
            When ``True`` the top-k routing weights are multiplied into each expert's input
            rather than its output. Matches the numerical form used by the trained checkpoint.
        moe_router_logit_softcapping (`float`, *optional*, defaults to 0.0):
            Optional soft-capping value ``c`` applied to router logits as
            ``x = tanh(x / c) * c`` before sigmoid + top-k. Disabled when ``0``.
        rope_parameters (`RopeParameters`, *optional*):
            RoPE configuration. Defaults to rope_theta=500000.0.
    """

    model_type = "laguna"
    keys_to_ignore_at_inference = ["past_key_values"]
    # PreTrainedConfig in transformers v5 no longer auto-declares these; subclasses
    # opt in by providing class-level annotations with defaults.
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.g_proj": "colwise",  # Laguna-specific gating projection
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int = 100352,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        qkv_bias: bool = False,
        attention_bias: bool = False,
        gating: bool | str = True,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        partial_rotary_factor: float | None = None,
        attention_dropout: float = 0.0,
        sliding_window: int | None = None,
        layer_types: list[str] | None = None,
        num_attention_heads_per_layer: list[int] | None = None,
        swa_attention_sink_enabled: bool = False,
        swa_rope_parameters: RopeParameters | None = None,
        num_experts: int = 256,
        num_experts_per_tok: int = 16,
        moe_intermediate_size: int = 1024,
        shared_expert_intermediate_size: int = 1024,
        norm_topk_prob: bool = True,
        decoder_sparse_step: int = 1,
        mlp_only_layers: list[int] | None = None,
        router_aux_loss_coef: float = 0.001,
        moe_routed_scaling_factor: float = 1.0,
        moe_apply_router_weight_on_input: bool = False,
        moe_router_logit_softcapping: float = 0.0,
        output_router_logits: bool = False,
        **kwargs,
    ):
        # Default mlp_only_layers: first layer is dense (moe_first_k_dense_replace=1)
        if mlp_only_layers is None:
            mlp_only_layers = [0]

        # Default layer_types: all layers use full attention (Laguna-M). Laguna-XS
        # ships an explicit list with a mix of "full_attention" and "sliding_attention".
        # Downstream mask builders (``create_masks_for_generate``) iterate
        # ``layer_types``, so it must be a list — not left as ``None``.
        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers

        # Default rope_parameters with Laguna's theta
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": 500000.0}

        # config.json stores SWA rope nested in rope_parameters["sliding_attention"]
        # and carries no top-level swa_rope_parameters. Derive it here, else the
        # sliding-window layers silently reuse the full-attention rope.
        if swa_rope_parameters is None and isinstance(rope_parameters, dict):
            swa_rope_parameters = rope_parameters.get("sliding_attention")

        # If ``partial_rotary_factor`` is set at the top level, inject it into any
        # rope dict that does not already carry one so the rotary embedding picks
        # it up consistently for both full-attention and SWA layers.
        if partial_rotary_factor is not None:
            if isinstance(rope_parameters, dict) and "partial_rotary_factor" not in rope_parameters:
                rope_parameters = {**rope_parameters, "partial_rotary_factor": partial_rotary_factor}
            if isinstance(swa_rope_parameters, dict) and "partial_rotary_factor" not in swa_rope_parameters:
                swa_rope_parameters = {
                    **swa_rope_parameters,
                    "partial_rotary_factor": partial_rotary_factor,
                }

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qkv_bias = qkv_bias
        self.attention_bias = attention_bias
        self.gating = gating
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_dropout = attention_dropout
        # Sliding window attention arguments
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.num_attention_heads_per_layer = num_attention_heads_per_layer
        self.swa_attention_sink_enabled = swa_attention_sink_enabled
        self.swa_rope_parameters = swa_rope_parameters
        # MoE arguments
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.decoder_sparse_step = decoder_sparse_step
        self.mlp_only_layers = mlp_only_layers
        self.router_aux_loss_coef = router_aux_loss_coef
        self.moe_routed_scaling_factor = moe_routed_scaling_factor
        self.moe_apply_router_weight_on_input = moe_apply_router_weight_on_input
        self.moe_router_logit_softcapping = moe_router_logit_softcapping
        self.output_router_logits = output_router_logits

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["LagunaConfig"]
