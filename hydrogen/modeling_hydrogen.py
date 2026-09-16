import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from torch import nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GenericForSequenceClassification, GenericForTokenClassification, GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from .configuration_hydrogen import HydrogenAudioConfig, HydrogenConfig, HydrogenVisionConfig


@dataclass
class HydrogenModelOutputWithPast(BaseModelOutputWithPast):
    """`skip_rate`, `exit_rate` and `exit_depth` are detached statistics; `decided` marks tokens that stopped early."""

    skip_rate: torch.FloatTensor | None = None
    exit_rate: torch.FloatTensor | None = None
    exit_depth: torch.FloatTensor | None = None
    decided: torch.BoolTensor | None = None
    early_hidden_states: torch.FloatTensor | None = None
    exit_hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class HydrogenCausalLMOutputWithPast(CausalLMOutputWithPast):
    """`early_logits` is what a stopped token would predict; `exit_logits` carries the intermediate heads."""

    skip_rate: torch.FloatTensor | None = None
    exit_rate: torch.FloatTensor | None = None
    exit_depth: torch.FloatTensor | None = None
    decided: torch.BoolTensor | None = None
    early_logits: torch.FloatTensor | None = None
    exit_logits: tuple[torch.FloatTensor, ...] | None = None


class HydrogenMLP(nn.Module):
    """Gated MLP. When skipping is on it also carries the router that decides whether the *next* layer's MLP runs."""

    def __init__(self, config: HydrogenConfig, predicts_skip: bool = False):
        super().__init__()

        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

        self.skip_head = None
        if predicts_skip:
            # Normal init, not zeros: identical logits for every token would make the rate quantile degenerate,
            # because every token would tie with the threshold and none would be selected. The bias sets the start rate.
            self.skip_head = nn.Linear(config.hidden_size, 1)
            self.skip_head._hydrogen_bias_init = config.skip_init_bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(2, dim=-1)

        return self.down_proj(up_states * self.activation_fn(gate))

    def skip_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.skip_head(hidden_states).squeeze(-1).float()


class HydrogenRotaryEmbedding(nn.Module):
    def __init__(self, config: HydrogenConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(config: HydrogenConfig, device=None, **kwargs) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE
        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return inv_freq.to(device), attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        # Disable any outside autocast context if any, to really force fp32
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = position_ids[:, :, None].float() * self.inv_freq.float()
            # Interleaved layout: every frequency covers two adjacent channels
            freqs = freqs.repeat_interleave(2, dim=-1)
            return (freqs.cos() * self.attention_scaling).to(x.dtype), (freqs.sin() * self.attention_scaling).to(x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies interleaved RoPE to the first `cos.shape[-1]` channels of `x` (batch, heads, seq_len, head_dim)."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    return torch.cat([(x_rot * cos) + (rotate_half(x_rot) * sin), x_pass], dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class HydrogenAttention(nn.Module):
    """Grouped-query attention with a query-dependent output gate (worth -0.24 loss at proxy scale)."""

    def __init__(self, config: HydrogenConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.gate_proj = None
        if config.attn_output_gate:
            self.gate_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        compute_output: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """With `compute_output=False` only the keys and values are computed and cached."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings
        key_states = apply_rotary(self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2), cos, sin)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        if not compute_output:
            return None, None

        query_states = apply_rotary(self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2), cos, sin)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if self.gate_proj is not None:
            attn_output = attn_output * torch.sigmoid(self.gate_proj(hidden_states))
        return self.o_proj(attn_output), attn_weights


@use_kernel_forward_from_hub("RMSNorm")
class HydrogenRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        HydrogenRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class HydrogenDecoderLayer(GradientCheckpointingLayer):
    """
    Pre-norm decoder layer. `keep` (batch, seq_len) is 0 for tokens whose MLP is skipped: attention always runs, so the
    residual stream and the KV cache stay coherent, and at inference the MLP runs only on the tokens that keep it.

    Returns `(hidden_states, skip_logits_for_the_next_layer)`.
    """

    def __init__(self, config: HydrogenConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = HydrogenAttention(config=config, layer_idx=layer_idx)

        self.mlp = HydrogenMLP(config, predicts_skip=config.layer_skip and layer_idx < config.num_hidden_layers - 1)
        self.input_layernorm = HydrogenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HydrogenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Calibrated skip threshold. Layer 0 is never skipped: nothing predicts it.
        self.skip_logit_threshold = nn.Buffer(torch.zeros(())) if config.layer_skip and layer_idx else None
        # Inference: run the MLP only on the kept tokens instead of computing it and multiplying by zero.
        # Set to False to force the masked path (used by the equivalence test).
        self.gather_skipped_mlp = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        keep: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        skip_logits = self.mlp.skip_logits(hidden_states) if self.mlp.skip_head is not None else None
        hidden_states = self._run_mlp(hidden_states, keep)
        return residual + hidden_states, skip_logits

    def _run_mlp(self, hidden_states: torch.Tensor, keep: torch.Tensor | None) -> torch.Tensor:
        if keep is None:
            return self.mlp(hidden_states)
        keep = keep.to(hidden_states.dtype).unsqueeze(-1)
        if self.training or not self.gather_skipped_mlp:
            return self.mlp(hidden_states) * keep  # training keeps the straight-through gradient path
        if bool(keep.all()):
            return self.mlp(hidden_states)

        # This is where skipping actually saves work: masking the output afterwards computes every skipped token anyway.
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        index = (keep.reshape(-1) > 0).nonzero(as_tuple=True)[0]
        if index.numel() == 0:
            return torch.zeros_like(hidden_states)
        kept = self.mlp(flat[index].unsqueeze(0)).squeeze(0)
        return torch.zeros_like(flat).index_copy(0, index, kept).view_as(hidden_states)

    def cache_only(self, hidden_states: torch.Tensor, **attn_kwargs) -> None:
        """Refresh this layer's KV cache without computing its output, for tokens that already stopped."""
        self.self_attn(self.input_layernorm(hidden_states), compute_output=False, **attn_kwargs)


class HydrogenPreTrainedModel(PreTrainedModel):
    config: HydrogenConfig
    config_class = HydrogenConfig  # so `from_pretrained` matches the checkpoint's model_type ("hydrogen")
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HydrogenDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = False  # skipping and stopping use data-dependent control flow at inference
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": HydrogenDecoderLayer,
        "attentions": HydrogenAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Linear):
            bias = getattr(module, "_hydrogen_bias_init", None)
            if bias is not None:
                init.constant_(module.bias, bias)
        elif isinstance(module, HydrogenDecoderLayer) and module.skip_logit_threshold is not None:
            init.zeros_(module.skip_logit_threshold)


class HydrogenModel(HydrogenPreTrainedModel):
    def __init__(self, config: HydrogenConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # No padding_idx: with tied embeddings the pad id is often also EOS, whose embedding must keep learning
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [HydrogenDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HydrogenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HydrogenRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # `None` -> half the layers, so no token can skip its way through the whole model; 0 -> uncapped
        budget = config.max_skipped_layers
        self.max_skips = max(1, config.num_hidden_layers // 2) if budget is None else (budget or config.num_hidden_layers)
        every = config.early_exit_every
        # The last layer is the full-depth prediction, so it is not a stopping point
        self.exit_layers = tuple(range(every - 1, config.num_hidden_layers - 1, every)) if every else ()

        # Initialize weights and apply final processing
        self.post_init()

    def set_exit_projection(self, projection: nn.Module) -> None:
        """
        Stores the output head used for early-stop predictions.

        Written straight into `__dict__` so it is *not* registered as a child module: it is the same (tied) `lm_head`
        the causal-LM head already owns, and registering it would duplicate that matrix in every state dict.
        """
        self.__dict__["exit_projection"] = projection

    def _skip_gate(self, layer, skip_logits, skips_used):
        """
        Turns the previous layer's router logits into a `keep` mask, enforcing `max_skipped_layers` per token.

        The boundary is the batch quantile that skips exactly `skip_target_rate` of tokens, and its running EMA is
        stored on the layer so inference reproduces that rate causally, one token at a time.
        """
        config = self.config
        if self.training:
            threshold = torch.quantile(skip_logits.detach().float().flatten(), 1 - config.skip_target_rate)
            if layer.skip_logit_threshold == 0:  # first update: adopt the quantile, do not crawl from zero
                layer.skip_logit_threshold.copy_(threshold)
            else:
                layer.skip_logit_threshold.mul_(config.skip_ema).add_((1 - config.skip_ema) * threshold)
        else:
            threshold = layer.skip_logit_threshold

        skip_prob = torch.sigmoid(skip_logits - threshold)
        hard = (skip_logits > threshold).to(skip_prob.dtype)
        if skips_used is None:
            skips_used = torch.zeros_like(skip_prob)
        allowed = (skips_used < self.max_skips).to(skip_prob.dtype)
        skip_prob, hard = skip_prob * allowed, hard * allowed
        # Straight-through: hard 0/1 decisions forward, sigmoid gradients backward
        keep = 1 - (hard + skip_prob - skip_prob.detach()) if self.training else 1 - hard
        return keep, skips_used + hard

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> HydrogenModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        layer_kwargs = dict(
            attention_mask=causal_mask,
            position_embeddings=self.rotary_emb(hidden_states, position_ids=position_ids),
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        stopping = bool(self.exit_layers) and not self.training
        exit_hidden_states = []
        previous_state = streak = decided = early_hidden = executed_depth = None
        if stopping:
            decided = torch.zeros(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
            early_hidden = torch.zeros_like(hidden_states)
            executed_depth = hidden_states.new_full(hidden_states.shape[:2], float(self.config.num_hidden_layers))

        skip_logits = skips_used = None
        gated_layers = 0
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            keep = None
            if skip_logits is not None:
                keep, skips_used = self._skip_gate(decoder_layer, skip_logits, skips_used)
                gated_layers += 1
            hidden_states, skip_logits = decoder_layer(hidden_states, keep=keep, **layer_kwargs, **kwargs)

            if layer_idx not in self.exit_layers:
                continue
            normed = self.norm(hidden_states)
            if self.training:
                exit_hidden_states.append(normed)  # the head is applied by the causal-LM wrapper
                continue
            if previous_state is None:
                previous_state = normed
                continue

            # A token is decided once its residual stream stops changing. No output projection is involved, which is
            # what makes stopping cheaper than the layers it replaces.
            settled = nn.functional.cosine_similarity(normed, previous_state, dim=-1) > self.config.exit_hidden_threshold
            previous_state = normed
            streak = settled.long() if streak is None else torch.where(settled, streak + 1, torch.zeros_like(streak))
            newly = (streak >= self.config.early_exit_agreement) & ~decided
            if newly.any():
                early_hidden = torch.where(newly[..., None], normed, early_hidden)
                executed_depth = torch.where(newly, torch.full_like(executed_depth, layer_idx + 1.0), executed_depth)
                decided = decided | newly
            if bool(decided.all()):
                # Everything is decided: the rest of the stack only refreshes its KV cache so later tokens can attend
                for remaining in self.layers[layer_idx + 1 : self.config.num_hidden_layers]:
                    remaining.cache_only(hidden_states, **layer_kwargs, **kwargs)
                break

        hidden_states = self.norm(hidden_states)
        exit_rate = exit_depth = None
        if decided is not None:
            # Positions that never settled keep their full-depth prediction
            early_hidden = torch.where(decided[..., None], early_hidden, hidden_states)
            exit_rate = decided.float().mean()
            exit_depth = executed_depth.mean() / self.config.num_hidden_layers
        return HydrogenModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            skip_rate=None if not gated_layers else skips_used.detach().mean() / gated_layers,
            exit_rate=exit_rate,
            exit_depth=exit_depth,
            decided=decided,
            early_hidden_states=early_hidden,
            exit_hidden_states=tuple(exit_hidden_states) or None,
        )


class HydrogenForCausalLM(HydrogenPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = HydrogenModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.set_exit_projection(self.lm_head)  # early stops reuse the tied head, adding no parameters

        # Initialize weights and apply final processing
        self.post_init()

    def split_decided(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits `config.exit_flag_channel` logits into `(token_logits, decided)`; `decided` is a boolean mask."""
        return logits[..., : self.config.vocab_size], logits[..., self.config.vocab_size] > 0.5

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> HydrogenCausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from Hydrogen_1 import HydrogenForCausalLM
        >>> from transformers import AutoTokenizer

        >>> model = HydrogenForCausalLM.from_pretrained("checkpoints/hydrogen-135m")
        >>> tokenizer = AutoTokenizer.from_pretrained("checkpoints/hydrogen-135m")
        >>> inputs = tokenizer("def fibonacci(n):", return_tensors="pt")
        >>> tokenizer.decode(model.generate(**inputs, max_new_tokens=30)[0])
        ```

        `exit_logits` is returned so training loops that compute their own cross-entropy can add the auxiliary term;
        when `labels` is passed it is already included in `loss`.
        """
        outputs: HydrogenModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        early_logits = None
        if outputs.early_hidden_states is not None:
            early_logits = self.lm_head(outputs.early_hidden_states[:, slice_indices, :])
        exit_logits = tuple(self.lm_head(states[:, slice_indices, :]) for states in outputs.exit_hidden_states or ())

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            if exit_logits and self.config.early_exit_loss_weight:
                exits = [
                    self.loss_function(logits=candidate, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
                    for candidate in exit_logits
                ]
                loss = loss + self.config.early_exit_loss_weight * torch.stack(exits).mean()

        if self.config.exit_flag_channel:
            # One extra channel after the vocabulary: 1 where the token stopped early. Sampling and the loss only ever
            # use the first `vocab_size` channels - see `split_decided`.
            if outputs.decided is None:
                flag = torch.zeros_like(logits[..., :1])
            else:
                flag = outputs.decided[:, slice_indices].unsqueeze(-1).to(logits.dtype)
            logits = torch.cat([logits, flag], dim=-1)

        return HydrogenCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            skip_rate=outputs.skip_rate,
            exit_rate=outputs.exit_rate,
            exit_depth=outputs.exit_depth,
            decided=outputs.decided,
            early_logits=early_logits,
            exit_logits=exit_logits or None,
        )


class HydrogenForSequenceClassification(GenericForSequenceClassification, HydrogenPreTrainedModel):
    pass


class HydrogenForTokenClassification(GenericForTokenClassification, HydrogenPreTrainedModel):
    pass


@torch.no_grad()
def recalibrate_skip_thresholds(model: nn.Module, batches, momentum: float = 0.0) -> None:
    """
    Re-estimate the skip thresholds on data from the deployment distribution.

    Those thresholds are quantiles of the router logits, so they only describe the data they were measured on: a model
    calibrated on pretraining text will skip at a different rate on, say, source code. Running a batch or two of
    `input_ids` here restores the configured rate without touching a single weight.

    The default `momentum=0.0` makes each batch *replace* the estimate, which is what restores the rate immediately;
    training's slow EMA would only crawl towards the new distribution and leave the budget short. Prefer one wide batch;
    with several batches a small momentum (e.g. 0.5) averages them instead of letting the last one win.
    """
    config = model.config
    was_training, ema = model.training, config.skip_ema
    config.skip_ema = momentum
    model.train()  # the threshold buffers only update in training mode; without gradients nothing else changes
    try:
        for batch in batches:
            model(input_ids=batch)
    finally:
        model.train(was_training)
        config.skip_ema = ema


def register_hydrogen_auto_classes() -> None:
    """
    Registers Hydrogen with the `Auto*` classes, so `AutoConfig`, `AutoModelForCausalLM` and `AutoTokenizer` work on Hydrogen
    checkpoints (whose `model_type` is `"hydrogen"`). Safe to call repeatedly.
    """
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )

    try:
        AutoConfig.register("hydrogen", HydrogenConfig)
    except ValueError:
        return  # already registered
    AutoModel.register(HydrogenConfig, HydrogenModel)
    AutoModelForCausalLM.register(HydrogenConfig, HydrogenForCausalLM)
    AutoModelForSequenceClassification.register(HydrogenConfig, HydrogenForSequenceClassification)
    AutoModelForTokenClassification.register(HydrogenConfig, HydrogenForTokenClassification)


# --------------------------------------------------------------------------------------------------------------------
# Perception: vision and audio encoders
#
# Both answer the same question - how does the model perceive more without spending context? A naive patch encoder costs
# one token per patch (576 for a 384px image at 16px patches), and a 10ms-hop spectrogram costs 100 tokens per second.
# Both encoders therefore end in a resampler, or in strided convolutions, so that what reaches the language model is a
# small, predictable number of tokens. They share the primitives below because perception is not causal in either case.
# --------------------------------------------------------------------------------------------------------------------


class EncoderRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon)).to(input_dtype)


def sinusoidal_positions(length: int, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Positions with no parameters and no length limit, so one checkpoint serves any duration or resolution."""
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    scale = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / width))
    angles = position * scale
    encoding = torch.zeros(length, width, device=device, dtype=torch.float32)
    encoding[:, 0::2] = angles.sin()
    encoding[:, 1::2] = angles.cos()[:, : encoding[:, 1::2].shape[-1]]
    return encoding.to(dtype)[None]


class EncoderBlock(nn.Module):
    """Pre-norm bidirectional transformer block: perception has no causal structure, so attention is unmasked."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.attention_norm = EncoderRMSNorm(hidden_size, eps)
        self.mlp_norm = EncoderRMSNorm(hidden_size, eps)
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.attention_norm(hidden_states)
        hidden_states = hidden_states + self.attention(normed, normed, normed, need_weights=False)[0]
        normed = self.mlp_norm(hidden_states)
        gate, up = self.gate_up_proj(normed).chunk(2, dim=-1)
        return hidden_states + self.down_proj(up * self.activation(gate))


class Resampler(nn.Module):
    """
    Learned queries cross-attend to the encoded input, fixing how many tokens it occupies.

    This is what keeps perception off the context budget: patches grow with resolution squared and audio frames grow
    with duration, but latents do not. Each latent carries its own embedding, so they can specialize instead of each
    summarizing an arbitrary slice.
    """

    def __init__(self, hidden_size: int, num_latents: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_size) * 0.02)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.latent_norm = EncoderRMSNorm(hidden_size, eps)
        self.input_norm = EncoderRMSNorm(hidden_size, eps)
        self.out_norm = EncoderRMSNorm(hidden_size, eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        latents = self.latent_norm(self.latents).expand(hidden_states.shape[0], -1, -1)
        encoded = self.input_norm(hidden_states)
        return self.out_norm(latents + self.attention(latents, encoded, encoded, need_weights=False)[0])


def encoder_block_flops(tokens: int, width: int, intermediate_size: int) -> float:
    """Matmul FLOPs for one `EncoderBlock`, counting a multiply-add as 2."""
    attention = 2 * 4 * tokens * width * width + 2 * 2 * tokens * tokens * width
    return attention + 2 * 3 * tokens * width * intermediate_size


class HydrogenVisionEncoder(nn.Module):
    """
    Pixels -> a fixed number of language-model tokens.

    `hidden_size` is the *language model's* width; the encoder works in its own `config.hidden_size` and projects once
    at the end, so a small encoder can feed a larger model (or the reverse) without changing either.
    """

    def __init__(self, config: HydrogenVisionConfig, hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(3, config.hidden_size, config.patch_size, stride=config.patch_size)
        self.position_embed = nn.Parameter(torch.randn(1, config.num_patches, config.hidden_size) * 0.02)
        self.blocks = nn.ModuleList(
            EncoderBlock(config.hidden_size, config.num_heads, config.intermediate_size, config.rms_norm_eps)
            for _ in range(config.num_layers)
        )
        self.resampler = Resampler(config.hidden_size, config.latent_tokens, config.resampler_heads, config.rms_norm_eps)
        self.projection = nn.Linear(config.hidden_size, hidden_size, bias=False)

    @property
    def tokens_per_image(self) -> int:
        return self.config.latent_tokens

    def flops_per_image(self, patches_per_side: int | None = None) -> float:
        """Matmul FLOPs for one image. Encoder compute is what buys the context saving, so both are reported."""
        config = self.config
        side = patches_per_side or config.patches_per_side
        patches, width = side * side, config.hidden_size
        patch_embed = 2 * patches * width * 3 * config.patch_size**2
        resampler = 2 * 4 * config.latent_tokens * width * width + 2 * 2 * config.latent_tokens * patches * width
        return patch_embed + config.num_layers * encoder_block_flops(patches, width, config.intermediate_size) + resampler

    def interpolate_positions(self, patches_per_side: int) -> torch.Tensor:
        """Position embeddings are resized, so one checkpoint serves several resolutions."""
        if patches_per_side == self.config.patches_per_side:
            return self.position_embed
        grid = self.position_embed.reshape(1, self.config.patches_per_side, self.config.patches_per_side, -1)
        grid = nn.functional.interpolate(
            grid.permute(0, 3, 1, 2), size=(patches_per_side, patches_per_side), mode="bicubic", align_corners=False
        )
        return grid.permute(0, 2, 3, 1).reshape(1, patches_per_side**2, -1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """`pixel_values` is (batch, 3, height, width) in [-1, 1]; returns (batch, latent_tokens, lm_hidden_size)."""
        patches = self.patch_embed(pixel_values)
        patches_per_side = patches.shape[-1]
        patches = patches.flatten(2).transpose(1, 2)
        patches = patches + self.interpolate_positions(patches_per_side).to(patches.dtype)
        for block in self.blocks:
            patches = block(patches)
        return self.projection(self.resampler(patches))


def mel_filterbank(n_mels: int, n_fft: int, sample_rate: int) -> torch.Tensor:
    """Triangular mel filters, (n_mels, n_fft // 2 + 1). Written in torch so inference needs no audio library."""

    def hz_to_mel(hz):
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    bins = torch.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)
    edges = mel_to_hz(torch.linspace(hz_to_mel(torch.tensor(0.0)), hz_to_mel(torch.tensor(sample_rate / 2.0)), n_mels + 2))
    lower = (bins[None, :] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None])
    upper = (edges[2:, None] - bins[None, :]) / (edges[2:, None] - edges[1:-1, None])
    return torch.clamp(torch.minimum(lower, upper), min=0.0)


class HydrogenAudioEncoder(nn.Module):
    """
    Waveform -> language-model tokens at `config.tokens_per_second` (or a fixed count with `latent_tokens`).

    The mel filterbank is built with plain torch, so a deployed model needs no torchaudio or librosa.
    """

    def __init__(self, config: HydrogenAudioConfig, hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.window = nn.Buffer(torch.hann_window(config.n_fft), persistent=False)
        self.filters = nn.Buffer(mel_filterbank(config.n_mels, config.n_fft, config.sample_rate), persistent=False)

        channels, convolutions = config.n_mels, []
        for stride in config.conv_strides:
            convolutions += [nn.Conv1d(channels, config.hidden_size, kernel_size=3, stride=stride, padding=1), nn.SiLU()]
            channels = config.hidden_size
        self.convolutions = nn.Sequential(*convolutions)

        self.blocks = nn.ModuleList(
            EncoderBlock(config.hidden_size, config.num_heads, config.intermediate_size, config.rms_norm_eps)
            for _ in range(config.num_layers)
        )
        self.resampler = (
            Resampler(config.hidden_size, config.latent_tokens, config.resampler_heads, config.rms_norm_eps)
            if config.latent_tokens
            else None
        )
        self.projection = nn.Linear(config.hidden_size, hidden_size, bias=False)

    def tokens_for(self, seconds: float) -> int:
        """How much context a clip of this length will occupy."""
        if self.resampler is not None:
            return self.config.latent_tokens
        frames = int(seconds * self.config.sample_rate) // self.config.hop_length + 1
        return math.ceil(frames / self.config.downsample)

    def flops_per_second(self) -> float:
        """Matmul FLOPs for one second of audio (convolutions, blocks and one resampler pass)."""
        config = self.config
        tokens, width = max(1, int(config.tokens_per_second)), config.hidden_size
        convolutions, channels, rate = 0.0, config.n_mels, config.sample_rate / config.hop_length
        for stride in config.conv_strides:
            rate = rate / stride
            convolutions += 2 * rate * width * channels * 3
            channels = width
        resampler = 0.0
        if self.resampler is not None:
            resampler = 2 * 4 * config.latent_tokens * width * width + 2 * 2 * config.latent_tokens * tokens * width
        return convolutions + config.num_layers * encoder_block_flops(tokens, width, config.intermediate_size) + resampler

    def resample(self, waveform: torch.Tensor, sample_rate: int | None) -> torch.Tensor:
        """
        Bring audio to the rate this encoder was configured for.

        The hop is measured in samples, so a 48kHz file would otherwise produce 300 spectrogram frames per second
        instead of 100 and silently cost three times the context. Linear interpolation keeps the model free of any
        audio library; resample offline with a better filter if fidelity matters.
        """
        if not sample_rate or sample_rate == self.config.sample_rate:
            return waveform
        target = round(waveform.shape[-1] * self.config.sample_rate / sample_rate)
        return nn.functional.interpolate(waveform[:, None], size=target, mode="linear", align_corners=False)[:, 0]

    def log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """(batch, samples) -> (batch, n_mels, frames), normalized the way Whisper does for stable scales."""
        spectrogram = torch.stft(
            waveform,
            self.config.n_fft,
            self.config.hop_length,
            window=self.window.to(waveform.dtype),
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )
        mel = self.filters.to(spectrogram.real.dtype) @ spectrogram.abs().pow(2)
        log_mel = torch.log10(mel.clamp_min(1e-10))
        log_mel = torch.maximum(log_mel, log_mel.amax(dim=(-2, -1), keepdim=True) - 8.0)
        return (log_mel + 4.0) / 4.0

    def forward(self, waveform: torch.Tensor, sample_rate: int | None = None) -> torch.Tensor:
        """
        `waveform` is (batch, samples); returns (batch, tokens, lm_hidden_size).

        Pass the file's `sample_rate` whenever it may differ from `config.sample_rate` - the audio is resampled so that
        a clip costs the same context no matter how it was recorded.
        """
        features = self.convolutions(self.log_mel(self.resample(waveform, sample_rate))).transpose(1, 2)
        features = features + sinusoidal_positions(features.shape[1], features.shape[-1], features.device, features.dtype)
        for block in self.blocks:
            features = block(features)
        if self.resampler is not None:
            features = self.resampler(features)
        return self.projection(features)


__all__ = [
    "HydrogenPreTrainedModel",
    "HydrogenModel",
    "HydrogenForCausalLM",
    "HydrogenForSequenceClassification",
    "HydrogenForTokenClassification",
    "HydrogenVisionEncoder",
    "HydrogenAudioEncoder",
    "EncoderBlock",
    "Resampler",
    "mel_filterbank",
    "recalibrate_skip_thresholds",
    "register_hydrogen_auto_classes",
]
