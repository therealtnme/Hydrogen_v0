import math
from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


# Deep-and-thin shapes with tied embeddings (the SmolLM2 tokenizer, 49152 tokens). Small models get more out of depth
# than width, and tying saves 19M-47M parameters that are better spent on layers. Layer counts account for the
# attention output gate (one extra hidden x hidden matrix per layer).
HYDROGEN_PRESETS = {
    "hydrogen-50m": dict(hidden_size=384, intermediate_size=1024, num_hidden_layers=18, num_attention_heads=6, num_key_value_heads=2),
    "hydrogen-135m": dict(hidden_size=576, intermediate_size=1536, num_hidden_layers=28, num_attention_heads=9, num_key_value_heads=3),
    "hydrogen-350m": dict(hidden_size=960, intermediate_size=2560, num_hidden_layers=28, num_attention_heads=15, num_key_value_heads=5),
}


@strict
class HydrogenConfig(PreTrainedConfig):
    r"""
    Configuration for Hydrogen ("Skip or Stop") models. The defaults describe `hydrogen-135m`; use [`HydrogenConfig.from_preset`] for
    other sizes. Every option here earned its place in `test/bench.py`; the disproven ones (a retry verifier, per-prompt
    neuron gating, QK-norm, sandwich norm, value residual, learned RoPE, logit softcapping, whole-layer skipping and
    agreement-by-argmax early exit) were measured and removed - see the project README for their numbers.

    Skip:
        layer_skip (`bool`): Each MLP predicts, per token, whether the *next* layer's MLP runs. Attention always runs, so
            the residual stream and the KV cache stay coherent, and at inference the MLP is evaluated only on the tokens
            that keep it. Measured free or slightly better than dense, and the advantage grows with depth.
        skip_target_rate (`float`): Fraction of tokens that skip each gated layer. A per-layer threshold is calibrated to
            this quantile during training and stored in the checkpoint, so the compute budget is exact rather than the
            accidental result of a penalty weight.
        max_skipped_layers (`int`, *optional*): Cap on how many layers one token may skip, so a token can never skip its
            way through the whole model. `None` allows half the layers; `0` removes the cap.
        skip_init_bias (`float`): Initial router bias; negative starts by running every layer.
        skip_ema (`float`): EMA decay for the calibrated threshold. The threshold is a quantile of that layer's router
            logits, so it only describes the data it was measured on - use
            [`~Hydrogen_1.modeling_hydrogen.recalibrate_skip_thresholds`] on a sample of your workload before deploying.

    Stop (early exit):
        early_exit_every (`int`): Test for an early stop every Nth layer. `0` disables it. A token stops once its
            residual stream has stopped changing (cosine similarity above `exit_hidden_threshold` for
            `early_exit_agreement` consecutive tests), after which the remaining layers only refresh their KV cache.
            This saturation test needs no output projection, which is what makes it cheaper than the model it shortens.
        early_exit_agreement (`int`): Consecutive saturated tests required before a token stops.
        exit_hidden_threshold (`float`): Cosine similarity above which the residual stream counts as settled.
        early_exit_loss_weight (`float`): Weight of the auxiliary loss that teaches intermediate layers to predict, so a
            stopped token's prediction is usable. Training loops that compute their own cross-entropy should add
            `early_exit_loss_weight * mean(CE(exit_logits))` themselves - the returned `exit_logits` exist for that.
        exit_flag_channel (`bool`): Append one extra channel to the logits at index `vocab_size`, holding 1.0 for tokens
            that stopped early. The loss and sampling only use the first `vocab_size` channels; see
            [`HydrogenForCausalLM.split_decided`]. Leave it off for plain `generate()`, which would treat it as a token.

    Example:

    ```python
    >>> from Hydrogen_1 import HydrogenConfig, HydrogenForCausalLM
    >>> model = HydrogenForCausalLM(HydrogenConfig.from_preset("hydrogen-50m"))
    ```"""

    model_type = "hydrogen"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_up_proj": "colwise_gather_output",  # we need to replicate here due to the `chunk` operation
        "layers.*.mlp.down_proj": "rowwise_split_input",  # input is replicated due to the `chunk` operation
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 28
    num_attention_heads: int = 9
    num_key_value_heads: int | None = 3
    head_dim: int | None = 64
    hidden_act: str = "silu"
    attention_dropout: float | int | None = 0.0
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    pad_token_id: int | None = None
    eos_token_id: int | list[int] | None = 0
    bos_token_id: int | None = 0
    attention_bias: bool = False
    attn_output_gate: bool = True

    layer_skip: bool = True
    skip_target_rate: float | int = 0.33
    max_skipped_layers: int | None = None
    skip_init_bias: float | int = -3.0
    skip_ema: float | int = 0.99

    early_exit_every: int = 0
    early_exit_agreement: int = 2
    exit_hidden_threshold: float | int = 0.95
    early_exit_loss_weight: float | int = 0.1
    exit_flag_channel: bool = False

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC
        if not 0 < self.skip_target_rate < 1:
            raise ValueError(f"skip_target_rate must be in (0, 1), got {self.skip_target_rate}")
        if self.max_skipped_layers is not None and self.max_skipped_layers < 0:
            raise ValueError("max_skipped_layers must be >= 0 or None")
        if self.early_exit_every < 0 or self.early_exit_agreement < 1:
            raise ValueError("early_exit_every must be >= 0 and early_exit_agreement >= 1")
        super().__post_init__(**kwargs)

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "HydrogenConfig":
        if name not in HYDROGEN_PRESETS:
            raise ValueError(f"Unknown preset {name!r}; choose from {sorted(HYDROGEN_PRESETS)}")
        return cls(**{**HYDROGEN_PRESETS[name], **overrides})


@dataclass
class HydrogenVisionConfig:
    """
    Vision encoder shape. The field that matters for deployment is `latent_tokens`: an image costs that many tokens of
    the language model's context whatever its resolution, so raising the resolution buys detail with encoder FLOPs
    instead of with context.
    """

    image_size: int = 224
    patch_size: int = 16
    hidden_size: int = 384
    num_layers: int = 6
    num_heads: int = 6
    intermediate_size: int = 1024
    latent_tokens: int = 64
    resampler_heads: int = 6
    rms_norm_eps: float = 1e-6

    @property
    def patches_per_side(self) -> int:
        if self.image_size % self.patch_size:
            raise ValueError(f"image_size {self.image_size} is not divisible by patch_size {self.patch_size}")
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.patches_per_side**2


@dataclass
class HydrogenAudioConfig:
    """
    Audio encoder shape. A 10ms hop gives 100 spectrogram frames per second; the strided convolutions reduce that to
    `tokens_per_second` (12.5 by default), and `latent_tokens > 0` adds a resampler for a fixed budget per clip.
    """

    sample_rate: int = 16000
    n_fft: int = 400
    """25ms analysis window at 16kHz."""
    hop_length: int = 160
    """10ms hop, so the spectrogram has 100 frames per second before downsampling."""
    n_mels: int = 80
    conv_strides: tuple[int, ...] = (2, 2, 2)
    """Each convolution halves the frame rate: three of them turn 100 frames/s into 12.5 tokens/s."""
    hidden_size: int = 384
    num_layers: int = 6
    num_heads: int = 6
    intermediate_size: int = 1024
    latent_tokens: int = 0
    """0 keeps one token per downsampled frame (length grows with duration); >0 fixes the budget per clip."""
    resampler_heads: int = 6
    rms_norm_eps: float = 1e-6

    @property
    def downsample(self) -> int:
        return math.prod(self.conv_strides)

    @property
    def tokens_per_second(self) -> float:
        return self.sample_rate / self.hop_length / self.downsample


__all__ = ["HydrogenConfig", "HYDROGEN_PRESETS", "HydrogenVisionConfig", "HydrogenAudioConfig"]
