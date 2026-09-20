# Hydrogen v0

Hydrogen v0 is an experimental multimodal language model built on top of the Hugging Face Transformers architecture.

The project explores compute-efficient inference through **token-level MLP skipping**, **early exit**, and fixed-budget **vision and audio perception**.

Hydrogen is designed to remain compatible with the Transformers ecosystem while adding dynamic computation to the model.

## Features

### Decoder-only language model

Hydrogen uses a pre-norm decoder architecture with:

* RMS normalization
* Gated MLPs
* Rotary positional embeddings (RoPE)
* Grouped-query attention (GQA)
* Optional attention output gating
* Tied input embeddings and language-model output weights
* KV caching through the Transformers `Cache` API
* Flash Attention, SDPA, and Flex Attention support through Transformers attention backends
* Gradient checkpointing support

### Token-level MLP skipping

Hydrogen can skip the MLP computation for individual tokens at individual layers.

Attention still runs for every token. This keeps the residual stream and KV cache coherent while allowing the expensive MLP computation to be skipped for selected tokens.

The skip system uses a learned router attached to the MLP:

```text
Hidden state
     │
     ▼
Skip router
     │
     ├── Skip MLP
     │
     └── Run MLP
```

The router is calibrated using a target skip rate. During training, the threshold is updated from the batch quantile and smoothed with an EMA.

At inference time, the stored threshold is used to make hard skip decisions.

The implementation also supports a per-token maximum skip budget so that a token cannot skip an unlimited number of layers.

When inference skipping is enabled, Hydrogen gathers only the tokens that need the MLP instead of running the MLP for every token and multiplying the result by a mask afterward.

This is important because masking the output after the MLP would still perform the skipped computation.

### Early exit

Hydrogen also supports token-level early exit.

At configured intermediate layers, the model compares the normalized hidden state against the hidden state from the previous exit point.

A token is considered settled after the configured number of consecutive exit points produce sufficiently similar hidden states.

Once a token is decided:

* Its intermediate hidden state is saved.
* It no longer needs to execute the remaining decoder layers for its prediction.
* The remaining layers still refresh the KV cache when required so later tokens can attend to the correct sequence state.
* The saved hidden state is passed through the same tied language-model head used by the full-depth prediction.

This allows early exit without requiring a separate output projection for every exit point.

The model exposes statistics including:

* `exit_rate`
* `exit_depth`
* `decided`
* `early_logits`
* `exit_logits`

### Combined dynamic computation

MLP skipping and early exit operate at different levels.

MLP skipping can reduce computation inside the decoder stack while a token continues through the model.

Early exit can stop a token from executing later decoder layers once its representation has stabilized.

Conceptually:

```text
Token
  │
  ▼
Layer 1
  │
  ├── MLP
  │
  ▼
Layer 2
  │
  ├── Skip MLP for this token
  │
  ▼
Layer N
  │
  ├── Representation has stabilized
  │
  ▼
Early exit
```

The goal is to reduce unnecessary computation while preserving the normal causal attention structure.

## Vision

Hydrogen includes a vision encoder that converts an image into a fixed number of language-model tokens.

The vision pipeline is:

```text
Image
  │
  ▼
Patch embedding
  │
  ▼
Bidirectional Transformer blocks
  │
  ▼
Learned latent resampler
  │
  ▼
Projection
  │
  ▼
Language-model tokens
```

The encoder uses:

* Convolutional patch embedding
* Learned 2D position embeddings
* Bidirectional Transformer blocks
* RMS normalization
* Gated MLPs
* Learned latent queries
* Cross-attention resampling
* A final projection into the language model's hidden size

The resampler is important because the number of image patches grows with image resolution, while the number of output latent tokens remains fixed.

For example, an image may contain hundreds of patches while the language model receives only the configured number of latent tokens.

Position embeddings can also be interpolated, allowing a checkpoint to operate at different image resolutions.

## Audio

Hydrogen includes an audio encoder that converts a waveform into language-model tokens.

The pipeline is:

```text
Waveform
  │
  ▼
Resampling
  │
  ▼
STFT
  │
  ▼
Mel filterbank
  │
  ▼
Convolutional downsampling
  │
  ▼
Bidirectional Transformer blocks
  │
  ▼
Optional latent resampler
  │
  ▼
Projection
  │
  ▼
Language-model tokens
```

The audio implementation is intentionally written using PyTorch operations.

The mel filterbank is constructed directly with PyTorch, so the encoder does not require an external audio library such as `torchaudio` or `librosa` for its core processing.

The encoder can operate in two modes:

* A configured token rate based on audio duration.
* A fixed number of latent tokens when the resampler is enabled.

Audio is resampled to the encoder's configured sample rate when necessary so that the same duration produces the expected number of tokens regardless of the original recording sample rate.

## Perception token budgeting

A central design goal of the vision and audio encoders is to control how much context perception consumes.

Raw perception inputs can become expensive very quickly:

```text
Higher image resolution
        ↓
More patches
        ↓
More context tokens
```

and:

```text
Longer audio
        ↓
More spectrogram frames
        ↓
More context tokens
```

Hydrogen reduces this cost by compressing perception features before they reach the language model.

The vision encoder uses learned latent queries to produce a fixed number of tokens.

The audio encoder can use the same approach when configured with latent tokens.

The encoders also expose FLOP-estimation methods to make the cost of perception processing explicit:

* `HydrogenVisionEncoder.flops_per_image()`
* `HydrogenAudioEncoder.flops_per_second()`

## Architecture

At a high level, Hydrogen consists of three main components:

```text
                    ┌──────────────────┐
                    │  Vision Encoder  │
                    └────────┬─────────┘
                             │
                             ▼
                         LM tokens
                             │
                             │
┌──────────────┐             │
│ Audio Encoder├─────────────┤
└──────────────┘             │
                             ▼
                    ┌──────────────────┐
                    │ Hydrogen Decoder │
                    │                  │
                    │ Attention        │
                    │ MLP skipping     │
                    │ Early exit       │
                    └────────┬─────────┘
                             │
                             ▼
                       LM Head
```

The language model and perception encoders are separate components.

The vision and audio encoders project their outputs into the language model's hidden dimension. This allows their internal widths to differ from the language model width.

## Hugging Face Transformers Integration

Hydrogen implements Transformers-compatible model classes including:

```python
HydrogenPreTrainedModel
HydrogenModel
HydrogenForCausalLM
HydrogenForSequenceClassification
HydrogenForTokenClassification
```

Hydrogen also provides:

```python
register_hydrogen_auto_classes()
```

This registers the model with the Transformers `Auto*` classes so Hydrogen checkpoints using:

```text
model_type = "hydrogen"
```

can be loaded through the standard Auto API.

The causal language model supports the normal `GenerationMixin` generation interface.

## Model Outputs

In addition to standard Transformers outputs, Hydrogen exposes information about its dynamic computation.

### `HydrogenModelOutputWithPast`

Additional fields include:

| Field                 | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| `skip_rate`           | Fraction of eligible layer/token computations skipped        |
| `exit_rate`           | Fraction of tokens that exited early                         |
| `exit_depth`          | Average normalized depth at which tokens exited              |
| `decided`             | Boolean mask identifying tokens that exited early            |
| `early_hidden_states` | Hidden states selected for early-exited tokens               |
| `exit_hidden_states`  | Intermediate hidden states collected at training exit points |

### `HydrogenCausalLMOutputWithPast`

Additional fields include:

| Field          | Description                                           |
| -------------- | ----------------------------------------------------- |
| `skip_rate`    | MLP skip statistic                                    |
| `exit_rate`    | Early-exit statistic                                  |
| `exit_depth`   | Average normalized exit depth                         |
| `decided`      | Early-exit token mask                                 |
| `early_logits` | Logits generated from early-exited hidden states      |
| `exit_logits`  | Logits generated at intermediate training exit points |

## Early-Exit Training

During training, Hydrogen can collect intermediate exit representations.

If `early_exit_loss_weight` is enabled, the model computes an auxiliary language-model loss for these intermediate predictions and adds their mean to the normal final-layer loss.

This allows intermediate representations to receive direct language-model supervision.

At inference time, the early-exit mechanism instead uses hidden-state agreement to determine when a token has stabilized.

## Skip Threshold Calibration

Skip thresholds depend on the distribution of router logits.

A threshold calibrated on one type of data can produce a different skip rate on another type of data.

Hydrogen therefore provides:

```python
recalibrate_skip_thresholds(model, batches)
```

This recalibrates the stored thresholds using deployment data without changing model weights.

For example, thresholds calibrated on general pretraining text may not produce the same skip behavior on source code.

The calibration function can therefore be used to adapt the configured skip rate to the data distribution used during deployment.

## Example

The repository includes `example_chat.py` for basic interactive generation.

Run:

```bash
python example_chat.py
```

The example loads a Hydrogen checkpoint and tokenizer, selects CUDA when available, and provides a streaming multi-turn chat interface.

## Repository Structure

```text
Hydrogen_v0/
├── hydrogen/
│   ├── configuration_hydrogen.py
│   ├── modeling_hydrogen.py
│   └── ...
├── example_chat.py
└── README.md
```

## Research and Development

Hydrogen v0 was developed through an AI-driven research and testing process.

All research and testing for the project was performed entirely by **Claude**, with minimal human interaction.

The human contribution was primarily to generate ideas and possible directions.

The development process then followed an iterative loop:

```text
Human proposes an idea
        │
        ▼
Claude researches the idea
        │
        ▼
Claude determines how it could be implemented
        │
        ▼
Claude implements and tests it
        │
        ├───────────────┐
        │               │
        ▼               ▼
   Can be implemented   Cannot be implemented
        │               │
        ▼               ▼
   Continue testing    Drop the idea
        │
        ▼
     Iterate
```

Ideas that could not be implemented successfully were dropped rather than forced into the architecture.

The human therefore acted primarily as the source of ideas and direction, while Claude performed the research, implementation investigation, and testing required to determine which ideas could actually work.

### AI Model

The Claude model used for the research and development was:

**Claude Opus 5**

## Project Status

Hydrogen v0 is an experimental research project.

The architecture and implementation are subject to change as new ideas are tested.

The current implementation should therefore be treated as an experimental model architecture rather than a stable production API.
