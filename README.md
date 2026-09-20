# Hydrogen v0

Hydrogen v0 is an experimental language model project focused on building and testing a custom causal language model implementation.

This repository contains the Hydrogen model implementation and a simple example for loading the model and interacting with it through a chat interface.

## Overview

Hydrogen v0 is designed around a custom model implementation that integrates with the Hugging Face Transformers ecosystem.

The repository currently contains:

```text
Hydrogen_v0/
├── hydrogen/
└── example_chat.py
```

The included example provides a simple way to:

* Load a Hydrogen model.
* Load its tokenizer.
* Automatically use CUDA when a compatible GPU is available.
* Fall back to CPU when CUDA is not available.
* Apply the model's chat template.
* Generate text with streaming output.
* Maintain a multi-turn conversation.
* Exit the chat with `exit` or `quit`.

## Installation

Clone the repository:

```bash
git clone https://github.com/therealtnme/Hydrogen_v0.git
cd Hydrogen_v0
```

Install the required Python dependencies for the project.

The exact dependency requirements can change as Hydrogen develops.

## Example

The repository includes `example_chat.py`.

Run it with:

```bash
python example_chat.py
```

The program asks for the path or identifier of a Hydrogen model:

```text
Path:
```

After loading the model, you can enter prompts:

```text
User: Hello
```

The generated response is streamed to the terminal.

Type:

```text
exit
```

or:

```text
quit
```

to close the program.

## Using a Model

The example uses the Hydrogen model classes:

```python
from hydrogen import HydrogenConfig, HydrogenForCausalLM
```

A model is loaded with its configuration and tokenizer. The example detects whether CUDA is available and moves the model to the selected device.

Generation currently uses sampling parameters including:

* `max_new_tokens=512`
* `temperature=0.7`
* `top_p=0.9`

These values are provided by the example and can be changed for experimentation.

## Research and Development

Hydrogen v0 was developed through an AI-assisted research and testing process.

The research and testing for this project was performed entirely by **Claude Opus 5**, with minimal human interaction.

The human role was primarily to come up with ideas and possible approaches. Claude would then:

1. Research the proposed idea.
2. Determine how it could potentially be implemented.
3. Test the proposed approach.
4. Identify implementation problems or limitations.
5. Drop approaches that could not be implemented successfully.
6. Continue developing the approaches that could be tested and implemented.

In other words, the development process was largely an iterative loop of:

```text
Idea
  ↓
Research
  ↓
Implementation
  ↓
Testing
  ↓
Works?
 ├── Yes → Continue development
 └── No  → Drop the approach
```

This repository therefore represents an experimental development process in which Claude performed the research, implementation investigation, and testing, while the human contribution was primarily the generation of ideas and direction.

### AI Model Used

The primary AI model used for the research and development of Hydrogen v0 was:

**Claude Opus 5**

## Status

Hydrogen v0 is an experimental project.

The architecture, implementation, APIs, and model behavior may change as development continues.

Do not assume that the current implementation represents a final or stable architecture.
