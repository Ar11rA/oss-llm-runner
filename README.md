# LLM Deeper

Learn Large Language Models from first principles — from attention math to production deployment.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What You'll Learn

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM DEEPER LEARNING PATH                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│   │   BASICS     │───▶│   HFCORE     │───▶│  INFERENCE   │                 │
│   │              │    │              │    │  & ADVANCED  │                 │
│   │ • NumPy      │    │ • TinyGPT    │    │              │                 │
│   │ • Attention  │    │ • HuggingFace│    │ • MLX / vLLM │                 │
│   │ • Positional │    │ • LoRA       │    │ • Tool Call  │                 │
│   │   Encoding   │    │ • Fine-tune  │    │ • Deployment │                 │
│   │ • Multi-Head │    │ • Evaluation │    │              │                 │
│   └──────────────┘    └──────────────┘    └──────────────┘                 │
│                                                                             │
│   Week 1               Week 2               Week 3+                        │
│   Foundations          Training             Production                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

This repository provides a **hands-on learning path** for understanding LLMs:

| Module | What You'll Learn |
|--------|-------------------|
| **Basics** | NumPy, attention math, positional encoding, multi-head attention |
| **HFCore** | Build TinyGPT from scratch, HuggingFace, LoRA, fine-tuning, evaluation |
| **Inference & Advanced** | MLX (Apple Silicon), vLLM (CUDA), tool calling, production deployment |

---

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/your-username/llm_deeper.git
cd llm_deeper

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Hardware Requirements

| Platform | Requirements | Notes |
|----------|-------------|-------|
| **CPU** | Any modern CPU | Slow, but works for learning |
| **Apple Silicon** | M1/M2/M3/M4 Mac | Use MLX for fast inference |
| **NVIDIA GPU** | 8GB+ VRAM | Use vLLM for production |

---

## Quick Start

### 1. Run Your First Attention Calculation

```bash
uv run python src/basics/03_attention_math.py
```

### 2. Train TinyGPT (2,453 parameters!)

```bash
uv run python src/hfcore/00_tinygpt.py
```

### 3. Fine-tune with LoRA

```bash
uv run python src/hfcore/04_dataset_plus_finetune_lora.py
```

### 4. Run Inference on Apple Silicon

```bash
uv run python src/inferencing_and_advanced/02_mlx_lora.py
```

---

## Project Structure

```
llm_deeper/
├── src/
│   ├── basics/                    # Foundational concepts
│   │   ├── 00_pandas.py           # Data manipulation
│   │   ├── 01_numpy.py            # Tensor operations
│   │   ├── 02_math.py             # Mathematical foundations
│   │   ├── 03_attention_math.py   # Attention step-by-step
│   │   ├── 04_attention_impl.py   # Full attention implementation
│   │   ├── 05_attention_positional_encoding.py
│   │   ├── 06_mha.py              # Multi-head attention
│   │   └── 07_conclusion.py       # Putting it together
│   │
│   ├── hfcore/                    # HuggingFace training pipeline
│   │   ├── 00_tinygpt.py          # Build GPT from scratch
│   │   ├── 01_transformers.py     # HuggingFace basics
│   │   ├── 02_peft_lora.py        # LoRA concepts
│   │   ├── 03_dataset_plus_finetune_full.py
│   │   ├── 04_dataset_plus_finetune_lora.py
│   │   ├── 05_finetuned_model_inference.py
│   │   └── 06_evals.py            # Evaluation metrics
│   │
│   └── inferencing_and_advanced/  # Production inference
│       ├── 00_base.py
│       ├── 01_peft_to_mlx.py      # Convert PEFT → MLX
│       ├── 02_mlx_lora.py         # MLX inference
│       ├── 03_server.py           # FastAPI server
│       ├── 04_tool_calling_basic.py
│       ├── 05_tool_calling_advanced.py
│       └── 07_vllm_cuda_plus_tools.py
│
├── docs/                          # Detailed documentation
│   ├── 00_overview.md             # Project overview
│   ├── 01_basics.md               # Foundations
│   ├── 02_hfcore.md               # Training pipeline
│   ├── 03_inferencing.md          # Inference
│   ├── 04_tool_calling.md         # Function calling
│   └── 05_deployment.md           # Production deployment
│
├── output_models/                 # Saved models & adapters
├── pyproject.toml
└── README.md
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [00_overview.md](docs/00_overview.md) | Learning path, prerequisites, LLM concepts |
| [01_basics.md](docs/01_basics.md) | Tensors, attention, positional encoding, MHA |
| [02_hfcore.md](docs/02_hfcore.md) | Tokenization, LoRA, fine-tuning, evaluation |
| [03_inferencing.md](docs/03_inferencing.md) | Model formats, quantization, MLX, servers |
| [04_tool_calling.md](docs/04_tool_calling.md) | Function calling, agents, multi-turn |
| [05_deployment.md](docs/05_deployment.md) | vLLM, GPU optimization, cloud deployment |

---

## Key Concepts

### Attention Mechanism

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

The attention mechanism allows tokens to "look at" other tokens and decide how much to focus on each one.

### LoRA (Low-Rank Adaptation)

Instead of fine-tuning all ~600M parameters, LoRA trains only ~3M parameters (0.5%) by adding small adapter matrices:

```
W' = W + BA    (where B is d×r and A is r×d, with r << d)
```

### KV Cache

During generation, transformers compute Key and Value vectors for every token. The KV cache stores these to avoid recomputation:

```
Without cache: O(n²) per token
With cache:    O(n) per token
```

---

## Example: Medical QA Fine-tuning

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from trl import SFTTrainer

# Load base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# Apply LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# Train with SFTTrainer
trainer = SFTTrainer(model=model, train_dataset=dataset, ...)
trainer.train()

# Save adapter (only 12MB vs 1.2GB full model)
model.save_pretrained("./medical-lora-adapter")
```

---

## Inference Options

### Apple Silicon (MLX)

```bash
uv run python -m mlx_lm.server --model ./merged-model --port 8000
```

### NVIDIA GPU (vLLM)

```bash
vllm serve Qwen/Qwen3-0.6B --enable-lora --lora-modules medical=./adapter
```

### API Call

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "medical", "messages": [{"role": "user", "content": "What are symptoms of diabetes?"}]}'
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [HuggingFace](https://huggingface.co/) for Transformers, PEFT, TRL
- [Apple MLX](https://github.com/ml-explore/mlx) for Apple Silicon support
- [vLLM](https://github.com/vllm-project/vllm) for production inference
- [Qwen](https://github.com/QwenLM/Qwen) for excellent open-source models
