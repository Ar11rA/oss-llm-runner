"""
Fine-tuning with LoRA (Low-Rank Adaptation)

LoRA drastically reduces trainable parameters:
- Full fine-tuning: ~0.6B params
- LoRA fine-tuning: ~1-10M params (0.1% of model)

This makes fine-tuning possible on consumer hardware.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer

# ============================================================================
# 1. Load Dataset
# ============================================================================
print("Loading dataset...")
dataset = load_dataset("OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B")

print(f"Dataset size: {len(dataset['train'])} samples")


# ============================================================================
# 2. Format Dataset for Chat
# ============================================================================
def format_conversation(example):
    """Convert messages to a single text string."""
    conversation = ""
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        conversation += f"<|{role}|>\n{content}\n"
    conversation += "<|end|>"
    return {"text": conversation}


formatted_dataset = dataset.map(format_conversation)
print(f"Formatted dataset sample:\n{formatted_dataset['train'][0]['text'][:300]}...")


# ============================================================================
# 3. Load Model and Tokenizer
# ============================================================================
model_id = "Qwen/Qwen3-0.6B"  # 0.6B parameter model

# Detect device
if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32  # MPS doesn't support fp16 training
elif torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"Loading model: {model_id}")
print(f"Using device: {device}, dtype: {dtype}")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Set padding token (required for training)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=dtype,
    device_map="auto" if device != "mps" else None,
    trust_remote_code=True,
)

if device == "mps":
    model = model.to(device)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()


# ============================================================================
# 4. Configure LoRA
# ============================================================================
"""
LoRA Key Parameters:
- r: Rank of the low-rank matrices (lower = fewer params, higher = more capacity)
- lora_alpha: Scaling factor (typically = r or 2*r)
- target_modules: Which layers to apply LoRA to
- lora_dropout: Dropout for regularization
"""

lora_config = LoraConfig(
    r=16,                          # Rank: balance between quality and efficiency
    lora_alpha=32,                 # Scaling factor (alpha/r = scaling)
    target_modules=[               # Qwen3 attention layers
        "q_proj",                  # Query projection
        "k_proj",                  # Key projection
        "v_proj",                  # Value projection
        "o_proj",                  # Output projection
        "gate_proj",               # FFN gate
        "up_proj",                 # FFN up projection
        "down_proj",               # FFN down projection
    ],
    lora_dropout=0.05,             # Small dropout for regularization
    bias="none",                   # Don't train biases
    task_type=TaskType.CAUSAL_LM,  # Causal language modeling
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# Example output: "trainable params: 3,407,872 || all params: 630,000,000 || trainable%: 0.54%"


# ============================================================================
# 5. Training Configuration
# ============================================================================
training_args = SFTConfig(
    output_dir="./output_models/qwen3-0.6b-medical-lora",
    
    # Batch size settings
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,    # Effective batch = 2 * 16 = 32
    
    # Training duration
    num_train_epochs=1,
    max_steps=100,                     # Limit steps for testing
    
    # Learning rate (higher for LoRA)
    learning_rate=2e-4,
    warmup_ratio=0.03,
    
    # Optimizer
    optim="adamw_torch",
    weight_decay=0.01,
    
    # Precision
    fp16=(device == "cuda"),
    
    # Logging & Saving
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    
    # Memory optimization
    gradient_checkpointing=True,
    dataloader_pin_memory=False,       # Required for MPS
    
    # SFT-specific settings
    max_length=512,                    # Max sequence length
    packing=False,                     # Don't pack sequences
    
    # Misc
    report_to="none",
    remove_unused_columns=False,
)


# ============================================================================
# 6. Initialize SFTTrainer
# ============================================================================
def formatting_func(example):
    """Format each example for training."""
    return example["text"]

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset["train"],
    processing_class=tokenizer,         # Use processing_class, not tokenizer
    formatting_func=formatting_func,
)


# ============================================================================
# 7. Train!
# ============================================================================
print("\n" + "="*60)
print("Starting LoRA Fine-tuning")
print("="*60)

trainer.train()


# ============================================================================
# 8. Save the LoRA Adapter
# ============================================================================
print("\nSaving LoRA adapter...")
model.save_pretrained("./output_models/qwen3-0.6b-medical-lora/final")
tokenizer.save_pretrained("./output_models/qwen3-0.6b-medical-lora/final")

print("Training complete!")


# ============================================================================
# 9. Inference with LoRA
# ============================================================================
print("\n" + "="*60)
print("Testing the fine-tuned model")
print("="*60)

# For inference, you can load the adapter later:
"""
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./output_models/qwen3-0.6b-medical-lora/final")

# Merge for faster inference (optional)
model = model.merge_and_unload()
"""

# Quick test with current model
model.eval()
test_prompt = "<|user|>\nWhat are the symptoms of diabetes?\n<|assistant|>\n"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nTest prompt: {test_prompt}")
print(f"Response: {response}")

