"""
Inference: Compare Base Model vs Fine-tuned Model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# Configuration
# ============================================================================
BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
FINETUNED_PATH = "./output_models/qwen3-0.6b-medical/checkpoint-100"

# Detect device
if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
elif torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"Using device: {device}, dtype: {dtype}")


# ============================================================================
# Load Base Model
# ============================================================================
print(f"\nLoading BASE model: {BASE_MODEL_ID}")
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=dtype,
    device_map="auto" if device != "mps" else None,
)
if device == "mps":
    base_model = base_model.to(device)
base_model.eval()
print("Base model loaded!")


# ============================================================================
# Load Fine-tuned Model
# ============================================================================
print(f"\nLoading FINE-TUNED model: {FINETUNED_PATH}")
ft_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_PATH)
ft_model = AutoModelForCausalLM.from_pretrained(
    FINETUNED_PATH,
    dtype=dtype,
    device_map="auto" if device != "mps" else None,
)
if device == "mps":
    ft_model = ft_model.to(device)
ft_model.eval()
print("Fine-tuned model loaded!")


# ============================================================================
# Generate Function
# ============================================================================
def generate(model, tokenizer, prompt, max_new_tokens=1000):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================================================
# Compare Outputs
# ============================================================================
prompt = "HIV infection and treatment in children and adolescents"

print("\n" + "=" * 60)
print(f"PROMPT: {prompt}")
print("=" * 60)

print("\n--- BASE MODEL ---")
base_response = generate(base_model, base_tokenizer, prompt)
print(base_response)

print("\n--- FINE-TUNED MODEL ---")
ft_response = generate(ft_model, ft_tokenizer, prompt)
print(ft_response)

print("\n" + "=" * 60)
