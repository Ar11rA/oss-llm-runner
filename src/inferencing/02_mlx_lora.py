"""
MLX-LM Inference: Base vs Merged Model Comparison

Compares outputs from base model vs fine-tuned merged model.
"""

from mlx_lm import load, generate

# ============================================================================
# Configuration
# ============================================================================
BASE_MODEL = "Qwen/Qwen3-0.6B"
MERGED_MODEL = "./output_models/qwen3-0.6b-medical-merged"

PROMPTS = [
    "What are the symptoms of diabetes?",
    "How is hypertension typically treated?",
    "Explain the mechanism of beta-blockers.",
]

# ============================================================================
# Load Models
# ============================================================================
print("Loading BASE model...")
base_model, base_tokenizer = load(BASE_MODEL)
print("✅ Base model loaded!\n")

print("Loading MERGED model...")
merged_model, merged_tokenizer = load(MERGED_MODEL)
print("✅ Merged model loaded!\n")

# ============================================================================
# Compare Outputs
# ============================================================================
for i, prompt in enumerate(PROMPTS):
    print(f"\n{'='*60}")
    print(f"PROMPT {i+1}: {prompt}")
    print("=" * 60)
    
    print("\n🔵 BASE MODEL:")
    print("-" * 40)
    base_response = generate(base_model, base_tokenizer, prompt=prompt, max_tokens=200)
    print(base_response)
    
    print("\n🟣 MERGED MODEL:")
    print("-" * 40)
    merged_response = generate(merged_model, merged_tokenizer, prompt=prompt, max_tokens=200)
    print(merged_response)

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
