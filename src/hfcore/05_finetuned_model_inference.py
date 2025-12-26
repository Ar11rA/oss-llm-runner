"""
Inference: Compare Base Model vs Full FT vs LoRA FT

Uses medical prompts from OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B format
Saves results to CSV for analysis
"""

import os
import csv
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional: LoRA support
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  Install peft for LoRA support: pip install peft")


# ============================================================================
# Configuration
# ============================================================================
BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
FULL_FT_PATH = "./output_models/qwen3-0.6b-medical/checkpoint-100"
LORA_FT_PATH = "./output_models/qwen3-0.6b-medical-lora/final"

# Output directory
OUTPUT_DIR = "./inference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

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
# Helper: Check if model path exists
# ============================================================================
def model_exists(path):
    """Check if a model path exists and contains model files."""
    if not os.path.exists(path):
        return False
    return os.path.exists(os.path.join(path, "config.json")) or \
           os.path.exists(os.path.join(path, "adapter_config.json"))


# ============================================================================
# Medical Prompts (from OpenMed dataset format)
# ============================================================================
SYSTEM_INSTRUCTION = """You are a helpful medical AI assistant. Provide accurate, 
evidence-based medical information. When discussing treatments or diagnoses, 
be clear about uncertainties and recommend consulting healthcare professionals."""

MEDICAL_PROMPTS = [
    {
        "user": "HIV infection and treatment in children and adolescents",
        "context": "Explain the key considerations for managing HIV in pediatric patients."
    },
    {
        "user": "What are the early warning signs of diabetic ketoacidosis?",
        "context": "A patient with Type 1 diabetes presents with nausea and fatigue."
    },
    {
        "user": "Explain the mechanism of action of beta-blockers in heart failure.",
        "context": "Patient is being started on carvedilol for chronic heart failure."
    },
    {
        "user": "What is the differential diagnosis for acute chest pain?",
        "context": "45-year-old male presents with sudden onset chest pain radiating to left arm."
    },
    {
        "user": "How do you manage hypertensive emergency?",
        "context": "Patient presents with BP 220/130 and signs of end-organ damage."
    },
]


def format_prompt(user_message: str, context: str = None, include_system: bool = True) -> str:
    """Format prompt like the OpenMed dataset structure."""
    prompt = ""
    
    if include_system:
        prompt += f"<|system|>\n{SYSTEM_INSTRUCTION}\n"
    
    user_content = user_message
    if context:
        user_content = f"{context}\n\nQuestion: {user_message}"
    
    prompt += f"<|user|>\n{user_content}\n<|assistant|>\n"
    
    return prompt


# ============================================================================
# Load Models
# ============================================================================
models = {}  # Dict of {name: (model, tokenizer)}

# 1. Base Model
print(f"\n🔵 Loading BASE model: {BASE_MODEL_ID}")
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=dtype,
    device_map="auto" if device != "mps" else None,
)
if device == "mps":
    base_model = base_model.to(device)
base_model.eval()
models["Base"] = (base_model, base_tokenizer)
print("   ✅ Base model loaded!")

# 2. Full Fine-tuned Model
if model_exists(FULL_FT_PATH):
    print(f"\n🟢 Loading FULL FT model: {FULL_FT_PATH}")
    ft_tokenizer = AutoTokenizer.from_pretrained(FULL_FT_PATH)
    ft_model = AutoModelForCausalLM.from_pretrained(
        FULL_FT_PATH,
        dtype=dtype,
        device_map="auto" if device != "mps" else None,
    )
    if device == "mps":
        ft_model = ft_model.to(device)
    ft_model.eval()
    models["Full-FT"] = (ft_model, ft_tokenizer)
    print("   ✅ Full FT model loaded!")
else:
    print(f"\n⚠️  Full FT model not found at: {FULL_FT_PATH}")

# 3. LoRA Fine-tuned Model
if PEFT_AVAILABLE and model_exists(LORA_FT_PATH):
    print(f"\n🟣 Loading LoRA FT model: {LORA_FT_PATH}")
    # LoRA requires loading base model first, then adapter
    lora_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=dtype,
        device_map="auto" if device != "mps" else None,
    )
    if device == "mps":
        lora_base = lora_base.to(device)
    
    lora_model = PeftModel.from_pretrained(lora_base, LORA_FT_PATH)
    lora_model.eval()
    models["LoRA-FT"] = (lora_model, base_tokenizer)  # LoRA uses base tokenizer
    print("   ✅ LoRA FT model loaded!")
else:
    if not PEFT_AVAILABLE:
        print("\n⚠️  LoRA model skipped (peft not installed)")
    elif not model_exists(LORA_FT_PATH):
        print(f"\n⚠️  LoRA FT model not found at: {LORA_FT_PATH}")

print(f"\n📊 Models loaded: {list(models.keys())}")


# ============================================================================
# Generate Function
# ============================================================================
def generate(model, tokenizer, prompt, max_new_tokens=500):
    """Generate response from model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# ============================================================================
# Run Inference and Collect Results
# ============================================================================
print("\n" + "=" * 80)
print("MEDICAL REASONING COMPARISON")
print(f"Models: {list(models.keys())}")
print("=" * 80)

# Collect all results for CSV
all_results = []

for i, prompt_data in enumerate(MEDICAL_PROMPTS):
    formatted_prompt = format_prompt(
        user_message=prompt_data["user"],
        context=prompt_data.get("context"),
        include_system=True
    )
    
    print(f"\n{'='*80}")
    print(f"EXAMPLE {i+1}")
    print(f"{'='*80}")
    
    print(f"\n📝 USER QUESTION:\n{prompt_data['user']}")
    if prompt_data.get("context"):
        print(f"\n📋 CONTEXT:\n{prompt_data['context']}")
    
    # Result row for CSV
    result_row = {
        "timestamp": TIMESTAMP,
        "example_id": i + 1,
        "user_question": prompt_data["user"],
        "context": prompt_data.get("context", ""),
    }
    
    # Generate from each model
    for model_name, (model, tokenizer) in models.items():
        emoji = "🔵" if model_name == "Base" else "🟢" if model_name == "Full-FT" else "🟣"
        print(f"\n{emoji} {model_name} RESPONSE:")
        print("-" * 40)
        
        response = generate(model, tokenizer, formatted_prompt)
        result_row[f"{model_name}_response"] = response
        
        # Print truncated for console
        print(response[:600] + "..." if len(response) > 600 else response)
    
    all_results.append(result_row)


# ============================================================================
# Save Results to CSV
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

csv_path = os.path.join(OUTPUT_DIR, f"inference_{TIMESTAMP}.csv")

# Get all column names dynamically
fieldnames = ["timestamp", "example_id", "user_question", "context"]
for model_name in models.keys():
    fieldnames.append(f"{model_name}_response")

with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results)

print(f"\n✅ Results saved to: {csv_path}")
print(f"   - {len(all_results)} examples")
print(f"   - {len(models)} models compared")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("INFERENCE COMPLETE")
print("=" * 80)
print(f"""
📁 Output: {csv_path}
📊 Models: {', '.join(models.keys())}
📝 Examples: {len(MEDICAL_PROMPTS)}

Open the CSV in Excel/Sheets to compare responses side-by-side!
""")


# ============================================================================
# Interactive Mode (Optional)
# ============================================================================
def interactive_chat():
    """Run interactive chat with all models."""
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE - Type 'quit' to exit")
    print("=" * 80)
    
    while True:
        user_input = input("\n📝 Your question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        formatted = format_prompt(user_input, include_system=True)
        
        for model_name, (model, tokenizer) in models.items():
            emoji = "🔵" if model_name == "Base" else "🟢" if model_name == "Full-FT" else "🟣"
            print(f"\n{emoji} {model_name}:")
            print(generate(model, tokenizer, formatted, max_new_tokens=200))


# Uncomment to enable interactive mode:
# interactive_chat()
