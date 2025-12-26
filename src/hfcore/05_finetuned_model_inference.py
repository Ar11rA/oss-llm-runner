"""
Inference: Compare Base Model vs Full FT vs LoRA FT

Loads examples from OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B dataset
Saves results to CSV including ground truth for comparison
"""

import os
import csv
import torch
from datetime import datetime
from datasets import load_dataset
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

# Number of examples to evaluate
NUM_EXAMPLES = 10

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
# Load Dataset from HuggingFace
# ============================================================================
print("\n📦 Loading OpenMed dataset...")
dataset = load_dataset("OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B")

# Select examples (from end of dataset to avoid training overlap)
total_samples = len(dataset["train"])
start_idx = total_samples - NUM_EXAMPLES
eval_examples = dataset["train"].select(range(start_idx, total_samples))

print(f"   ✅ Loaded {NUM_EXAMPLES} examples (indices {start_idx} to {total_samples-1})")


# ============================================================================
# Extract User Question and Ground Truth from Dataset
# ============================================================================
def extract_from_messages(example):
    """Extract user prompt and assistant answer from messages format."""
    messages = example["messages"]
    user_content = None
    assistant_content = None
    
    for msg in messages:
        if msg["role"] == "user":
            user_content = msg["content"]
        elif msg["role"] == "assistant":
            assistant_content = msg["content"]
    
    return {
        "user_question": user_content,
        "ground_truth": assistant_content
    }


# Process examples
processed_examples = [extract_from_messages(ex) for ex in eval_examples]
print(f"   ✅ Processed {len(processed_examples)} examples")


# ============================================================================
# Format Prompt for Model
# ============================================================================
SYSTEM_INSTRUCTION = """You are a helpful medical AI assistant. Provide accurate, 
evidence-based medical information. When discussing treatments or diagnoses, 
be clear about uncertainties and recommend consulting healthcare professionals."""


def format_prompt(user_message: str) -> str:
    """Format prompt with system instruction and user message."""
    return f"<|system|>\n{SYSTEM_INSTRUCTION}\n<|user|>\n{user_message}\n<|assistant|>\n"


# ============================================================================
# Load Models
# ============================================================================
models = {}

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
    lora_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=dtype,
        device_map="auto" if device != "mps" else None,
    )
    if device == "mps":
        lora_base = lora_base.to(device)
    
    lora_model = PeftModel.from_pretrained(lora_base, LORA_FT_PATH)
    lora_model.eval()
    models["LoRA-FT"] = (lora_model, base_tokenizer)
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
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
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
print(f"Examples: {NUM_EXAMPLES} from OpenMed dataset")
print("=" * 80)

all_results = []

for i, example in enumerate(processed_examples):
    user_question = example["user_question"]
    ground_truth = example["ground_truth"]
    
    if not user_question:
        continue
    
    formatted_prompt = format_prompt(user_question)
    
    print(f"\n{'='*80}")
    print(f"EXAMPLE {i+1}/{NUM_EXAMPLES}")
    print(f"{'='*80}")
    
    # Truncate question for display
    display_question = user_question[:300] + "..." if len(user_question) > 300 else user_question
    print(f"\n📝 USER QUESTION:\n{display_question}")
    
    # Result row for CSV
    result_row = {
        "timestamp": TIMESTAMP,
        "example_id": i + 1,
        "user_question": user_question,
        "ground_truth": ground_truth,
    }
    
    # Generate from each model
    for model_name, (model, tokenizer) in models.items():
        emoji = "🔵" if model_name == "Base" else "🟢" if model_name == "Full-FT" else "🟣"
        print(f"\n{emoji} {model_name} RESPONSE:")
        print("-" * 40)
        
        response = generate(model, tokenizer, formatted_prompt)
        result_row[f"{model_name}_response"] = response
        
        # Print truncated for console
        display_response = response[:400] + "..." if len(response) > 400 else response
        print(display_response)
    
    # Show ground truth
    display_gt = ground_truth[:400] + "..." if len(ground_truth) > 400 else ground_truth
    print("\n✅ GROUND TRUTH:")
    print("-" * 40)
    print(display_gt)
    
    all_results.append(result_row)


# ============================================================================
# Save Results to CSV
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

csv_path = os.path.join(OUTPUT_DIR, f"inference_{TIMESTAMP}.csv")

# Build fieldnames dynamically
fieldnames = ["timestamp", "example_id", "user_question", "ground_truth"]
for model_name in models.keys():
    fieldnames.append(f"{model_name}_response")

with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results)

print(f"\n✅ Results saved to: {csv_path}")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("INFERENCE COMPLETE")
print("=" * 80)
print(f"""
📁 Output: {csv_path}
📊 Models: {', '.join(models.keys())}
📝 Examples: {len(all_results)} from OpenMed dataset

CSV Columns:
  - timestamp
  - example_id
  - user_question (from dataset)
  - ground_truth (from dataset)
  - Base_response
  - Full-FT_response (if available)
  - LoRA-FT_response (if available)

Open the CSV to compare model outputs against ground truth!
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
        
        formatted = format_prompt(user_input)
        
        for model_name, (model, tokenizer) in models.items():
            emoji = "🔵" if model_name == "Base" else "🟢" if model_name == "Full-FT" else "🟣"
            print(f"\n{emoji} {model_name}:")
            print(generate(model, tokenizer, formatted, max_new_tokens=200))


# Uncomment to enable interactive mode:
# interactive_chat()
