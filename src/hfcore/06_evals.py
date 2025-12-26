"""
Evaluation for Fine-tuned Medical Reasoning Model

Compares 3 models:
1. Base model (Qwen3-0.6B)
2. Full fine-tuned model
3. LoRA fine-tuned model

Metrics:
1. Loss/Perplexity - on held-out test set
2. ROUGE-L - lexical overlap with ground truth
3. BERTScore - semantic similarity
4. Generation comparison - side-by-side outputs
"""

import os
import csv
import torch
import numpy as np
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Optional: LoRA support
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  Install peft for LoRA support: pip install peft")

# Optional metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  Install rouge-score for ROUGE metrics: pip install rouge-score")

try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("⚠️  Install bert-score for BERTScore: pip install bert-score")


# ============================================================================
# CONFIG
# ============================================================================
BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
FULL_FT_PATH = "./output_models/qwen3-0.6b-medical/checkpoint-100"
LORA_FT_PATH = "./output_models/qwen3-0.6b-medical-lora/final"  # or checkpoint-100

MAX_LEN = 512
EVAL_SAMPLES = 20  # Number of samples for evaluation
GEN_SAMPLES = 3    # Number of samples for generation comparison

# Output directory for eval results
EVAL_OUTPUT_DIR = "./eval_results"
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

# Timestamp for this run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Device detection
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
# HELPER: Check if path exists and has model files
# ============================================================================
def model_exists(path):
    """Check if a model path exists and contains model files."""
    if not os.path.exists(path):
        return False
    # Check for config.json (present in all HF models)
    return os.path.exists(os.path.join(path, "config.json")) or \
           os.path.exists(os.path.join(path, "adapter_config.json"))


# ============================================================================
# LOAD BASE MODEL & TOKENIZER
# ============================================================================
print("\n" + "="*70)
print("LOADING MODELS")
print("="*70)

print(f"\n🔵 Loading BASE model: {BASE_MODEL_ID}")
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=dtype,
    device_map="auto" if device != "mps" else None,
)
if device == "mps":
    base_model = base_model.to(device)
base_model.eval()
print("   ✅ Base model loaded")


# ============================================================================
# LOAD FULL FINE-TUNED MODEL
# ============================================================================
full_ft_model = None
full_ft_tokenizer = None

if model_exists(FULL_FT_PATH):
    print(f"\n🟢 Loading FULL FT model: {FULL_FT_PATH}")
    full_ft_tokenizer = AutoTokenizer.from_pretrained(FULL_FT_PATH)
    if full_ft_tokenizer.pad_token is None:
        full_ft_tokenizer.pad_token = full_ft_tokenizer.eos_token
    
    full_ft_model = AutoModelForCausalLM.from_pretrained(
        FULL_FT_PATH,
        dtype=dtype,
        device_map="auto" if device != "mps" else None,
    )
    if device == "mps":
        full_ft_model = full_ft_model.to(device)
    full_ft_model.eval()
    print("   ✅ Full FT model loaded")
else:
    print(f"\n⚠️  FULL FT model not found at: {FULL_FT_PATH}")


# ============================================================================
# LOAD LORA FINE-TUNED MODEL
# ============================================================================
lora_ft_model = None

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
    
    lora_ft_model = PeftModel.from_pretrained(lora_base, LORA_FT_PATH)
    lora_ft_model.eval()
    print("   ✅ LoRA FT model loaded")
else:
    if not PEFT_AVAILABLE:
        print("\n⚠️  LoRA model skipped (peft not installed)")
    else:
        print(f"\n⚠️  LoRA FT model not found at: {LORA_FT_PATH}")


# Build list of models to evaluate
models = [
    ("Base", base_model, base_tokenizer),
]
if full_ft_model is not None:
    models.append(("Full-FT", full_ft_model, full_ft_tokenizer))
if lora_ft_model is not None:
    models.append(("LoRA-FT", lora_ft_model, base_tokenizer))  # LoRA uses base tokenizer

print(f"\n📊 Models to evaluate: {[m[0] for m in models]}")


# ============================================================================
# LOAD DATASET
# ============================================================================
print("\nLoading dataset...")
dataset = load_dataset("OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B")

# Use last N samples as held-out test set
total_samples = len(dataset["train"])
eval_ds = dataset["train"].select(range(total_samples - EVAL_SAMPLES, total_samples))
print(f"Evaluation set: {len(eval_ds)} samples")


# ============================================================================
# EXTRACT PROMPT AND GROUND TRUTH
# ============================================================================
def extract_prompt_and_answer(example):
    """Extract user prompt and assistant answer from messages format."""
    messages = example["messages"]
    prompt = None
    answer = None
    
    for msg in messages:
        if msg["role"] == "user":
            prompt = msg["content"]
        elif msg["role"] == "assistant":
            answer = msg["content"]
    
    return {"prompt": prompt, "ground_truth": answer}


eval_ds = eval_ds.map(extract_prompt_and_answer)


# ============================================================================
# GENERATION FUNCTION
# ============================================================================
@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=150):
    """Generate response from model."""
    enc = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    ).to(model.device)

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Greedy for reproducibility
        pad_token_id=tok.pad_token_id,
    )
    
    response = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# ============================================================================
# METRIC 1: PERPLEXITY
# ============================================================================
@torch.no_grad()
def compute_perplexity(model, tok, texts):
    """Compute average perplexity on texts."""
    total_loss = 0
    total_tokens = 0
    
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        
        out = model(**enc, labels=enc["input_ids"])
        
        num_tokens = enc["attention_mask"].sum().item()
        total_loss += out.loss.item() * num_tokens
        total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity, avg_loss


# ============================================================================
# METRIC 2: ROUGE-L
# ============================================================================
def compute_rouge(predictions, references):
    """Compute ROUGE-L scores."""
    if not ROUGE_AVAILABLE:
        return None
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    
    for pred, ref in zip(predictions, references):
        if pred and ref:
            score = scorer.score(ref, pred)
            scores.append(score['rougeL'].fmeasure)
    
    return np.mean(scores) if scores else 0.0


# ============================================================================
# METRIC 3: BERTSCORE
# ============================================================================
def compute_bertscore(predictions, references):
    """Compute BERTScore (semantic similarity)."""
    if not BERTSCORE_AVAILABLE:
        return None
    
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
    if not valid_pairs:
        return 0.0
    
    preds, refs = zip(*valid_pairs)
    
    P, R, F1 = bert_score_fn(
        list(preds), 
        list(refs), 
        lang="en", 
        model_type="distilbert-base-uncased",
        verbose=False
    )
    
    return F1.mean().item()


# ============================================================================
# CSV EXPORT FUNCTIONS
# ============================================================================
def save_metrics_csv(ppl_results, rouge_results, bert_results):
    """Save metric summary to CSV."""
    filepath = os.path.join(EVAL_OUTPUT_DIR, f"metrics_{TIMESTAMP}.csv")
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'model', 'perplexity', 'rouge_l', 'bertscore_f1'])
        
        for model_name in ppl_results.keys():
            ppl = ppl_results.get(model_name, '')
            rouge = rouge_results.get(model_name, '')
            bert = bert_results.get(model_name, '')
            writer.writerow([TIMESTAMP, model_name, ppl, rouge, bert])
    
    print(f"\n📁 Metrics saved to: {filepath}")
    return filepath


def save_generations_csv(prompts, references, all_responses):
    """Save generated outputs to CSV for qualitative review."""
    filepath = os.path.join(EVAL_OUTPUT_DIR, f"generations_{TIMESTAMP}.csv")
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header: timestamp, example_id, prompt, ground_truth, model1_response, model2_response, ...
        header = ['timestamp', 'example_id', 'prompt', 'ground_truth'] + list(all_responses.keys())
        writer.writerow(header)
        
        for i in range(len(prompts)):
            row = [
                TIMESTAMP,
                i + 1,
                prompts[i],
                references[i],
            ]
            # Add each model's response
            for model_name in all_responses.keys():
                row.append(all_responses[model_name][i])
            writer.writerow(row)
    
    print(f"📁 Generations saved to: {filepath}")
    return filepath


# ============================================================================
# RUN EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

ground_truths = [ex["ground_truth"] for ex in eval_ds if ex["ground_truth"]]

# ============================================================================
# 1. PERPLEXITY
# ============================================================================
print("\n📊 METRIC 1: PERPLEXITY (lower is better)")
print("-" * 60)
print(f"{'Model':<15} {'Perplexity':>12} {'Loss':>10}")
print("-" * 60)

ppl_results = {}
for name, model, tok in tqdm(models, desc="Computing perplexity"):
    ppl, loss = compute_perplexity(model, tok, ground_truths)
    ppl_results[name] = ppl
    print(f"{name:<15} {ppl:>12.2f} {loss:>10.4f}")


# ============================================================================
# 2. GENERATE RESPONSES
# ============================================================================
print("\n📊 GENERATING RESPONSES...")
print("-" * 60)

gen_samples_ds = eval_ds.select(range(min(GEN_SAMPLES, len(eval_ds))))
references = [ex["ground_truth"] or "" for ex in gen_samples_ds]
prompts = [ex["prompt"] for ex in gen_samples_ds]

all_responses = {}
for name, model, tok in models:
    responses = []
    for prompt in tqdm(prompts, desc=f"Generating ({name})"):
        if prompt:
            responses.append(generate(model, tok, prompt))
        else:
            responses.append("")
    all_responses[name] = responses


# ============================================================================
# 3. ROUGE-L
# ============================================================================
print("\n📊 METRIC 2: ROUGE-L (higher is better)")
print("-" * 60)

rouge_results = {}
if ROUGE_AVAILABLE:
    print(f"{'Model':<15} {'ROUGE-L':>12}")
    print("-" * 60)
    for name in all_responses:
        rouge = compute_rouge(all_responses[name], references)
        rouge_results[name] = rouge
        print(f"{name:<15} {rouge:>12.4f}")
else:
    print("⚠️  ROUGE not available. Install: pip install rouge-score")


# ============================================================================
# 4. BERTSCORE
# ============================================================================
print("\n📊 METRIC 3: BERTSCORE F1 (higher is better)")
print("-" * 60)

bert_results = {}
if BERTSCORE_AVAILABLE:
    print(f"{'Model':<15} {'BERTScore':>12}")
    print("-" * 60)
    for name in all_responses:
        bert = compute_bertscore(all_responses[name], references)
        bert_results[name] = bert
        print(f"{name:<15} {bert:>12.4f}")
else:
    print("⚠️  BERTScore not available. Install: pip install bert-score")


# ============================================================================
# 5. QUALITATIVE COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("QUALITATIVE COMPARISON")
print("=" * 70)

for i in range(len(prompts)):
    print(f"\n{'='*70}")
    print(f"EXAMPLE {i+1}")
    print(f"{'='*70}")
    
    prompt = prompts[i]
    ref = references[i]
    
    print(f"\n📝 PROMPT:\n{prompt[:400]}..." if len(prompt) > 400 else f"\n📝 PROMPT:\n{prompt}")
    print(f"\n✅ GROUND TRUTH:\n{ref[:400]}..." if len(ref) > 400 else f"\n✅ GROUND TRUTH:\n{ref}")
    
    for name in all_responses:
        resp = all_responses[name][i]
        emoji = "🔵" if name == "Base" else "🟢" if name == "Full-FT" else "🟣"
        print(f"\n{emoji} {name}:\n{resp[:400]}..." if len(resp) > 400 else f"\n{emoji} {name}:\n{resp}")


# ============================================================================
# 6. SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\n📈 PERPLEXITY COMPARISON:")
base_ppl = ppl_results.get("Base", 0)
for name, ppl in ppl_results.items():
    if name != "Base" and base_ppl > 0:
        improvement = (base_ppl - ppl) / base_ppl * 100
        print(f"   {name}: {improvement:+.1f}% vs Base")

print("""
📈 INTERPRETATION GUIDE:
  - Perplexity: FT models should be LOWER (better prediction)
  - ROUGE-L: FT models should be HIGHER (more overlap with ground truth)
  - BERTScore: FT models should be HIGHER (better semantic similarity)
  
⚠️  RED FLAGS:
  - FT perplexity ~0 → Overfitting/memorization
  - FT outputs are verbatim copies → Overfitting
  - Base model outputs look better → Don't ship!
  
✅ EXPECTED RESULTS:
  - Full-FT: Best perplexity (saw all training data)
  - LoRA-FT: Good perplexity with fewer params
  - Both should show clearer medical reasoning than base
""")


# ============================================================================
# 7. SAVE RESULTS TO CSV
# ============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save metrics summary
save_metrics_csv(ppl_results, rouge_results, bert_results)

# Save generation outputs
save_generations_csv(prompts, references, all_responses)

print(f"\n✅ All results saved to: {EVAL_OUTPUT_DIR}/")
print(f"   - metrics_{TIMESTAMP}.csv")
print(f"   - generations_{TIMESTAMP}.csv")
