import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print(torch.backends.mps.is_available())  # True on M1/M2/M3
print(torch.backends.mps.is_built()) 

model_id = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a helpful assistant. Give brief, direct answers."},
    {"role": "user", "content": "What is the capital of France?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
encoded = tokenizer(text, return_tensors="pt")

print("tokenizer special tokens map:", tokenizer.special_tokens_map)

print("encoded:", encoded)
print("encoded shape:", encoded["input_ids"].shape)
print("encoded input_ids:", encoded["input_ids"])

decoded = tokenizer.decode(encoded["input_ids"][0])
print("decoded:", decoded)

# Use CPU to avoid MPS issues with this model
# MPS has compatibility issues with some Qwen models
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using: {device}")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
)
model = model.to(device)
print("model:", model)

# Move inputs to same device as model
input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)

# Generate text
output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=5000,
    pad_token_id=tokenizer.eos_token_id,
    
    # Stop repetition
    repetition_penalty=1.2,               # Penalize repeating tokens
    no_repeat_ngram_size=3,               # Don't repeat 3-grams
    
    # Or use sampling for variety
    do_sample=False,
    temperature=0.01,
)

print("--------------------------------")
print("output:", output)
print("output shape:", output.shape)

# Decode output
decoded = tokenizer.decode(output[0], skip_special_tokens=True)
print("decoded output:", decoded)