from datasets import load_dataset

# Load the dataset
dataset = load_dataset("OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B")

# Access training split
train_data = dataset["train"]

print("Length of train_data:", len(train_data))

def format_conversation(example):
    conversation = ""
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        conversation += f"{role}: {content}\n"
    return {"text": conversation}

formatted_dataset = dataset.map(format_conversation)

print("Length of formatted_dataset:", len(formatted_dataset))
print("First sample role:", formatted_dataset["train"][0]["messages"][0]["role"])
print("First sample content first 100 characters:", formatted_dataset["train"][0]["messages"][0]["content"][:100])
print("First sample type:", type(formatted_dataset["train"][0]))
print("First sample keys:", formatted_dataset["train"][0].keys())
print('--------------------------------')


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Qwen uses eos_token as pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Detect device
if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32  # MPS doesn't support fp16 training well
elif torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"Using device: {device}, dtype: {dtype}")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=dtype,
    device_map="auto" if device != "mps" else None,  # device_map causes issues on MPS
)

if device == "mps":
    model = model.to(device)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=formatted_dataset["train"].column_names,
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM, not masked LM
)

training_args = TrainingArguments(
    output_dir="./output_models/qwen3-0.6b-medical",
    per_device_train_batch_size=1,          # Reduced for memory (especially on MPS)
    gradient_accumulation_steps=32,         # Effective batch = 1*32 = 32
    num_train_epochs=1,                     # Start with 1 for testing
    max_steps=100,                          # Limit steps for testing
    learning_rate=2e-5,
    warmup_ratio=0.1,                       # 10% warmup
    weight_decay=0.01,                      # Regularization
    save_steps=500,
    logging_steps=10,
    fp16=(device == "cuda"),                # Only use fp16 on CUDA
    optim="adamw_torch",                    # PyTorch AdamW optimizer
    gradient_checkpointing=True,            # Save memory
    dataloader_pin_memory=False,            # Required for MPS
    report_to="none",                       # Disable wandb/tensorboard
    save_total_limit=2,                     # Keep only 2 checkpoints
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

trainer.train()