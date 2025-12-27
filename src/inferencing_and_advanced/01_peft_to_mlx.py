from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model and merge LoRA
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = PeftModel.from_pretrained(base_model, "./output_models/qwen3-0.6b-medical-lora-mlxlm/final")
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./output_models/qwen3-0.6b-medical-merged")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer.save_pretrained("./output_models/qwen3-0.6b-medical-merged")