# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-33B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-33B")
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

from vllm import LLM, SamplingParams

import gc
import torch

del model
gc.collect()
torch.cuda.empty_cache()

# Now recreate
llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-33B",
    enforce_eager=True,
    gpu_memory_utilization=0.9,
)
params = SamplingParams(temperature=0.7, max_tokens=2000)
outputs = llm.generate(["What are the symptoms of diabetes?"], params)
print(outputs[0].outputs[0].text)