# ===== MUST BE AT THE VERY TOP OF YOUR NOTEBOOK =====
import os
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams

import gc
import torch

# del model
gc.collect()
torch.cuda.empty_cache()

# Now recreate
llm = LLM(
    model="mistralai/Devstral-Small-2-24B-Instruct-2512",
    enforce_eager=True,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    dtype="float16",           # Load in FP16
    quantization="fp8",        # Quantize to FP8 for inference
)

messages = [
    {
        "role": "system",
        "content": "You are a Python code generator. You MUST output ONLY raw Python code. Any text that is not valid Python syntax is FORBIDDEN. Do not use markdown. Start immediately with 'def' or 'import'."
    },
    {
        "role": "user",
        "content": "Code to check if number is prime"
    }
]

params = SamplingParams(
    temperature=0.2, 
    max_tokens=2500,
)

# Use chat() instead of generate()
outputs = llm.chat([messages], params)
print(outputs[0].outputs[0].text)

messages = [
    {
        "role": "system",
        "content": "You are a Python code generator. You MUST output ONLY raw Python code. Any text that is not valid Python syntax is FORBIDDEN. Do not use markdown. Start immediately with 'def' or 'import'."
    },
    {
        "role": "user",
        "content": "Code to check if number is prime"
    }
]

params = SamplingParams(
    temperature=0.05, 
    max_tokens=5000,
)

# Use chat() instead of generate()
outputs = llm.chat([messages], params)
print(outputs[0].outputs[0].text)