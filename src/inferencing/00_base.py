from mlx_lm import load, generate

model, tokenizer = load("Qwen/Qwen3-0.6B")
response = generate(model, tokenizer, prompt="What is diabetes?", max_tokens=10000)

print(response)