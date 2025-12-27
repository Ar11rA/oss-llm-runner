"""
FastAPI Inference Server for MLX-LM

Run:
    uv run uvicorn src.inferencing.03_server:app --reload --port 8000

Or:
    uv run python src/inferencing/03_server.py

Test:
    curl -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "What are the symptoms of diabetes?"}'
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from mlx_lm import load, generate

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = "./output_models/qwen3-0.6b-medical-merged"

# ============================================================================
# Load Model (once at startup)
# ============================================================================
print(f"Loading model from {MODEL_PATH}...")
model, tokenizer = load(MODEL_PATH)
print("✅ Model loaded!")

# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(title="MLX-LM Inference Server")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    prompt: str
    response: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    response = generate(
        model,
        tokenizer,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
    )
    return GenerateResponse(prompt=request.prompt, response=response)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

