"""
OpenAI-compatible chat completions server for self-hosted inference (e.g. AMD MI350X).

Usage:
  export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct   # or Qwen/QwQ-32B
  export HF_TOKEN=...   # if model is gated
  uv sync --extra server
  uv run uvicorn scripts.local_inference_server:app --host 0.0.0.0 --port 8000

Then run KernelBench with server_type=local model_name=local.
"""
from __future__ import annotations

import os

def _create_app():
    import torch
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    model_name = os.environ.get("MODEL_NAME")
    if not model_name:
        raise RuntimeError("Set MODEL_NAME (e.g. meta-llama/Llama-3.1-8B-Instruct or Qwen/QwQ-32B)")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")

    fastapi_app = FastAPI(title="KernelBench local inference")

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str = "default"
        messages: list[ChatMessage]
        max_tokens: int = 2048
        temperature: float = 0.3
        top_p: float = 1.0
        frequency_penalty: float = 0.0

    @fastapi_app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        text = tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in request.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature if request.temperature > 0 else 1e-6,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        reply = tokenizer.decode(gen[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return JSONResponse(
            content={
                "id": "local",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": inputs.input_ids.numel(), "completion_tokens": 0, "total_tokens": 0},
            }
        )

    return fastapi_app


# Uvicorn expects `app`. Load model at startup if MODEL_NAME is set.
if os.environ.get("MODEL_NAME"):
    app = _create_app()
else:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    app = FastAPI(title="KernelBench local inference (stub)")
    @app.post("/v1/chat/completions")
    def _err():
        return JSONResponse(status_code=503, content={"error": "Set MODEL_NAME and restart the server."})
