from __future__ import annotations

import os
from pydantic import BaseModel
from peft import PeftModel


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


def _create_app():
    import torch
    from fastapi import FastAPI, Body
    from fastapi.responses import JSONResponse

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
    adapter_dir = os.environ.get("ADAPTER_DIR")
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    print("Model loaded.")

    fastapi_app = FastAPI(title="KernelBench local inference")

    @fastapi_app.post("/v1/chat/completions")
    def chat_completions(body: ChatCompletionRequest = Body(...)):
        text = tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in body.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=body.max_tokens,
            temperature=body.temperature if body.temperature > 0 else 1e-6,
            top_p=body.top_p,
            do_sample=body.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        reply = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return JSONResponse(
            content={
                "id": "local",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": int(inputs.input_ids.numel()),
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

    return fastapi_app


if os.environ.get("MODEL_NAME"):
    app = _create_app()
else:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="KernelBench local inference (stub)")

    @app.post("/v1/chat/completions")
    def _err():
        return JSONResponse(status_code=503, content={"error": "Set MODEL_NAME and restart the server."})