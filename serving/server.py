"""Minimal OpenAI-compatible server for the locally quantized model.
Model loads ONCE at startup (module-level, executed on import/process
start), not per-request."""
import time

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_config import BNB_CONFIG, MODEL_ID

app = FastAPI()

print(f"loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=BNB_CONFIG,
    device_map="auto",
    dtype=torch.float16,
)
DEVICE = str(model.device)
print(f"model loaded on {DEVICE}")


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = 256


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model_id": MODEL_ID, "device": DEVICE}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    messages = [m.model_dump() for m in req.messages]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    input_ids = encoded["input_ids"].to(model.device)
    prompt_tokens = input_ids.shape[1]

    start = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=req.max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    completion_tokens = output.shape[1] - prompt_tokens
    text = tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True)

    print(f"[request] prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} latency={elapsed:.2f}s ({completion_tokens / elapsed:.2f} tok/s)")

    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
