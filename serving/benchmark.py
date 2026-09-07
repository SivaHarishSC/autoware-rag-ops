"""Load the quantized model on the real 4050 and run a realistic RAG-shaped
prompt through generation, measuring what actually matters: peak VRAM
during generation (not just after load) and tokens/sec."""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_config import BNB_CONFIG, MODEL_ID
from rag_prompt import build_rag_prompt

MAX_NEW_TOKENS = 256


def gb(x):
    return x / (1024 ** 3)


def main():
    print(f"loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        dtype=torch.float16,
    )

    torch.cuda.synchronize()
    print(f"\nafter load: allocated={gb(torch.cuda.memory_allocated()):.3f} GB, reserved={gb(torch.cuda.memory_reserved()):.3f} GB")

    messages = build_rag_prompt()
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    input_ids = encoded["input_ids"].to(model.device)
    prompt_tokens = input_ids.shape[1]
    print(f"realistic RAG prompt: {prompt_tokens} tokens")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    try:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.time() - start
        new_tokens = output.shape[1] - prompt_tokens
        peak_allocated = gb(torch.cuda.max_memory_allocated())
        peak_reserved = gb(torch.cuda.max_memory_reserved())

        print(f"\nSTATUS: success")
        print(f"generated {new_tokens} new tokens in {elapsed:.2f}s -> {new_tokens / elapsed:.2f} tok/s")
        print(f"peak VRAM during generation: allocated={peak_allocated:.3f} GB, reserved={peak_reserved:.3f} GB")
        print(f"\n--- generated text ---")
        print(tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True))
    except torch.cuda.OutOfMemoryError as e:
        print(f"\nSTATUS: OOM")
        print(str(e))


if __name__ == "__main__":
    main()
