"""
Shared model config for the serving stack.

MODEL_ID points at the user's own HF repo. Its name says "llama-3.1-8b-
instruct-nf4" but its actual config.json says architectures=["MistralForCausalLM"],
model_type="mistral", vocab_size=32768 (Llama 3.1's vocab is 128256) --
this is really the Mistral 7B-class model, not Llama 3.1 8B, most likely
because Llama gated access didn't clear before the Colab quantization
step. Not ambiguous (the weights settle it), so this points at what's
actually there rather than what the repo name claims.
"""
import torch
from transformers import BitsAndBytesConfig

MODEL_ID = "Siva-harish27/llama-3.1-8b-instruct-nf4"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
