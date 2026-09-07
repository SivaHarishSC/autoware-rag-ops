"""Builds a realistic RAG-style test prompt from real chunks.jsonl content --
not a short synthetic prompt. 5 real chunks (150-400 tok each, not the tiny
ones), all about Localization so the concatenated context reads as a
coherent real retrieval result, + a system instruction + a real question."""
import json
from pathlib import Path

CHUNKS_PATH = Path(__file__).parent.parent / "chunks.jsonl"

CONTEXT_CHUNK_IDS = [
    r"design\autoware-architecture-v1\components\localization\index.md::chunk1",
    r"design\autoware-architecture-v1\components\localization\index.md::chunk3",
    r"design\autoware-architecture-v1\components\localization\index.md::chunk6",
    r"design\autoware-architecture-v1\components\localization\index.md::chunk19",
    r"design\autoware-architecture-v1\components\localization\index.md::chunk20",
]

SYSTEM_PROMPT = (
    "You are a technical assistant answering questions about the Autoware "
    "autonomous driving software architecture. Answer using ONLY the "
    "retrieved context below. If the context doesn't contain the answer, say so."
)

QUESTION = "Based on the context, what sensors does Autoware's localization component rely on, and what happens if a sensor's expected conditions aren't met?"


def format_context_prompt(chunks, question, system_prompt=SYSTEM_PROMPT):
    """chunks: list of chunk dicts (need 'breadcrumb' and 'text'). Same
    structure validated in the standalone VRAM/load test -- shared here
    so the real retrieval pipeline and that benchmark never drift apart."""
    context_blocks = [f"[Source: {c['breadcrumb']}]\n{c['text']}" for c in chunks]
    context = "\n\n---\n\n".join(context_blocks)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Retrieved context:\n\n{context}\n\nQuestion: {question}"},
    ]


def build_rag_prompt():
    chunks = {c["chunk_id"]: c for c in (json.loads(l) for l in CHUNKS_PATH.open(encoding="utf-8"))}
    selected = [chunks[cid] for cid in CONTEXT_CHUNK_IDS]
    return format_context_prompt(selected, QUESTION)


if __name__ == "__main__":
    messages = build_rag_prompt()
    total_chars = sum(len(m["content"]) for m in messages)
    print(f"messages built, ~{total_chars} chars, ~{total_chars // 4} tokens (rough estimate)")
