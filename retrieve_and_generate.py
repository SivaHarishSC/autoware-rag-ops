"""
Wires the evaluated Phase 1 retrieval pipeline to the FastAPI serving
endpoint. Frozen production config -- the only one that matched-or-beat
dense-only on both Recall@5 and MRR across eval_set.jsonl:
  hybrid RRF (w_dense=0.7, w_bm25=0.3) over dense+BM25 top-20 each,
  reranked with BAAI/bge-reranker-large, doc_type boost applied,
  top-K taken after boost.

K=5, matching TOP_K_FINAL in evaluate_doctype.py -- the value Recall@5 was
actually evaluated at. NOT derived from serving/rag_prompt.py's hardcoded
5-chunk benchmark list, which is an unrelated ad hoc token-budget test
that happens to also use 5.

Query embedding uses embed_chunks.MODEL_NAME (bge-large-en-v1.5) imported
directly, never a separately hardcoded model string -- the same model
that built the Qdrant index, verified by construction rather than by
inspection.
"""
import sys
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoTokenizer

from bm25_index import load_bm25_index
from chunk_corpus import count_tokens
from doctype import apply_doc_type_boost, detect_doc_type_hint
from embed_chunks import COLLECTION, MODEL_NAME, QDRANT_PATH, is_markdown_table
from hybrid_search import bm25_search, dense_search, rrf_fuse
from rerank import RERANKER_NAME

sys.path.insert(0, str(Path(__file__).parent / "serving"))
from model_config import MODEL_ID  # noqa: E402
from rag_prompt import format_context_prompt  # noqa: E402

TOP_K_CANDIDATES = 20
TOP_K_FINAL = 5
W_DENSE, W_BM25 = 0.7, 0.3
SERVER_URL = "http://localhost:8000/v1/chat/completions"

# Retrieved-context token budget (separate from the small, fixed system
# instruction + question). Caps the KV cache growth that pushed a single
# model instance to 95% VRAM utilization when a top-5 candidate happened
# to be an oversized atomic chunk (e.g. the 2710-token planning table) --
# leaving no headroom for a second concurrent instance in a K8s canary.
CONTEXT_TOKEN_BUDGET = 2000


def _truncate_to_budget(text, budget_tokens, count_fn):
    """Truncate at a clean boundary -- end of a table row, or end of a
    line -- never mid-row/mid-word. Measures the REAL assembled string's
    token count at each step via count_fn (not a sum of independently-
    rounded per-line counts, which undercounts -- floor-division per
    short line loses a few chars each time, and summed counts never
    account for the newlines joining them back together). Caller's
    budget covers the body only; the appended '[... truncated ...]'
    note adds a small, fixed amount on top so the model doesn't present
    partial data as complete."""
    lines = text.strip("\n").split("\n")

    if is_markdown_table(text):
        header, sep, rows = lines[0], lines[1], lines[2:]
        kept = []
        for row in rows:
            candidate = f"{header}\n{sep}\n" + "\n".join(kept + [row])
            if count_fn(candidate) > budget_tokens:
                break
            kept.append(row)
        note = f"[table truncated, showing first {len(kept)} of {len(rows)} rows]"
        return f"{header}\n{sep}\n" + "\n".join(kept) + f"\n{note}", True

    kept = []
    for line in lines:
        candidate = "\n".join(kept + [line])
        if count_fn(candidate) > budget_tokens:
            break
        kept.append(line)
    note = f"[content truncated, showing first {len(kept)} of {len(lines)} lines]"
    return "\n".join(kept) + f"\n{note}", True


def assemble_capped_context(ranked_chunks, budget_tokens=CONTEXT_TOKEN_BUDGET, count_fn=count_tokens):
    """ranked_chunks: chunk dicts in ranked (best-first) order. Adds
    chunks until the next one would exceed budget; if the #1-ranked
    chunk alone exceeds budget, truncates just that one chunk at a
    clean boundary and stops. Returns (chunks_for_prompt, meta) where
    meta reports exactly what happened, for per-request logging.

    count_fn defaults to chunk_corpus.count_tokens (the ~4-chars/token
    approximation used everywhere else in this project) for standalone/
    offline use. RetrievalPipeline passes the REAL model tokenizer's
    count instead -- the approximation was found to undercount markup-
    heavy chunks (e.g. the mermaid diagram) badly enough that a 5-chunk
    context estimated at 1938 tokens actually tokenized to ~3088 real
    tokens, letting the exact motivating case for this budget slip
    through uncapped. Enforcing the budget against an approximation
    that's wrong in precisely the direction that matters isn't a fix."""
    selected = []
    running = 0
    truncated = False

    for c in ranked_chunks:
        tokens = count_fn(c["text"])
        if not selected and tokens > budget_tokens:
            truncated_text, truncated = _truncate_to_budget(c["text"], budget_tokens, count_fn)
            selected.append({**c, "text": truncated_text})
            break
        if running + tokens > budget_tokens:
            break
        selected.append(c)
        running += tokens

    meta = {
        "candidates": len(ranked_chunks),
        "included": len(selected),
        "truncated": truncated,
        "included_chunk_ids": [c["chunk_id"] for c in selected],
    }
    return selected, meta


class RetrievalPipeline:
    """Loads every retrieval-side model ONCE; call retrieve()/retrieve_and_generate()
    per query rather than constructing a new instance per call."""

    def __init__(self):
        self.embed_model = SentenceTransformer(MODEL_NAME)
        self.qdrant = QdrantClient(path=str(QDRANT_PATH))
        self.bm25, self.chunk_ids, chunks = load_bm25_index()
        self.chunk_text_by_id = {c["chunk_id"]: c for c in chunks}
        self.doc_type_by_id = {c["chunk_id"]: c["doc_type"] for c in chunks}
        self.reranker = CrossEncoder(RERANKER_NAME)
        self._llm_tokenizer = None  # lazy: needs HF auth for this gated repo,
        # and retrieve()-only callers (e.g. the CI eval gate) never touch it

    def count_real_tokens(self, text):
        if self._llm_tokenizer is None:
            # same tokenizer the LLM actually uses -- for the context budget
            # check, not the chars/4 approximation (see assemble_capped_context
            # docstring)
            self._llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        return len(self._llm_tokenizer.encode(text, add_special_tokens=False))

    def retrieve(self, query, k=TOP_K_FINAL):
        dense_ids, _ = dense_search(self.embed_model, self.qdrant, query, top_k=TOP_K_CANDIDATES)
        bm25_ids, _ = bm25_search(self.bm25, self.chunk_ids, query, top_k=TOP_K_CANDIDATES)
        fused = [cid for cid, _, _ in rrf_fuse(dense_ids, bm25_ids, w_dense=W_DENSE, w_bm25=W_BM25)][:TOP_K_CANDIDATES]

        pairs = [(query, self.chunk_text_by_id[cid]["text"]) for cid in fused]
        base_scores = list(self.reranker.predict(pairs))

        hint = detect_doc_type_hint(query)
        boosted_scores = apply_doc_type_boost(fused, base_scores, self.doc_type_by_id, hint)
        order = sorted(range(len(fused)), key=lambda i: -boosted_scores[i])
        reranked = [fused[i] for i in order]

        return reranked[:k], hint

    def retrieve_and_generate(self, query, k=TOP_K_FINAL, max_tokens=256):
        top_ids, hint = self.retrieve(query, k)
        ranked_chunks = [self.chunk_text_by_id[cid] for cid in top_ids]
        context_chunks, context_meta = assemble_capped_context(ranked_chunks, count_fn=self.count_real_tokens)
        messages = format_context_prompt(context_chunks, query)

        print(f"[context] candidates={context_meta['candidates']} included={context_meta['included']} "
              f"truncated={context_meta['truncated']} chunk_ids={context_meta['included_chunk_ids']}")

        resp = requests.post(SERVER_URL, json={"messages": messages, "max_tokens": max_tokens}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]

        return {
            "query": query,
            "doc_type_hint": hint,
            "retrieved_chunk_ids": top_ids,
            "context_meta": context_meta,
            "answer": answer,
            "usage": data["usage"],
        }


if __name__ == "__main__":
    pipeline = RetrievalPipeline()
    result = pipeline.retrieve_and_generate("What sensors does the localization component rely on?")
    print(result)
