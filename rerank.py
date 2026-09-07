"""Cross-encoder reranking stage.

Reranker: BAAI/bge-reranker-large (not the lighter cross-encoder/ms-marco-
MiniLM-L-6-v2). Chosen for quality in the original offline eval (18 queries
x 20 candidates); now also used live per-request in retrieve_and_generate.py
-- reranking 20 candidates is a few hundred ms on CPU, acceptable next to
the multi-second LLM generation it feeds. Swap to MiniLM if rerank latency
ever becomes the bottleneck.
"""
RERANKER_NAME = "BAAI/bge-reranker-large"


def rerank(model, query, chunk_ids, chunk_text_by_id):
    """Returns chunk_ids reordered by the cross-encoder's relevance score."""
    pairs = [(query, chunk_text_by_id[cid]) for cid in chunk_ids]
    scores = model.predict(pairs)
    order = sorted(range(len(chunk_ids)), key=lambda i: -scores[i])
    return [chunk_ids[i] for i in order]
