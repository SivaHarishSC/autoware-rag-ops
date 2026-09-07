"""
Targeted fix for the adversarial-category regression: a rerank-stage
doc_type boost using a keyword-detected query hint (see doctype.py for
the exact phrase list -- fragile by design, not hidden).

4 configs, same eval_set.jsonl, one consolidated table:
  dense                       -- bge-large-en-v1.5 / Qdrant alone
  hybrid(naive)                -- dense + BM25 top-20, unweighted RRF
  hybrid+rerank                -- weighted RRF (0.7/0.3, from the prior
                                   fix) + cross-encoder rerank
  hybrid+rerank+doctype-boost  -- same, then a doc_type boost applied to
                                   the cross-encoder scores before the
                                   final top-5 cut
"""
import json

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from bm25_index import load_bm25_index
from doctype import apply_doc_type_boost, detect_doc_type_hint
from embed_chunks import COLLECTION, MODEL_NAME, QDRANT_PATH
from hybrid_search import bm25_search, dense_search, rrf_fuse
from metrics import recall_at_k, reciprocal_rank
from rerank import RERANKER_NAME

TOP_K_CANDIDATES = 20
TOP_K_FINAL = 5
W_DENSE, W_BM25 = 0.7, 0.3
CONFIGS = ["dense", "hybrid(naive)", "hybrid+rerank", "hybrid+rerank+doctype"]
QUERY_TYPES = ["conceptual", "exact_term", "adversarial"]


def main():
    eval_rows = [json.loads(l) for l in open("eval_set.jsonl", encoding="utf-8")]
    chunks = [json.loads(l) for l in open("chunks.jsonl", encoding="utf-8")]
    chunk_text_by_id = {c["chunk_id"]: c["text"] for c in chunks}
    doc_type_by_id = {c["chunk_id"]: c["doc_type"] for c in chunks}

    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(path=str(QDRANT_PATH))
    bm25, chunk_ids, _ = load_bm25_index()
    reranker = CrossEncoder(RERANKER_NAME)

    results = {name: [] for name in CONFIGS}
    adversarial_trace = []

    for row in eval_rows:
        query, expected = row["query"], set(row["expected_chunk_ids"])

        dense_ids, _ = dense_search(model, client, query, top_k=TOP_K_CANDIDATES)
        bm25_ids, _ = bm25_search(bm25, chunk_ids, query, top_k=TOP_K_CANDIDATES)
        naive = [cid for cid, _, _ in rrf_fuse(dense_ids, bm25_ids)][:TOP_K_CANDIDATES]
        fixed = [cid for cid, _, _ in rrf_fuse(dense_ids, bm25_ids, w_dense=W_DENSE, w_bm25=W_BM25)][:TOP_K_CANDIDATES]

        pairs = [(query, chunk_text_by_id[cid]) for cid in fixed]
        base_scores = list(reranker.predict(pairs))
        order = sorted(range(len(fixed)), key=lambda i: -base_scores[i])
        reranked = [fixed[i] for i in order]

        hint = detect_doc_type_hint(query)
        boosted_scores = apply_doc_type_boost(fixed, base_scores, doc_type_by_id, hint)
        order2 = sorted(range(len(fixed)), key=lambda i: -boosted_scores[i])
        reranked_boosted = [fixed[i] for i in order2]

        rankings = {"dense": dense_ids, "hybrid(naive)": naive, "hybrid+rerank": reranked, "hybrid+rerank+doctype": reranked_boosted}
        for name, ranking in rankings.items():
            results[name].append({
                "type": row["query_type"],
                "recall@5": recall_at_k(ranking, expected, TOP_K_FINAL),
                "rr": reciprocal_rank(ranking, expected),
            })

        if row["query_type"] == "adversarial":
            rank_before = next((i + 1 for i, c in enumerate(reranked) if c in expected), None)
            rank_after = next((i + 1 for i, c in enumerate(reranked_boosted) if c in expected), None)
            adversarial_trace.append((query, hint, rank_before, rank_after))

    def agg(rows):
        n = len(rows)
        return sum(r["recall@5"] for r in rows) / n, sum(r["rr"] for r in rows) / n, n

    print(f"\n{'config':26s} {'scope':12s} {'n':>3s} {'Recall@5':>9s} {'MRR':>7s}")
    print("-" * 62)
    for name in CONFIGS:
        r, m, n = agg(results[name])
        print(f"{name:26s} {'OVERALL':12s} {n:3d} {r:9.3f} {m:7.3f}")
        for qtype in QUERY_TYPES:
            sub = [row for row in results[name] if row["type"] == qtype]
            r, m, n = agg(sub)
            print(f"{'':26s} {qtype:12s} {n:3d} {r:9.3f} {m:7.3f}")
        print("-" * 62)

    print("\nadversarial-query trace (hybrid+rerank -> +doctype-boost):")
    for query, hint, before, after in adversarial_trace:
        change = "unchanged" if before == after else ("IMPROVED" if (after or 999) < (before or 999) else "WORSE")
        print(f"  hint={str(hint):10s} rank {str(before):>4s} -> {str(after):>4s}  [{change:9s}]  {query[:70]}")


if __name__ == "__main__":
    main()
