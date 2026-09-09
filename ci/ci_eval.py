"""
CI eval gate. Retrieval-only -- reuses RetrievalPipeline.retrieve() from
retrieve_and_generate.py exactly, the SAME code path production calls,
so there's no separate "CI version" of the retrieval logic that could
drift from what actually serves users. Does NOT call the LLM server or
score generation quality: a GPU-backed Mistral-7B server isn't
available on a standard GitHub-hosted runner, and standing one up is a
distinct problem from this eval gate. Scope: retrieval quality only
(Recall@5, MRR against eval_set.jsonl's ground truth), same metric
definitions as metrics.py's recall_at_k/reciprocal_rank used throughout
Phase 1.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics import recall_at_k, reciprocal_rank
from retrieve_and_generate import RetrievalPipeline, TOP_K_FINAL

EVAL_SET_PATH = Path(__file__).parent.parent / "eval_set.jsonl"
RESULT_PATH = Path(__file__).parent / "eval_result.json"


def main():
    eval_rows = [json.loads(l) for l in EVAL_SET_PATH.open(encoding="utf-8")]
    pipeline = RetrievalPipeline()

    per_query = []
    for row in eval_rows:
        query, expected = row["query"], set(row["expected_chunk_ids"])
        top_ids, _hint, _scores = pipeline.retrieve(query, k=TOP_K_FINAL)
        per_query.append({
            "query": row["query"],
            "type": row["query_type"],
            "recall_at_5": recall_at_k(top_ids, expected, TOP_K_FINAL),
            "rr": reciprocal_rank(top_ids, expected),
        })

    n = len(per_query)
    recall_at_5 = sum(r["recall_at_5"] for r in per_query) / n
    mrr = sum(r["rr"] for r in per_query) / n

    result = {
        "n_queries": n,
        "recall_at_5": recall_at_5,
        "mrr": mrr,
        "per_query": per_query,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"n={n} Recall@5={recall_at_5:.4f} MRR={mrr:.4f}")
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
