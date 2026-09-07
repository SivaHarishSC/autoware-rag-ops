"""
Runs eval_set.jsonl end-to-end through retrieve_and_generate() (real
retrieval + real generation via the live server), not retrieval alone.
Records retrieval hit/miss, generated answer, and per-query latency
breakdown (retrieval time vs. generation/HTTP time).
"""
import json
import time

import requests

from retrieve_and_generate import RetrievalPipeline, assemble_capped_context
from serving.rag_prompt import format_context_prompt

OUT_PATH = "eval_generation_results_capped.json"


def main():
    eval_rows = [json.loads(l) for l in open("eval_set.jsonl", encoding="utf-8")]
    pipeline = RetrievalPipeline()

    results = []
    for i, row in enumerate(eval_rows, 1):
        query = row["query"]
        expected = set(row["expected_chunk_ids"])

        t0 = time.time()
        top_ids, hint = pipeline.retrieve(query, k=5)
        ranked_chunks = [pipeline.chunk_text_by_id[cid] for cid in top_ids]
        context_chunks, context_meta = assemble_capped_context(ranked_chunks, count_fn=pipeline.count_real_tokens)
        t1 = time.time()

        messages = format_context_prompt(context_chunks, query)
        prompt_chars = sum(len(m["content"]) for m in messages)

        resp = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={"messages": messages, "max_tokens": 256},
            timeout=180,
        )
        t2 = time.time()
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]

        hit = bool(expected & set(top_ids))
        included_hit = bool(expected & set(context_meta["included_chunk_ids"]))
        record = {
            "n": i,
            "query": query,
            "type": row["query_type"],
            "expected_chunk_ids": sorted(expected),
            "retrieved_chunk_ids": top_ids,
            "doc_type_hint": hint,
            "retrieval_hit": hit,
            "context_candidates": context_meta["candidates"],
            "context_included": context_meta["included"],
            "context_truncated": context_meta["truncated"],
            "included_chunk_ids": context_meta["included_chunk_ids"],
            "included_hit": included_hit,  # correct chunk retrieved AND survived capping into the actual prompt
            "prompt_chars_est": prompt_chars,
            "prompt_tokens_actual": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
            "retrieval_latency_s": round(t1 - t0, 2),
            "generation_latency_s": round(t2 - t1, 2),
            "total_latency_s": round(t2 - t0, 2),
            "answer": answer,
        }
        results.append(record)
        print(f"[{i}/18] hit={hit} included_hit={included_hit} truncated={context_meta['truncated']} "
              f"type={row['query_type']:11s} prompt_tok={record['prompt_tokens_actual']:4d} "
              f"retrieval={record['retrieval_latency_s']:5.2f}s gen={record['generation_latency_s']:5.2f}s | {query[:60]}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
