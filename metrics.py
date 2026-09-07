def recall_at_k(ranked_ids, expected_ids, k):
    """1.0 if ANY acceptable expected_id is in the top-k, else 0.0 -- a hit
    rate, not the fraction of all expected_ids retrieved, since eval_set.jsonl
    treats multiple expected_chunk_ids as alternative acceptable answers,
    not a set that must all be found."""
    return 1.0 if any(cid in expected_ids for cid in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids, expected_ids):
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in expected_ids:
            return 1.0 / i
    return 0.0
