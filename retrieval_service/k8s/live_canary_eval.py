"""
Live canary-vs-stable eval, run against the REAL pods through the
cluster (port-forwarded to retrieval-service-stable-only /
retrieval-service-canary-only), not offline/simulated. Reuses the same
eval_set.jsonl ground truth and metrics.py functions Phase 1 used, so
the comparison is apples-to-apples with the frozen production numbers.

Usage: port-forward both Services first, e.g.:
  kubectl port-forward svc/retrieval-service-stable-only 8011:8001 &
  kubectl port-forward svc/retrieval-service-canary-only 8012:8001 &
  python live_canary_eval.py
"""
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from metrics import recall_at_k, reciprocal_rank  # noqa: E402

STABLE_URL = "http://localhost:8011/retrieve"
CANARY_URL = "http://localhost:8012/retrieve"
EVAL_SET_PATH = Path(__file__).parent.parent.parent / "eval_set.jsonl"


def run_eval(url, eval_rows):
    per_query = []
    for row in eval_rows:
        resp = requests.post(url, json={"query": row["query"], "k": 5}, timeout=60)
        resp.raise_for_status()
        ranked_ids = [r["chunk_id"] for r in resp.json()["results"]]
        expected = set(row["expected_chunk_ids"])
        per_query.append({
            "query": row["query"],
            "type": row["query_type"],
            "recall_at_5": recall_at_k(ranked_ids, expected, 5),
            "rr": reciprocal_rank(ranked_ids, expected),
        })
    n = len(per_query)
    return {
        "n_queries": n,
        "recall_at_5": sum(r["recall_at_5"] for r in per_query) / n,
        "mrr": sum(r["rr"] for r in per_query) / n,
        "per_query": per_query,
    }


def main():
    eval_rows = [json.loads(l) for l in EVAL_SET_PATH.open(encoding="utf-8")]

    print("querying STABLE pod live through the cluster ...")
    stable = run_eval(STABLE_URL, eval_rows)
    print("querying CANARY pod live through the cluster ...")
    canary = run_eval(CANARY_URL, eval_rows)

    print(f"\n{'':10s} {'Recall@5':>10s} {'MRR':>8s}")
    print(f"{'stable':10s} {stable['recall_at_5']:10.4f} {stable['mrr']:8.4f}")
    print(f"{'canary':10s} {canary['recall_at_5']:10.4f} {canary['mrr']:8.4f}")

    print("\nby query_type:")
    for qtype in ["conceptual", "exact_term", "adversarial"]:
        s_sub = [r for r in stable["per_query"] if r["type"] == qtype]
        c_sub = [r for r in canary["per_query"] if r["type"] == qtype]
        s_r = sum(r["recall_at_5"] for r in s_sub) / len(s_sub)
        c_r = sum(r["recall_at_5"] for r in c_sub) / len(c_sub)
        print(f"  {qtype:12s} stable={s_r:.3f}  canary={c_r:.3f}")

    # Rollback decision logic -- the real signal a rollback trigger keys
    # off: zero tolerance on Recall@5 regression, same rule ci/gate.py
    # uses for the CI eval gate (18 queries is too small a set to treat
    # any drop as noise).
    regressed = canary["recall_at_5"] < stable["recall_at_5"]
    print(f"\nROLLBACK TRIGGER: {'YES -- canary Recall@5 regressed vs stable' if regressed else 'no'}")
    if regressed:
        print(f"  stable Recall@5={stable['recall_at_5']:.4f} > canary Recall@5={canary['recall_at_5']:.4f}")

    result = {"stable": stable, "canary": canary, "rollback_triggered": regressed}
    out_path = Path(__file__).parent / "live_canary_eval_result.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
