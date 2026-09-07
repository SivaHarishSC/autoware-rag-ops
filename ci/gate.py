"""
Compares this run's eval result (ci/eval_result.json, written by
ci_eval.py) against the stored champion baseline (ci/champion.json).

Gate:
  - FAIL if new Recall@5 < champion Recall@5. Zero tolerance -- 18
    queries is small enough that ANY drop reflects a real behavior
    change (a config edit, a dependency bump changing model output),
    not sampling noise; there's no "acceptable" regression to allow.
  - FAIL if new MRR < champion MRR - MRR_TOLERANCE. MRR is a continuous
    average of reciprocal ranks, not a binary hit/miss like Recall@5 --
    it can shift by a hair between runs from floating-point/library-
    version jitter in the cross-encoder's scores even with identical
    logic and identical retrieved chunks. MRR_TOLERANCE=0.005 (~1% of
    the ~0.56 baseline) absorbs that jitter without hiding a real
    regression, which would show up as a much larger drop.

On pass: ratchets champion.json forward ONLY if the new result is
strictly better on at least one metric (and never worse on the other,
since a pass already guarantees that) -- so champion updates when a
genuine improvement lands, and stays untouched (no pointless commit)
when a run just matches the existing baseline.
"""
import json
import sys
from pathlib import Path

MRR_TOLERANCE = 0.005

RESULT_PATH = Path(__file__).parent / "eval_result.json"
CHAMPION_PATH = Path(__file__).parent / "champion.json"


def evaluate_gate(new, champion, mrr_tolerance=MRR_TOLERANCE):
    """Pure decision logic, no I/O -- returns (passed, failures, should_ratchet)."""
    failures = []
    if new["recall_at_5"] < champion["recall_at_5"]:
        failures.append(
            f"Recall@5 regressed: {new['recall_at_5']:.4f} < champion {champion['recall_at_5']:.4f} (zero tolerance)"
        )
    if new["mrr"] < champion["mrr"] - mrr_tolerance:
        failures.append(
            f"MRR regressed beyond tolerance: {new['mrr']:.4f} < champion {champion['mrr']:.4f} - {mrr_tolerance}"
        )

    passed = not failures
    should_ratchet = passed and (new["recall_at_5"] > champion["recall_at_5"] or new["mrr"] > champion["mrr"])
    return passed, failures, should_ratchet


def main():
    new = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    if not CHAMPION_PATH.exists():
        print("no champion baseline yet -- seeding with this run's result")
        write_champion(new)
        return

    champion = json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))
    print(f"champion: Recall@5={champion['recall_at_5']:.4f} MRR={champion['mrr']:.4f}")
    print(f"new run:  Recall@5={new['recall_at_5']:.4f} MRR={new['mrr']:.4f}")

    passed, failures, should_ratchet = evaluate_gate(new, champion)

    if not passed:
        print("GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("GATE PASSED")
    if should_ratchet:
        print("new result beats champion on at least one metric -- ratcheting baseline forward")
        write_champion(new)
    else:
        print("result matches champion exactly, baseline left unchanged")


def write_champion(result):
    CHAMPION_PATH.write_text(json.dumps({
        "recall_at_5": result["recall_at_5"],
        "mrr": result["mrr"],
        "n_queries": result["n_queries"],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
