from gate import MRR_TOLERANCE, evaluate_gate

champion = {"recall_at_5": 0.778, "mrr": 0.559}

# equal to champion -> pass, no ratchet
passed, failures, ratchet = evaluate_gate({"recall_at_5": 0.778, "mrr": 0.559}, champion)
assert passed and not ratchet

# strictly better on both -> pass, ratchet
passed, failures, ratchet = evaluate_gate({"recall_at_5": 0.85, "mrr": 0.60}, champion)
assert passed and ratchet

# Recall@5 drops even slightly -> fail, zero tolerance
passed, failures, ratchet = evaluate_gate({"recall_at_5": 0.777, "mrr": 0.559}, champion)
assert not passed and any("Recall@5" in f for f in failures)

# MRR drops within tolerance -> still passes
passed, failures, ratchet = evaluate_gate({"recall_at_5": 0.778, "mrr": 0.559 - MRR_TOLERANCE}, champion)
assert passed

# MRR drops beyond tolerance -> fails
passed, failures, ratchet = evaluate_gate({"recall_at_5": 0.778, "mrr": 0.559 - MRR_TOLERANCE - 0.001}, champion)
assert not passed and any("MRR" in f for f in failures)

# MRR improves but Recall@5 regresses -> still fails overall (Recall@5 has zero tolerance, no offsetting)
passed, failures, ratchet = evaluate_gate({"recall_at_5": 0.7, "mrr": 0.9}, champion)
assert not passed

print("ok")
