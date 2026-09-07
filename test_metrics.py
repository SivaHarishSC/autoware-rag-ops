from metrics import recall_at_k, reciprocal_rank

assert recall_at_k(["a", "b", "c"], {"c"}, 5) == 1.0
assert recall_at_k(["a", "b", "c"], {"z"}, 5) == 0.0
assert recall_at_k(["a", "b", "c", "d", "e", "f"], {"f"}, 5) == 0.0  # rank 6, outside top-5
assert recall_at_k(["a", "b", "c"], {"x", "c"}, 5) == 1.0  # any acceptable match counts

assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3
assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0
assert reciprocal_rank(["a", "b", "c"], {"a", "c"}) == 1.0  # first match wins

print("ok")
