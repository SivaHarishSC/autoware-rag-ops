from hybrid_search import rrf_fuse, RRF_K

# chunk 'a' retrieved by both systems -> summed contribution
# chunk 'b' dense only, chunk 'c' bm25 only -> single-system score, not dropped
dense_ids = ["a", "b"]
bm25_ids = ["a", "c"]
fused = rrf_fuse(dense_ids, bm25_ids)
scores = dict((cid, score) for cid, score, _ in fused)
sources = dict((cid, srcs) for cid, _, srcs in fused)

assert scores["a"] == 1 / (RRF_K + 1) + 1 / (RRF_K + 1)
assert scores["b"] == 1 / (RRF_K + 2)
assert scores["c"] == 1 / (RRF_K + 2)
assert sources["a"] == {"dense", "bm25"}
assert sources["b"] == {"dense"}
assert sources["c"] == {"bm25"}

# both-system chunk outranks either single-system chunk at equal rank
assert scores["a"] > scores["b"] == scores["c"]

# order matters: better rank (lower number) scores higher
fused2 = rrf_fuse(["x", "y"], [])
s2 = dict((cid, score) for cid, score, _ in fused2)
assert s2["x"] > s2["y"]

# weighted RRF: downweighting BM25 should let a dense-only rank-1 chunk
# beat a chunk BM25 ranks rank-1 but dense doesn't retrieve at all
weighted = rrf_fuse(["p"], ["q"], w_dense=0.7, w_bm25=0.3)
wscores = dict((cid, score) for cid, score, _ in weighted)
assert wscores["p"] == 0.7 / (RRF_K + 1)
assert wscores["q"] == 0.3 / (RRF_K + 1)
assert wscores["p"] > wscores["q"]

print("ok")
