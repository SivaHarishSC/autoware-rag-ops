"""
Hybrid retrieval: dense (Qdrant/bge-large) + sparse (BM25) fused with
Reciprocal Rank Fusion. RRF score for a chunk = sum over every system
that retrieved it of 1/(k + rank_in_that_system), k=60 (the standard
default from the original RRF paper). A chunk retrieved by only one
system still scores, from that system alone -- no averaging, no
dropping singly-retrieved results.
"""
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from bm25_index import load_bm25_index, tokenize
from embed_chunks import COLLECTION, MODEL_NAME, QDRANT_PATH, QUERY_INSTRUCTION

RRF_K = 60


def dense_search(model, client, query, top_k=10):
    """Dense hits are embedding PIECES (a sub-piece of an oversized chunk
    shares its parent's chunk_id). Dedupe to chunk-level ranking, keeping
    each chunk's best (first-seen) rank, before it ever reaches fusion --
    otherwise one oversized chunk could double-dip via two sub-pieces."""
    vec = model.encode(QUERY_INSTRUCTION + query, normalize_embeddings=True).tolist()
    hits = client.query_points(COLLECTION, query=vec, limit=max(top_k * 3, 30)).points

    ranked_ids, scores = [], {}
    for h in hits:
        cid = h.payload["chunk_id"]
        if cid not in scores:
            scores[cid] = h.score
            ranked_ids.append(cid)
        if len(ranked_ids) >= top_k:
            break
    return ranked_ids, scores


def bm25_search(bm25, chunk_ids, query, top_k=10):
    scores_arr = bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores_arr)), key=lambda i: -scores_arr[i])[:top_k]
    ranked_ids = [chunk_ids[i] for i in order if scores_arr[i] > 0]
    scores = {chunk_ids[i]: float(scores_arr[i]) for i in order if scores_arr[i] > 0}
    return ranked_ids, scores


def rrf_fuse(dense_ids, bm25_ids, k=RRF_K, w_dense=1.0, w_bm25=1.0):
    """Returns [(chunk_id, rrf_score, {'dense','bm25'} subset), ...] sorted desc.

    w_dense/w_bm25 default to 1.0 (unweighted/naive RRF, the original
    paper's formula). Weighting down BM25 matters because BM25 always
    returns SOME ranking even when it has no real lexical signal for a
    query (generic English, no rare terms) -- unweighted RRF then blends
    that noise in at full strength alongside a good dense ranking."""
    rrf_scores, contributors = {}, {}
    for rank, cid in enumerate(dense_ids, start=1):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + w_dense / (k + rank)
        contributors.setdefault(cid, set()).add("dense")
    for rank, cid in enumerate(bm25_ids, start=1):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + w_bm25 / (k + rank)
        contributors.setdefault(cid, set()).add("bm25")

    fused = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return [(cid, score, contributors[cid]) for cid, score in fused]


def hybrid_search(model, client, bm25, chunk_ids, query, top_k=10):
    dense_ids, dense_scores = dense_search(model, client, query, top_k)
    bm25_ids, bm25_scores = bm25_search(bm25, chunk_ids, query, top_k)
    fused = rrf_fuse(dense_ids, bm25_ids)
    return fused, dense_ids, bm25_ids


def _label(sources):
    if sources == {"dense", "bm25"}:
        return "both"
    return "dense only" if sources == {"dense"} else "bm25 only"


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What sensors does the localization component rely on?"
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(path=str(QDRANT_PATH))
    bm25, chunk_ids, chunks = load_bm25_index()
    breadcrumb = {c["chunk_id"]: c["breadcrumb"] for c in chunks}

    fused, dense_ids, bm25_ids = hybrid_search(model, client, bm25, chunk_ids, query, top_k=10)
    print(f"QUERY: {query}")
    for cid, score, sources in fused[:10]:
        print(f"  {score:.5f}  [{_label(sources):10s}]  {cid}  |  {breadcrumb[cid]}")
