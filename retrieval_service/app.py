"""
Retrieval/reranking service -- Phase 6 canary target. Wraps the FINAL
frozen config: RetrievalPipeline.retrieve() from retrieve_and_generate.py
(weighted RRF 0.7/0.3 dense+BM25, bge-reranker-large rerank, doc_type
boost) -- the SAME code path ci/ci_eval.py gates and retrieve_and_generate
calls for real generation. Not the naive/intermediate hybrid configs from
evaluate_doctype.py's comparison table.

Deliberately separate from serving/ (the GPU LLM container): this service
is CPU-only (dense embed + BM25 + cross-encoder rerank, no LLM), so it
gets its own Dockerfile with a plain python base image, not the CUDA-
adjacent one.

Startup rebuilds the Qdrant + BM25 index fresh from chunks.jsonl (same
choice as the CI eval gate, same reason: a pod should come up clean from
the image + repo alone, with no risk of a baked-in index silently
drifting from chunks.jsonl -- the tradeoff is a slower cold start, traded
for reproducibility a K8s context should prioritize).
"""
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
import embed_chunks  # noqa: E402
from retrieve_and_generate import RetrievalPipeline, TOP_K_FINAL  # noqa: E402

app = FastAPI()
pipeline = None


@app.on_event("startup")
def startup():
    global pipeline
    print("[startup] building Qdrant index from chunks.jsonl ...")
    embed_chunks.main()
    print("[startup] loading retrieval pipeline (dense + BM25 + reranker) ...")
    pipeline = RetrievalPipeline()
    print("[startup] ready")


class RetrieveRequest(BaseModel):
    query: str
    k: int = TOP_K_FINAL


@app.get("/health")
def health():
    return {"status": "ok", "ready": pipeline is not None}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    top_ids, hint, scores = pipeline.retrieve(req.query, k=req.k)
    results = [
        {
            "chunk_id": cid,
            "breadcrumb": pipeline.chunk_text_by_id[cid]["breadcrumb"],
            "text": pipeline.chunk_text_by_id[cid]["text"],
            "score": float(scores[cid]),
        }
        for cid in top_ids
    ]
    return {"query": req.query, "doc_type_hint": hint, "results": results}
