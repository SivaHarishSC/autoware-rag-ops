# Retrieval/reranking service -- CPU only, no GPU/CUDA base image needed
# (dense embed + BM25 + cross-encoder rerank all run on CPU). Separate
# from serving/Dockerfile (the GPU LLM container) on purpose: different
# resource profile, different failure modes, different canary target.
#
# Build from the PROJECT ROOT as context (not this directory), since it
# needs sibling modules + chunks.jsonl:
#   docker build -f retrieval_service/Dockerfile -t retrieval-service:local .
FROM python:3.12-slim

WORKDIR /app

COPY requirements-retrieval.txt retrieval_service/requirements.txt ./
RUN pip install --no-cache-dir -r requirements-retrieval.txt -r requirements.txt

# Retrieval-side modules + the source-of-truth corpus. serving/model_config.py
# and serving/rag_prompt.py are pulled in only because retrieve_and_generate.py
# imports them at module scope (for the LLM-serving wiring); this container
# never calls generation, but needs those two lightweight files present to
# satisfy the import (no torch/CUDA/gated-model dependency triggered -- see
# retrieval_service/app.py's docstring).
COPY bm25_index.py chunk_corpus.py doctype.py embed_chunks.py hybrid_search.py \
     metrics.py rerank.py retrieve_and_generate.py chunks.jsonl ./
COPY serving/model_config.py serving/rag_prompt.py ./serving/
COPY retrieval_service/app.py ./retrieval_service/

# Bake the model WEIGHTS into the image at build time, downloaded once
# here rather than from the pod at every cold start -- this is a
# genuinely separate decision from the index-rebuild-at-startup choice
# above. The index rebuild is about freshness (chunks.jsonl can change
# between builds, the Qdrant/BM25 index must never silently drift from
# it); the model weights are fixed, pinned-by-name artifacts that never
# change between builds of the same image, so there's no freshness
# argument for re-downloading them at runtime -- only image-size cost,
# which is the deliberate trade being made here (measured before/after
# in retrieval_service's cold-start report).
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
from embed_chunks import MODEL_NAME; \
from rerank import RERANKER_NAME; \
SentenceTransformer(MODEL_NAME); \
CrossEncoder(RERANKER_NAME)"

EXPOSE 8001

# Startup still rebuilds the Qdrant + BM25 index fresh from chunks.jsonl
# (see app.py's docstring) -- that part of cold start is unchanged and
# deliberately so. Only the model-weight download moved to build time.
CMD ["uvicorn", "retrieval_service.app:app", "--host", "0.0.0.0", "--port", "8001"]
