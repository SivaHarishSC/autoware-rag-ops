# Autoware RAG — retrieval, serving, and CI/CD

A RAG pipeline over a subset of [Autoware](https://github.com/autowarefoundation/autoware-documentation)
architecture docs: hybrid retrieval (dense + BM25, RRF-fused, cross-encoder
reranked, doc-type boosted) feeding a locally-served NF4-quantized LLM, gated
by a CI eval that blocks regressions before an image is built.

Full diagnostic narrative (including a self-caught analysis mistake, kept
verbatim) is in [RETRIEVAL_NOTES.md](RETRIEVAL_NOTES.md). This file is the
current-state map: what's here, how to run it, what's proven and what isn't.

## Pipeline

```
autoware-documentation/  (raw clone)
  -> dataset_making/make.py           filter to a 25-file subset
  -> dataset_making/clean_corpus.py   strip MkDocs admonitions/links
  -> chunk_corpus.py                  header+paragraph chunking -> chunks.jsonl (178 chunks)
  -> embed_chunks.py                  bge-large-en-v1.5 -> Qdrant (local mode, qdrant_data/)
  -> bm25_index.py                    BM25Okapi, rebuilt in-memory (no persisted index)

query -> hybrid_search.py (dense + BM25 top-20, weighted RRF 0.7/0.3)
      -> rerank.py (BAAI/bge-reranker-large over the fused top-20)
      -> doctype.py (rerank-stage boost for design-doc vs interface-doc queries)
      -> top-5 -> retrieve_and_generate.py -> serving/server.py (NF4 Mistral-7B)
```

`retrieve_and_generate.py`'s `RetrievalPipeline` is the single production
retrieval code path — the CI eval gate, the eval-generation script, and the
CLI demo all call the same `retrieve()` method, so there's no separate
"eval version" of the logic that can drift from what actually serves a query.

Every raw `chunks.jsonl` row is reproducible from the checked-in scripts —
verified by rerunning `make.py` -> `clean_corpus.py` -> `chunk_corpus.py`
from a clean clone and diffing the output against the committed file
(byte-identical, all 178 rows, `doc_type` included).

## Results (18-query eval set, `eval_set.jsonl`)

| config | Recall@5 | MRR |
|---|---|---|
| dense-only | 0.778 | 0.555 |
| hybrid (naive RRF) | 0.611 | 0.479 |
| hybrid (0.7/0.3) + rerank | 0.722 | 0.557 |
| **hybrid + rerank + doctype (production)** | **0.778** | **0.559** |

Naive hybrid fusion regresses vs. dense-only alone — root cause and the
doctype-boost fix (plus its honest coverage limit: it only fires when a
query literally says "interface spec" / "design doc") are in
RETRIEVAL_NOTES.md. Numbers above are read directly from a real
`ci/ci_eval.py` run, not retyped from memory.

## Running it

```bash
pip install -r requirements-retrieval.txt

python dataset_making/make.py          # needs autoware-documentation/ cloned alongside
python dataset_making/clean_corpus.py
python chunk_corpus.py
python embed_chunks.py                 # builds qdrant_data/

python ci/ci_eval.py                   # retrieval-only Recall@5/MRR
python retrieve_and_generate.py        # needs serving/server.py running (see below)
```

Self-checks (plain `assert`-based, no framework): `test_*.py` at the repo
root, `ci/test_gate.py`. Run any of them directly with `python <file>`.

## Serving

`serving/server.py` — FastAPI, OpenAI-compatible `/v1/chat/completions` +
`/health`. Loads an NF4 bitsandbytes-quantized model from a private HF repo
(`serving/model_config.py`); despite its name (`llama-3.1-8b-instruct-nf4`),
`config.json` shows this is actually a Mistral-7B-class model
(`MistralForCausalLM`, vocab 32768) — the code follows the real
architecture, not the repo name.

Measured on an RTX 4050 (~5.9GB usable VRAM): a single instance peaks at
~5.4-5.8GB depending on context size. `retrieve_and_generate.py` caps
assembled context to `CONTEXT_TOKEN_BUDGET=2000` tokens (real tokenizer
count, not the chars/4 approximation used elsewhere — that approximation
was found to undercount markup-heavy chunks like mermaid diagrams badly
enough to miss the cap firing) to keep headroom tight but real.

```bash
cd serving && pip install -r requirements.txt
HF_TOKEN=<token> python server.py
```

## CI/CD (`.github/workflows/ci-cd.yml`)

Two jobs, gated:

1. **eval-gate** — rebuilds the Qdrant index fresh from `chunks.jsonl` on
   every push/PR, runs `ci/ci_eval.py` (the same `RetrievalPipeline.retrieve()`
   production calls), compares Recall@5/MRR against `ci/champion.json`.
   Recall@5 has zero regression tolerance (18 queries is too small a set for
   "acceptable" drift); MRR tolerates a 0.005 jitter band (cross-encoder
   floating-point/library-version noise). On pass, ratchets the champion
   baseline forward only on a genuine improvement.
2. **build-and-push** — only runs if `eval-gate` passed. Builds
   `serving/Dockerfile`, pushes to GHCR tagged with the git SHA (never
   `latest`), authenticated via the workflow's auto-issued `GITHUB_TOKEN`
   (short-lived, repo-scoped, nothing to leak or rotate — not a stored
   secret, though not a full OIDC token exchange like Azure ACR federation
   would be, since this project has no cloud tenant set up for that).

**Scope, stated plainly:** the eval gate is retrieval-only. It never calls
the LLM server or scores generated-answer quality — a GPU-backed model
server isn't available on a standard GitHub-hosted runner, and standing one
up is a separate problem from this gate. `RetrievalPipeline`'s LLM tokenizer
(needed only for the context-budget check in real generation) loads lazily,
specifically so the CI gate never needs auth for the gated model repo.

**Verified, not just written:** ran `ci/ci_eval.py` for real (Recall@5=0.7778,
MRR=0.5593, seeded as the champion baseline from this actual run) and
confirmed `ci/gate.py` genuinely blocks — hand-edited a copy of the result to
simulate a regression and watched the gate exit 1 with both failure reasons
printed, which is what `needs: eval-gate` uses to stop `build-and-push`.

**Real Actions runs, not just local verification:** pushed to
[github.com/SivaHarishSC/autoware-rag-ops](https://github.com/SivaHarishSC/autoware-rag-ops)
and watched GitHub Actions actually execute the YAML. The first real run hit
a genuine bug: `docker/build-push-action` rejected the image tag with
`invalid tag ... repository name must be lowercase`, because
`github.repository` (`SivaHarishSC/autoware-rag-ops`) has uppercase letters
and GHCR/OCI image names must be all-lowercase. Fixed by adding a step that
lowercases the repo name once (`${GITHUB_REPOSITORY,,}`) and referencing that
computed output in the tag, instead of the raw context. The retriggered run
went green end to end — `eval-gate` passed (same 0.7778/0.5593 numbers) and
`build-and-push` produced a real image:

```
ghcr.io/sivaharishsc/autoware-rag-ops/rag-serving:25c9c80630f7ac45b09035c8907c45de53f832da
```

confirmed via GHCR's own package page, not assumed from a green checkmark.

## Repo layout

| Path | What |
|---|---|
| `dataset_making/` | raw corpus filter + clean (Phase 1, steps 0-2) |
| `chunk_corpus.py`, `embed_chunks.py`, `bm25_index.py` | chunking + indexing |
| `hybrid_search.py`, `rerank.py`, `doctype.py`, `metrics.py` | retrieval + scoring |
| `evaluate_doctype.py` | the 4-config comparison that produced the results table above |
| `eval_set.jsonl` | 18 hand-labeled, ground-truth-verified eval queries |
| `retrieve_and_generate.py` | production retrieval + generation pipeline |
| `run_eval_generation.py` | full end-to-end eval (real retrieval + real generation) |
| `serving/` | FastAPI serving layer + Dockerfile |
| `ci/` | CI eval gate (`ci_eval.py`, `gate.py`) + champion baseline |
| `RETRIEVAL_NOTES.md` | full diagnostic narrative behind the numbers above |
