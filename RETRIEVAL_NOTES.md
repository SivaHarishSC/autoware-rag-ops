# Phase 1 retrieval — working notes

Rough writeup, captured while the details are fresh rather than reconstructed later for
a polished README. Purpose is to preserve the actual diagnostic narrative — including
where an early read of the data was wrong and got corrected — not just final numbers.

## Pipeline as built

1. `chunk_corpus.py` — header-based split + paragraph fallback, tables/code fences kept
   atomic. 178 chunks from `autoware-rag-corpus-clean/` → `chunks.jsonl`.
2. `embed_chunks.py` — BAAI/bge-large-en-v1.5, chunks >450 tok sub-split for embedding
   only (table row-groups / code line-groups, header or fence repeated in each piece),
   stored in Qdrant local mode (`qdrant_data/`, no server). 220 vectors.
3. `bm25_index.py` — rank_bm25 BM25Okapi over the full, un-split 178 chunks (verified
   from source that BM25 has no token-window limit, so no sub-splitting need there).
4. `hybrid_search.py` — RRF fusion of dense + BM25 top-20 each.
5. `rerank.py` — BAAI/bge-reranker-large cross-encoder over the fused top-20.
6. `doctype.py` — keyword-detected query hint + rerank-stage boost for design-doc vs
   interface-doc precision.
7. `eval_set.jsonl` — 18 hand-labeled queries (7 conceptual / 7 exact-term / 4
   adversarial), every ground-truth chunk_id checked against actual chunk text, not
   guessed from breadcrumbs.

## Finding 1 — naive RRF fusion is net *negative* vs. dense-only

| config | Recall@5 | MRR |
|---|---|---|
| dense-only | 0.778 | 0.555 |
| hybrid (naive RRF, w=1.0/1.0) | 0.611 | 0.479 |

This was the first real surprise. Diagnosis, done by tracing individual queries rather
than trusting the aggregate:

- Query: *"Why is the Mission Planning sub-component kept separate from the rest of
  Planning?"* — target chunk is `planning/index.md::chunk5`.
  - Dense alone: rank 2 (correct, comfortably in top-5).
  - BM25 alone: target **not in BM25's own top-20 at all**.
  - Fused (unweighted RRF): target pushed to **rank 8** — falls out of top-5.

- The initial hypothesis was "BM25 returns noise for generic conceptual queries with no
  rare terms, and RRF blends that noise in at full strength." That's *half* right, but
  the real mechanism is more specific: BM25's top-20 for this query wasn't random noise —
  it was **other sections of the exact same source document** (`planning/index.md`
  chunks 11, 3, 6, 16, 2, 24, 20 — sibling headers like "Replacing Sub-components of
  Planning," "Separation of Validation sub-component"). Because the whole file shares
  heavy vocabulary, BM25 legitimately ranks several *sibling* chunks decently for a
  generic question about that file's topic. 6–7 of those siblings appear in **both**
  dense's and BM25's top-20, so RRF gives them the "both" bonus — while the actual
  target, being dense-only, can never out-sum a chunk that's getting credit from two
  systems, regardless of how the weights are split between those two systems.
  This is a property of **document granularity** (one long file, many similar
  subsections), not query noise — worth remembering when deciding how finely to chunk
  in later phases.

## Finding 2 — the weight-sweep artifact (a real self-correction, kept verbatim)

To test whether downweighting BM25 would fix the above, I swept `w_dense/w_bm25` for
that same query and initially misread the result:

> `w_dense=1.0 w_bm25=1.0`: fused rank = 8
> `w_dense=0.9 w_bm25=0.1`: fused rank = 5   <- looked like a fix
> `w_dense=0.8 w_bm25=0.2`: fused rank = 8   <- looked like a regression
> `w_dense=0.7 w_bm25=0.3`: fused rank = 8
> `w_dense=0.5 w_bm25=0.5`: fused rank = 8

At first glance this reads as non-monotonic — improving then immediately un-improving —
which would suggest something unstable or buggy in the fusion math. It isn't. The sweep
was changing **both** weights in the same direction on each step (`w_dense` down *and*
`w_bm25` up simultaneously), which is two changes stacked, not one. Both changes push
against the target in the same way (less dense credit for the target, more bm25 credit
for the siblings), so the *correct* expectation was monotonic degradation moving away
from the 0.9/0.1 point — and that's exactly what the dump of raw scores showed once I
printed the actual candidate list instead of just the rank number. There is no bug in
`rrf_fuse`; there was a flawed read of my own sweep design. The real, non-artifactual
conclusion: recovery only happens **very close to the pure-dense extreme** (~0.9/0.1 or
higher), and the paper-suggested 0.7/0.3 starting point is functionally identical to
unweighted RRF for this failure mode — it does not fix it.

Net effect of 0.7/0.3 weighting across the full 18-query eval: Recall@5 unchanged
(0.611 → 0.611, identical per-category too), MRR nudged up marginally (0.479 → 0.502).
Reranking, not weighting, is what actually recovered the conceptual category
(back to 1.000 Recall@5 once rerank was added on top).

## Finding 3 — reranking recovers conceptual + exact-term, not adversarial

| config | OVERALL Recall@5 | OVERALL MRR |
|---|---|---|
| dense-only | 0.778 | 0.555 |
| hybrid (naive) | 0.611 | 0.479 |
| hybrid (0.7/0.3) + rerank | 0.722 | 0.557 |

Reranking (BAAI/bge-reranker-large over the fused top-20) fixed the two exact-term
queries where dense missed the answer entirely but BM25 found it deep in the pool
(fused ranks 6 and 10) — the cross-encoder scores candidates directly rather than by
blended rank, so it correctly promoted both to rank 1. `exact_term` Recall@5 went
0.571 → 1.000 (matching dense's own ceiling for that category).

Adversarial stayed flat or got slightly worse (0.250 → 0.250 Recall@5, MRR
0.115 → 0.092). Traced per-query: 2 of the 4 adversarial queries never had the correct
chunk anywhere in the top-20 candidate pool from *either* dense or BM25 — a genuine
retrieval-stage miss that no reranker can fix (it can only reorder what's already
there). A third had the right chunk in the pool (rank 8) and the cross-encoder still
didn't promote it — the near-duplicate distractor chunks (same topic, wrong doc type)
score just as "relevant" to a general-purpose cross-encoder, since it has no signal for
"right topic, wrong document," only topical relevance.

## Finding 4 — doc_type boost: a real, narrow fix

Added `doc_type` (`design` / `interface` / `other`, derived from `source_file` path) to
every chunk's metadata, plus a keyword-based query-hint detector
(`interface spec` / `interface doc` / `per the interface` / `as opposed to the design
doc` → `interface`; `design doc` / `design spec` / `component design` → `design`;
otherwise `None`). Applied as a rerank-stage score boost (0.5× the score spread among
that query's own candidates), not a hard filter — deliberately, so a wrong or missing
hint can't remove the correct answer, only fail to help it.

| config | OVERALL Recall@5 | OVERALL MRR | adversarial Recall@5 |
|---|---|---|---|
| dense-only | 0.778 | 0.555 | 0.500 |
| hybrid+rerank | 0.722 | 0.557 | 0.250 |
| hybrid+rerank+doctype | **0.778** | **0.559** | **0.500** |

This is the first config to match-or-beat dense-only on both metrics. Per-query trace
on the 4 adversarial queries:

| hint detected | rank before boost | rank after |
|---|---|---|
| `interface` | None (never retrieved) | None (unchanged — boost can't invent a candidate) |
| **None (missed)** | None | None (unchanged) |
| `interface` | 6 | **5 — flipped into top-5** |
| None (correctly — not a design/interface case) | 5 | 5 (already fine) |

**Explicit limitation, not to be lost in a later cleanup pass:** the fix only fired on
2 of the 4 adversarial queries, and only recovered 1 of those 2 (the other's answer was
never in the retrieval candidate pool at all, so no rerank-stage trick could reach it).
The miss (`"What data does the Vehicle Interface component send back to Autoware..."`)
is a real design-vs-interface adversarial case, phrased just as naturally as the ones
that worked — it simply never says the literal words "interface spec" or "interface
doc," it says "Vehicle Interface component," which the keyword list doesn't catch. **A
real user asking the same underlying question in different words gets zero benefit from
this fix.** This is a narrow patch for queries that happen to name their target
document type explicitly, not a general solution to the near-duplicate-document
problem. A general fix would need something more like a classifier over query intent,
or surfacing both doc-type variants of a topic to the generation stage and letting it
sort out which is authoritative — out of scope for Phase 1.

## Open items carried forward (not fixed, deliberately deferred)

- Adversarial-category retrieval (not just reranking) still fails 2/4 outright — the
  correct chunk is never retrieved by dense or BM25 in the first place for near-
  duplicate design-doc/interface-doc pairs. This is a retrieval-stage gap, and with
  only 4 adversarial eval queries it's not clear yet whether it generalizes or whether
  these 2 happen to be unusually hard.
- Document granularity (one long file per component, many similar subsections) is the
  root cause of Finding 1's sibling-competition problem — worth reconsidering if
  chunking strategy changes in a later phase.
- Token-count approximation (~4 chars/token) is still a heuristic, not a real
  tokenizer — flagged since Phase 1 step 3, not yet swapped in.
