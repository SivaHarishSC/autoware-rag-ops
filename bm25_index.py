"""
Sparse (BM25) index over the full 178 chunks.jsonl rows.

Library: rank_bm25 (BM25Okapi) -- not Qdrant sparse vectors. Read
rank_bm25's source: BM25Okapi just does term-frequency/IDF math over
whatever token list you hand it (doc_len = len(tokens), no cap anywhere).
So the "does BM25 need sub-splitting like the dense embeddings did"
question is a genuine no: verified from the library, not assumed. Picking
rank_bm25 over Qdrant sparse vectors keeps the two retrieval systems
decoupled and the tokenization fully visible/tunable in plain Python,
which matters here since tokenization choice is called out as a
methodology decision.

Tokenization: lowercase + split on \\w+ (regex). No stemming, no
stopword removal -- flagging this as a deliberate simple baseline, not
an oversight.
"""
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path("chunks.jsonl")


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def load_bm25_index():
    """Returns (bm25, chunk_ids, chunks) -- chunk_ids[i] is the chunk_id
    for tokenized-corpus row i, so BM25 scores join back to chunks.jsonl."""
    chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]
    chunk_ids = [c["chunk_id"] for c in chunks]
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunk_ids, chunks


if __name__ == "__main__":
    bm25, chunk_ids, chunks = load_bm25_index()
    print(f"BM25 index built over {len(chunk_ids)} chunks (full chunks.jsonl, no sub-splitting)")
