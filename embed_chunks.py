#!/usr/bin/env python3
"""
Phase 1, Step 4 — embed chunks.jsonl into Qdrant with BAAI/bge-large-en-v1.5.

chunks.jsonl stays the source of truth for what the generation model sees
(full table/diagram text). The model's 512-token window means anything over
EMBED_TOKEN_LIMIT gets a lighter EMBEDDING-ONLY sub-split here (tables:
row-groups with the header/separator repeated; code/diagrams: line-groups
with the fence repeated) so nothing gets silently truncated. Each sub-piece's
payload carries chunk_id pointing back at the original row in chunks.jsonl,
so retrieval always resolves to the full chunk.

Qdrant runs in embedded/local mode (QdrantClient(path=...), no server) —
at ~180 chunks a Docker daemon is pure overhead; local mode persists to
disk and needs nothing running in the background.
"""
import json
import re
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from chunk_corpus import count_tokens

CHUNKS_PATH = Path("chunks.jsonl")
QDRANT_PATH = Path("qdrant_data")
COLLECTION = "autoware_chunks"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBED_TOKEN_LIMIT = 450  # headroom below the model's 512 hard limit
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

TABLE_SEP_RE = re.compile(r"^\|[\s:|-]+\|$")


def is_markdown_table(text):
    lines = text.strip("\n").split("\n")
    return len(lines) >= 3 and lines[0].strip().startswith("|") and bool(TABLE_SEP_RE.match(lines[1].strip()))


def _group_by_budget(items, prefix, suffix, target_tokens):
    """Accumulate items (rows or lines) into prefix+items+suffix pieces
    capped at target_tokens. An item that alone busts the budget ships
    alone rather than being cut."""
    overhead = count_tokens(prefix + suffix)
    pieces, buf, buf_tokens = [], [], overhead

    def flush():
        if buf:
            pieces.append(prefix + "\n".join(buf) + suffix)

    for item in items:
        item_tokens = count_tokens(item)
        if buf and buf_tokens + item_tokens > target_tokens:
            flush()
            buf, buf_tokens = [], overhead
        buf.append(item)
        buf_tokens += item_tokens
    flush()
    return pieces


def split_for_embedding(text, target_tokens):
    lines = text.strip("\n").split("\n")

    if is_markdown_table(text):
        header, sep, rows = lines[0], lines[1], lines[2:]
        return _group_by_budget(rows, prefix=f"{header}\n{sep}\n", suffix="", target_tokens=target_tokens)

    if lines[0].strip().startswith("```"):
        fence_open, fence_close, body = lines[0], lines[-1], lines[1:-1]
        return _group_by_budget(body, prefix=f"{fence_open}\n", suffix=f"\n{fence_close}", target_tokens=target_tokens)

    # generic fallback for a large paragraph/list that isn't a table or code
    # block — split on blank lines, same accumulation strategy
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return _group_by_budget(paras, prefix="", suffix="", target_tokens=target_tokens)


def build_embedding_pieces(chunks):
    """Returns (texts, payloads) — one entry per embeddable piece."""
    texts, payloads = [], []
    for c in chunks:
        if c["tokens"] <= EMBED_TOKEN_LIMIT:
            pieces = [c["text"]]
        else:
            pieces = split_for_embedding(c["text"], EMBED_TOKEN_LIMIT)
        is_sub = len(pieces) > 1
        for i, piece in enumerate(pieces):
            texts.append(piece)
            payloads.append({
                "id_str": c["chunk_id"] if not is_sub else f"{c['chunk_id']}::sub{i}",
                "chunk_id": c["chunk_id"],
                "source_file": c["source_file"],
                "breadcrumb": c["breadcrumb"],
                "tokens": c["tokens"],
                "is_subpiece": is_sub,
                "doc_type": c["doc_type"],
            })
    return texts, payloads


def main():
    chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]
    texts, payloads = build_embedding_pieces(chunks)

    # verification: nothing over the model's hard limit went in unsplit
    oversized_originals = {c["chunk_id"] for c in chunks if c["tokens"] > EMBED_TOKEN_LIMIT}
    split_originals = {p["chunk_id"] for p in payloads if p["is_subpiece"]}
    unsplit_oversized = oversized_originals - split_originals
    still_too_big = [(p["id_str"], count_tokens(t)) for t, p in zip(texts, payloads) if count_tokens(t) > 512]

    print(f"chunks.jsonl rows: {len(chunks)}")
    print(f"embedding pieces (post sub-split): {len(texts)}")
    print(f"oversized original chunks (>{EMBED_TOKEN_LIMIT} tok): {len(oversized_originals)}, all sub-split: {not unsplit_oversized}")
    if unsplit_oversized:
        print(f"  !! NOT sub-split, would be truncated: {sorted(unsplit_oversized)}")
    if still_too_big:
        print(f"  !! pieces still over the 512 hard limit after sub-split: {still_too_big}")

    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)

    client = QdrantClient(path=str(QDRANT_PATH))
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
    )
    client.upload_points(
        COLLECTION,
        points=[PointStruct(id=i, vector=vectors[i].tolist(), payload=payloads[i]) for i in range(len(texts))],
    )
    print(f"stored {len(texts)} vectors in Qdrant collection '{COLLECTION}' at {QDRANT_PATH}")


if __name__ == "__main__":
    main()
