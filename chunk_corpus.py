#!/usr/bin/env python3
"""
Phase 1, Step 3 — hybrid chunking.

Split on markdown headers first (chunk never crosses a header boundary),
then recursively fall back to paragraph-based splitting for any section
still over MAX_CHUNK_TOKENS. Code fences and tables are protected as
atomic units through both passes — never split internally, even if that
means a chunk exceeds the target size (a table row without its header
row is meaningless, so oversize is the lesser evil).

Each chunk carries: source file, header breadcrumb, token count.

Reads from:  ./autoware-rag-corpus-clean/
Writes to:   ./chunks.jsonl
"""

import json
import re
from pathlib import Path

from doctype import derive_doc_type

SRC_ROOT = Path("autoware-rag-corpus-clean")
OUT_PATH = Path("chunks.jsonl")

MAX_CHUNK_TOKENS = 600
MIN_CHUNK_TOKENS = 40  # below this, merge into the previous chunk instead of keeping a fragment

CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
TABLE_BLOCK = re.compile(
    r"(?:^\|.*\|[ \t]*\n)(?:^\|[ \t:-]+\|[ \t]*\n)(?:^\|.*\|[ \t]*\n?)+",
    re.MULTILINE,
)
HEADER_LINE = re.compile(r"^(#{1,6})[ \t]+(.*)$", re.MULTILINE)


def count_tokens(text):
    """~4 chars/token approx (good enough for chunk-size decisions, not exact).
    Collapses whitespace first so table-alignment padding doesn't inflate it."""
    collapsed = re.sub(r"[ \t]+", " ", text)
    return max(1, len(collapsed) // 4)


def protect_atomic_blocks(text):
    """Stash code fences and tables behind placeholders so no split logic
    can cut through them. Returns (text, blocks)."""
    blocks = []

    def _stash(match):
        blocks.append(match.group(0))
        return f"\x00BLOCK{len(blocks) - 1}\x00"

    text = CODE_FENCE.sub(_stash, text)
    text = TABLE_BLOCK.sub(_stash, text)
    return text, blocks


def restore_atomic_blocks(text, blocks):
    for i, block in enumerate(blocks):
        text = text.replace(f"\x00BLOCK{i}\x00", block)
    return text


def split_into_header_sections(text):
    """Returns [(breadcrumb, section_text), ...]. breadcrumb is e.g.
    "Planning component design > Detailed design > Supported features"."""
    matches = list(HEADER_LINE.finditer(text))
    if not matches:
        return [("(no header)", text)] if text.strip() else []

    sections = []
    stack = []  # (level, title) currently open

    preamble = text[: matches[0].start()].strip("\n")
    if preamble.strip():
        sections.append(("(no header)", preamble))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()

        stack = [s for s in stack if s[0] < level]
        stack.append((level, title))

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip("\n")

        sections.append((" > ".join(t for _, t in stack), body))

    return sections


def recursive_fallback_split(text, blocks, max_tokens):
    """Accumulates paragraphs into chunks up to max_tokens. A paragraph
    that's a protected-block placeholder ships alone if it's already over
    budget once restored, rather than being cut."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    buffer = []
    buffer_tokens = 0

    def flush():
        nonlocal buffer, buffer_tokens
        if buffer:
            chunks.append(restore_atomic_blocks("\n\n".join(buffer), blocks))
            buffer = []
            buffer_tokens = 0

    for para in paragraphs:
        if not para.strip():
            continue
        para_tokens = count_tokens(restore_atomic_blocks(para, blocks))

        if buffer_tokens + para_tokens > max_tokens and buffer:
            flush()

        buffer.append(para)
        buffer_tokens += para_tokens

        if para_tokens > max_tokens:
            flush()

    flush()
    return chunks


def has_real_content(text):
    stripped = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()
    return len(stripped) >= 10


def chunk_file(path, corpus_root):
    raw = path.read_text(encoding="utf-8")
    rel_path = str(path.relative_to(corpus_root))

    protected_text, blocks = protect_atomic_blocks(raw)
    sections = split_into_header_sections(protected_text)

    results = []
    for breadcrumb, body in sections:
        if not body.strip():
            continue
        restored_full = restore_atomic_blocks(body, blocks)
        tokens = count_tokens(restored_full)

        if tokens <= MAX_CHUNK_TOKENS:
            results.append({"breadcrumb": breadcrumb, "text": restored_full, "tokens": tokens})
        else:
            sub_chunks = recursive_fallback_split(body, blocks, MAX_CHUNK_TOKENS)
            for j, sc in enumerate(sub_chunks):
                results.append({
                    "breadcrumb": f"{breadcrumb} [part {j + 1}/{len(sub_chunks)}]",
                    "text": sc,
                    "tokens": count_tokens(sc),
                })

    # merge tiny trailing fragments into the previous chunk, but never into
    # one that's already over budget (e.g. a large atomic table)
    merged = []
    for r in results:
        can_merge = (
            merged
            and r["tokens"] < MIN_CHUNK_TOKENS
            and merged[-1]["tokens"] <= MAX_CHUNK_TOKENS
        )
        if can_merge:
            merged[-1]["text"] += "\n\n" + r["text"]
            merged[-1]["tokens"] += r["tokens"]
            merged[-1]["breadcrumb"] += f" + {r['breadcrumb']}"
        else:
            merged.append(r)

    merged = [r for r in merged if has_real_content(r["text"])]

    doc_type = derive_doc_type(rel_path)
    return [
        {
            "chunk_id": f"{rel_path}::chunk{idx}",
            "source_file": rel_path,
            "breadcrumb": r["breadcrumb"],
            "tokens": r["tokens"],
            "text": r["text"],
            "doc_type": doc_type,
        }
        for idx, r in enumerate(merged)
    ]


def main():
    if not SRC_ROOT.exists():
        raise SystemExit(f"Source not found at {SRC_ROOT}. Run clean_corpus.py first.")

    all_chunks = []
    for f in sorted(SRC_ROOT.rglob("*.md")):
        all_chunks.extend(chunk_file(f, SRC_ROOT))

    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    token_counts = [c["tokens"] for c in all_chunks]
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Token range: {min(token_counts)} - {max(token_counts)}")
    print(f"Mean tokens/chunk: {sum(token_counts) / len(token_counts):.0f}")
    over_budget = [c for c in all_chunks if c["tokens"] > MAX_CHUNK_TOKENS]
    print(f"Chunks over {MAX_CHUNK_TOKENS} tokens (expected only for un-splittable tables/code): {len(over_budget)}")
    for c in over_budget:
        print(f"  - {c['chunk_id']} ({c['tokens']} tok) — {c['breadcrumb']}")


if __name__ == "__main__":
    main()
