#!/usr/bin/env python3
"""
Phase 1, Step 2 — corpus cleaning.

Takes the raw filtered markdown (from filter_corpus.py) and produces
cleaned markdown ready for chunking:

  1. Admonition blocks (!!! question "title" / !!! example "title" / !!! info)
     are converted from MkDocs callout syntax into plain headed text, so the
     content survives but the syntax noise doesn't pollute embeddings.
  2. Markdown links [text](url) -> text (drops URLs; relative links break
     once files leave their original folder structure anyway).
  3. Code blocks (```...```) are left completely untouched — don't strip
     anything inside them, they need to stay intact for chunking.
  4. Blank-line normalization (collapse 3+ blank lines to 1).

Reads from:  ./autoware-rag-corpus/
Writes to:   ./autoware-rag-corpus-clean/
"""

import re
from pathlib import Path

SRC_ROOT = Path("autoware-rag-corpus")
DST_ROOT = Path("autoware-rag-corpus-clean")

CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
ADMONITION_HEADER = re.compile(r'^!!!\s+(\w+)\s*"([^"]*)"\s*$', re.MULTILINE)
# Images (including a linked/clickable image: [![alt](thumb)](url)) are dropped
# entirely — alt text on diagrams/video thumbnails isn't useful prose for
# embedding, and trying to preserve it correctly through nested brackets
# is not worth the complexity here.
MD_IMAGE_LINK = re.compile(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)")  # [![alt](thumb)](url)
MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # ![alt](url)
MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def protect_code_blocks(text):
    """Pull code blocks out and replace with placeholders so cleaning
    regexes never touch code content. Returns (text_with_placeholders, blocks)."""
    blocks = []

    def _stash(match):
        blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(blocks) - 1}__"

    return CODE_FENCE.sub(_stash, text), blocks


def restore_code_blocks(text, blocks):
    for i, block in enumerate(blocks):
        text = text.replace(f"__CODE_BLOCK_{i}__", block)
    return text


def clean_admonitions(text):
    """
    !!! question "What is X?"

        Indented body text here.
        More body text.

    becomes:

    **What is X?**

    Indented body text here.
    More body text.

    (dedents the body, keeps the title as a bold line so it's still
    visually distinct and semantically searchable)
    """
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^!!!\s+(\w+)\s*(?:"([^"]*)")?\s*$', line)
        if m:
            title = m.group(2)
            if title:
                out.append(f"**{title}**")
            out.append("")
            i += 1
            # consume the indented body block that follows
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].strip() == ""):
                body_line = lines[i]
                out.append(body_line[4:] if body_line.startswith("    ") else body_line)
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def clean_links(text):
    # Order matters: kill the nested [![alt](thumb)](url) pattern first,
    # then any remaining bare images, then loop plain links to convergence
    # (a single pass can miss links left partially exposed by a prior match).
    text = MD_IMAGE_LINK.sub("", text)
    text = MD_IMAGE.sub("", text)
    prev = None
    while prev != text:
        prev = text
        text = MD_LINK.sub(r"\1", text)
    return text


def normalize_blank_lines(text):
    return re.sub(r"\n{3,}", "\n\n", text)


def clean_file(text):
    text, code_blocks = protect_code_blocks(text)
    text = clean_admonitions(text)
    text = clean_links(text)
    text = normalize_blank_lines(text)
    text = restore_code_blocks(text, code_blocks)
    return text.strip() + "\n"


def main():
    if not SRC_ROOT.exists():
        raise SystemExit(f"Source not found at {SRC_ROOT}. Run filter_corpus.py first.")

    DST_ROOT.mkdir(exist_ok=True)

    md_files = list(SRC_ROOT.rglob("*.md"))
    for src in md_files:
        rel = src.relative_to(SRC_ROOT)
        dst = DST_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw = src.read_text(encoding="utf-8")
        cleaned = clean_file(raw)
        dst.write_text(cleaned, encoding="utf-8")

    print(f"Cleaned {len(md_files)} files -> {DST_ROOT}/")


if __name__ == "__main__":
    main()