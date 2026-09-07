from embed_chunks import is_markdown_table, split_for_embedding, build_embedding_pieces, EMBED_TOKEN_LIMIT
from chunk_corpus import count_tokens

# table detection
table = "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
assert is_markdown_table(table)
assert not is_markdown_table("just text\n")

# a big table gets split into header-repeating pieces, each under budget
big_table = "| A | B |\n| --- | --- |\n" + "".join(f"| row{i} | {'x' * 200} |\n" for i in range(30))
assert count_tokens(big_table) > EMBED_TOKEN_LIMIT
pieces = split_for_embedding(big_table, EMBED_TOKEN_LIMIT)
assert len(pieces) > 1
for p in pieces:
    assert p.startswith("| A | B |\n| --- | --- |\n")
    assert count_tokens(p) <= EMBED_TOKEN_LIMIT

# a big fenced code block gets split with the fence repeated
big_code = "```mermaid\n" + "\n".join(f"node{i} --> node{i+1}" for i in range(200)) + "\n```"
assert count_tokens(big_code) > EMBED_TOKEN_LIMIT
pieces = split_for_embedding(big_code, EMBED_TOKEN_LIMIT)
assert len(pieces) > 1
for p in pieces:
    assert p.startswith("```mermaid\n") and p.endswith("\n```")

# build_embedding_pieces: small chunk stays whole, oversized chunk maps
# every sub-piece's chunk_id back to the original
chunks = [
    {"chunk_id": "f.md::chunk0", "source_file": "f.md", "breadcrumb": "A", "tokens": 50, "text": "small", "doc_type": "other"},
    {"chunk_id": "f.md::chunk1", "source_file": "f.md", "breadcrumb": "B", "tokens": count_tokens(big_table), "text": big_table, "doc_type": "other"},
]
texts, payloads = build_embedding_pieces(chunks)
assert texts[0] == "small" and payloads[0]["is_subpiece"] is False and payloads[0]["chunk_id"] == "f.md::chunk0"
sub_payloads = [p for p in payloads if p["chunk_id"] == "f.md::chunk1"]
assert len(sub_payloads) > 1
assert all(p["is_subpiece"] and p["chunk_id"] == "f.md::chunk1" for p in sub_payloads)

print("ok")
