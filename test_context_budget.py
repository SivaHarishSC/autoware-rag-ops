from chunk_corpus import count_tokens
from retrieve_and_generate import CONTEXT_TOKEN_BUDGET, assemble_capped_context

def chunk(cid, text):
    return {"chunk_id": cid, "breadcrumb": cid, "text": text, "tokens": count_tokens(text)}

# several small chunks under budget -> all included, no truncation
small = [chunk(f"c{i}", "short chunk text " * 5) for i in range(3)]
selected, meta = assemble_capped_context(small, budget_tokens=1000)
assert meta["included"] == 3 and not meta["truncated"]
assert meta["candidates"] == 3

# chunks that together exceed budget -> stop before the one that would tip over
big_chunks = [chunk(f"b{i}", "word " * 300) for i in range(5)]  # ~375 tok each
selected, meta = assemble_capped_context(big_chunks, budget_tokens=1000)
assert meta["included"] < 5  # didn't fit all 5
assert not meta["truncated"]  # none of these individually exceeds budget
assert sum(count_tokens(c["text"]) for c in selected) <= 1000

# a single oversized table chunk (bigger alone than the whole budget) -> truncated at a row boundary
table_rows = "\n".join(f"| row{i} | value{i} |" for i in range(200))
table_text = "| A | B |\n| --- | --- |\n" + table_rows
assert count_tokens(table_text) > 500
selected, meta = assemble_capped_context([chunk("table", table_text)], budget_tokens=500)
assert meta["included"] == 1 and meta["truncated"]
out_text = selected[0]["text"]
assert out_text.startswith("| A | B |\n| --- | --- |\n")
assert "[table truncated, showing first" in out_text
# never cuts mid-row: every content line (excluding header/sep/note) is a complete "| rowN | valueN |" row
for line in out_text.split("\n")[2:-1]:
    assert line.startswith("| row") and line.endswith("|")
note_line = out_text.rsplit("\n", 1)[1]
body_only = out_text.rsplit("\n", 1)[0]
assert count_tokens(body_only) <= 500  # body itself respects budget exactly
assert count_tokens(note_line) < 20  # note is a small fixed overhead on top, not unbounded

# a single oversized NON-table chunk -> truncated at a line boundary
long_text = "\n".join(f"This is line number {i} of the document." for i in range(200))
selected, meta = assemble_capped_context([chunk("doc", long_text)], budget_tokens=300)
assert meta["truncated"]
out_text = selected[0]["text"]
assert "[content truncated, showing first" in out_text
for line in out_text.split("\n")[:-1]:
    assert line.startswith("This is line number") or line == ""

# pluggable count_fn: a "real tokenizer" stand-in that inflates markup-
# heavy text more than chars/4 does must actually change what gets
# included -- this is the exact bug found with the mermaid diagram chunk,
# where chars/4 undercounted badly enough that nothing got dropped
def inflating_count_fn(text):
    return count_tokens(text) + text.count("<") * 5  # pretend markup tokenizes expensively

markup_chunk = chunk("markup", "plain text here " * 20 + "<font><b>tag soup</b></font>" * 20)
approx_tokens = count_tokens(markup_chunk["text"])
inflated_tokens = inflating_count_fn(markup_chunk["text"])
assert inflated_tokens > approx_tokens  # the stand-in really does diverge from the approximation

other = chunk("other", "filler " * 50)
budget = inflated_tokens - 10  # just under the REAL count, comfortably under the approx count
assert approx_tokens < budget  # chars/4 alone would think both chunks fit

_, meta_default = assemble_capped_context([markup_chunk, other], budget_tokens=budget)  # uses chars/4
_, meta_real = assemble_capped_context([markup_chunk, other], budget_tokens=budget, count_fn=inflating_count_fn)
assert meta_default["truncated"] is False and meta_default["included"] == 2  # approx counter misses the problem
assert meta_real["truncated"] is True  # real-tokenizer-style counter catches it

print("ok")
