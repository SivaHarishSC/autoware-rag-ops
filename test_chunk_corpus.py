from chunk_corpus import split_into_header_sections, protect_atomic_blocks, restore_atomic_blocks

# preamble before first header must not be dropped
text = "Intro line\n\n# Header\n\nBody text\n"
sections = split_into_header_sections(text)
assert sections[0] == ("(no header)", "Intro line"), sections[0]
assert sections[1] == ("Header", "Body text"), sections[1]

# no headers at all -> whole doc is one section
assert split_into_header_sections("just text\n") == [("(no header)", "just text\n")]

# doc starting with a header has no spurious preamble section
sections2 = split_into_header_sections("# Title\n\nBody\n")
assert sections2[0][0] == "Title" and len(sections2) == 1, sections2

# a header line inside a code fence must not split the doc
raw = "# Title\n\n```\n# not a header\n```\n\n## Real\n\nBody\n"
protected, blocks = protect_atomic_blocks(raw)
sections3 = split_into_header_sections(protected)
assert [b for b, _ in sections3] == ["Title", "Title > Real"], sections3
assert restore_atomic_blocks(sections3[0][1], blocks) == "```\n# not a header\n```"

print("ok")
