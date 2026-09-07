from doctype import apply_doc_type_boost, derive_doc_type, detect_doc_type_hint

assert derive_doc_type(r"design\autoware-architecture-v1\components\control\index.md") == "design"
assert derive_doc_type(r"design\autoware-architecture-v1\interfaces\components\control.md") == "interface"
assert derive_doc_type(r"design\repository-structure.md") == "other"

assert detect_doc_type_hint("Per the interface spec, what does Localization output?") == "interface"
assert detect_doc_type_hint("Per the component design, what are Planning's goals?") == "design"
assert detect_doc_type_hint("What is microautonomy architecture?") is None
# a query naming both should resolve to interface (checked first) since
# the "not the X doc" phrasing names the doc type being excluded
assert detect_doc_type_hint("Per the interface spec (not the component design doc), X?") == "interface"
# a component literally named "Interface" isn't the same as saying
# "interface spec/doc" -- known miss, not a false positive to hide
assert detect_doc_type_hint("What does the Vehicle Interface component send back?") is None

chunk_ids = ["a", "b", "c"]
doc_type_by_id = {"a": "design", "b": "interface", "c": "other"}

# no hint -> scores unchanged
assert apply_doc_type_boost(chunk_ids, [1.0, 2.0, 3.0], doc_type_by_id, None) == [1.0, 2.0, 3.0]

# hint boosts the matching candidate but doesn't touch the others
boosted = apply_doc_type_boost(chunk_ids, [1.0, 2.0, 3.0], doc_type_by_id, "design")
assert boosted[1] == 2.0 and boosted[2] == 3.0  # unmatched untouched
assert boosted[0] > 1.0  # matched candidate boosted

# boost can flip a close call (target trails the top candidate by only
# 0.1, well within the boost, even though a third low-scoring candidate
# widens the overall spread the boost is computed from)...
close = apply_doc_type_boost(
    ["low", "target", "top"], [0.0, 5.0, 5.1], {"low": "other", "target": "design", "top": "other"}, "design"
)
assert close[1] > close[2]  # target now beats top

# ...but can't override a decisively higher-scoring wrong-doctype candidate
decisive = apply_doc_type_boost(["a", "b"], [1.0, 100.0], {"a": "design", "b": "other"}, "design")
assert decisive[1] > decisive[0]

print("ok")
