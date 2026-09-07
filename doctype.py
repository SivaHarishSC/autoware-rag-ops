"""
doc_type classification (design vs interface vs other) + a fragile,
explicit keyword heuristic for detecting when a QUERY is asking about
one doc type specifically -- used as a rerank-stage boost, not a filter.
"""

DESIGN_PATH = "design/autoware-architecture-v1/components/"
INTERFACE_PATH = "design/autoware-architecture-v1/interfaces/"


def derive_doc_type(source_file):
    normalized = source_file.replace("\\", "/")
    if DESIGN_PATH in normalized:
        return "design"
    if INTERFACE_PATH in normalized:
        return "interface"
    return "other"


# Exact keyword list -- this is the whole heuristic, nothing hidden.
# Checked in this order: a query matching an INTERFACE phrase is
# classified "interface" even if it also happens to contain a DESIGN
# phrase (e.g. "not the component design doc" as a negation clause).
INTERFACE_PHRASES = [
    "interface spec",
    "interface doc",
    "per the interface",
    "as opposed to the design doc",
]
DESIGN_PHRASES = [
    "design doc",
    "design spec",
    "component design",
]


def detect_doc_type_hint(query):
    q = query.lower()
    if any(p in q for p in INTERFACE_PHRASES):
        return "interface"
    if any(p in q for p in DESIGN_PHRASES):
        return "design"
    return None


def apply_doc_type_boost(chunk_ids, scores, doc_type_by_id, hint):
    """scores aligned with chunk_ids (raw cross-encoder scores). Adds
    0.5x the score spread among THESE candidates to anything matching
    the hinted doc_type -- big enough to flip close calls, not big
    enough to guarantee an override of a much more relevant wrong-
    doctype candidate (that's the point of a boost over a hard filter:
    a wrong hint can't nuke the right answer if it's already winning
    decisively on relevance)."""
    if hint is None:
        return list(scores)
    spread = max(scores) - min(scores)
    boost = 0.5 * spread
    return [s + boost if doc_type_by_id.get(cid) == hint else s for cid, s in zip(chunk_ids, scores)]
