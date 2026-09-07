#!/usr/bin/env python3
"""
Phase 1 corpus filter — pulls a small, coherent subset of Autoware docs
(architecture + core concepts, no auto-generated API stub files) for the
first end-to-end RAG pipeline test.

Usage:
    python3 filter_corpus.py

Reads from:  ./autoware-documentation/docs/
Writes to:   ./autoware-rag-corpus/
"""

import shutil
from pathlib import Path

SRC_ROOT = Path("autoware-documentation/docs")
DST_ROOT = Path("autoware-rag-corpus")

# Explicit include list — every path is a real file confirmed to exist in
# the repo (checked via GitHub API before writing this). No globs into
# folders we haven't inspected, so nothing surprising sneaks in.
INCLUDE_FILES = [
    # Core concepts — richest explanatory prose, start here
    "design/autoware-concepts/index.md",
    "design/autoware-concepts/core-package-inclusion-criteria.md",
    "design/autoware-concepts/difference-from-ai-and-auto.md",
    # Top-level design docs
    "design/index.md",
    "design/repository-structure.md",
    "design/repos-files.md",
    "design/versioning-and-release.md",
    "design/autoware-system-capabilities.md",
    # Architecture v1 — component overviews (what each module does)
    "design/autoware-architecture-v1/index.md",
    "design/autoware-architecture-v1/components/control/index.md",
    "design/autoware-architecture-v1/components/localization/index.md",
    "design/autoware-architecture-v1/components/map/index.md",
    "design/autoware-architecture-v1/components/perception/index.md",
    "design/autoware-architecture-v1/components/planning/index.md",
    "design/autoware-architecture-v1/components/sensing/index.md",
    "design/autoware-architecture-v1/components/vehicle/index.md",
    # Architecture v1 — interface-level docs (how each module talks to others)
    "design/autoware-architecture-v1/interfaces/index.md",
    "design/autoware-architecture-v1/interfaces/components/index.md",
    "design/autoware-architecture-v1/interfaces/components/control.md",
    "design/autoware-architecture-v1/interfaces/components/localization.md",
    "design/autoware-architecture-v1/interfaces/components/map.md",
    "design/autoware-architecture-v1/interfaces/components/perception.md",
    "design/autoware-architecture-v1/interfaces/components/planning.md",
    "design/autoware-architecture-v1/interfaces/components/sensing.md",
    "design/autoware-architecture-v1/interfaces/components/vehicle-interface.md",
]


def main():
    if not SRC_ROOT.exists():
        raise SystemExit(
            f"Source not found at {SRC_ROOT}. "
            "Clone first: git clone --depth 1 "
            "https://github.com/autowarefoundation/autoware-documentation.git"
        )

    DST_ROOT.mkdir(exist_ok=True)

    copied, missing = [], []
    for rel_path in INCLUDE_FILES:
        src = SRC_ROOT / rel_path
        if not src.exists():
            missing.append(rel_path)
            continue
        dst = DST_ROOT / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel_path)

    print(f"Copied {len(copied)} files to {DST_ROOT}/")
    if missing:
        print(f"\nWARNING — {len(missing)} paths not found (repo may have moved):")
        for m in missing:
            print(f"  - {m}")

    total_chars = sum((DST_ROOT / p).stat().st_size for p in copied)
    print(f"\nTotal corpus size: {total_chars:,} bytes (~{total_chars // 4:,} tokens, rough estimate)")


if __name__ == "__main__":
    main()