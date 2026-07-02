#!/usr/bin/env python3
"""Execute every Python code block in a docs markdown page, in order.

Verifies that the code shown to readers actually runs against the current
library. Blocks share one namespace (like a reader typing them into one
session). Non-python fenced blocks are skipped. Lines starting with
``>>>``/``...`` (doctest style) are unwrapped.

Usage:
    python scripts/run_page_snippets.py docs/guide/guide06.md [--verbose]

Exit code 0 if every block executes, 1 otherwise (each failing block is
reported with its index, first line, and the exception).
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def extract_python_blocks(text: str) -> list[str]:
    blocks = []
    for lang, body in FENCE_RE.findall(text):
        if lang.lower() not in ("python", "py"):
            continue
        lines = []
        for line in body.splitlines():
            if line.startswith(">>> ") or line.startswith("... "):
                lines.append(line[4:])
            elif line.strip() in (">>>", "..."):
                continue
            elif line.startswith(">>>"):
                lines.append(line[3:].lstrip())
            else:
                lines.append(line)
        blocks.append("\n".join(lines))
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("page", type=Path)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")

    blocks = extract_python_blocks(args.page.read_text())
    ns: dict = {"__name__": "__main__"}
    failures = 0
    for i, block in enumerate(blocks, 1):
        head = next((ln for ln in block.splitlines() if ln.strip()), "")
        try:
            exec(compile(block, f"{args.page.name}[block {i}]", "exec"), ns)
            if args.verbose:
                print(f"  ok  block {i:>2}: {head[:70]}")
        except Exception:
            failures += 1
            print(f"FAIL block {i:>2}: {head[:70]}")
            print("     " + traceback.format_exc().strip().splitlines()[-1])
    print(f"{args.page}: {len(blocks) - failures}/{len(blocks)} blocks execute")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
