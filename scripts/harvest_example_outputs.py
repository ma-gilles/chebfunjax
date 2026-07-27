#!/usr/bin/env python3
"""Harvest published MATLAB output blocks from chebfun.org example pages.

Each chebfun.org example page (``examples/<category>/<Stem>.html``) interleaves
``<pre class="mcode-input">`` (the MATLAB source a reader would type) with
``<pre class="mcode-output">`` (the numeric result MATLAB echoed back). The
output blocks are the reference for the EXAMPLE-SCRIPT OUTPUT-PARITY axis: the
numbers our ``examples/**/*.py`` ports print should reproduce them to the
precision MATLAB displayed.

This harvester discovers the page URL for each example script from its
``Original MATLAB Chebfun: <url>`` docstring line, fetches the page, extracts
the ordered output blocks, parses every numeric token (with the display
precision implied by its printed digits), and writes a per-example JSON
reference file. Reference files live under the audit scratch tree, NOT the repo
(they are downloaded artifacts, refreshable at any time).

Usage::

    python scripts/harvest_example_outputs.py --per-category 3 --limit 40
    python scripts/harvest_example_outputs.py approx/AAAApprox stats/Random
    python scripts/harvest_example_outputs.py --all

Numeric token model
--------------------
A printed number carries an implied absolute tolerance equal to half the place
value of its last shown digit -- "matches to displayed precision". For example
``0.0164`` is trusted to +/- 5e-5, ``3.141592653589793`` to +/- 5e-16, and the
mantissa of ``1.234e-05`` to +/- 5e-9. Integers are treated as exact. Complex
tokens ``a + bi`` are decomposed into their real and imaginary parts, each with
its own display tolerance. ``compare_outputs.py`` consumes these tolerances.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import html
import json
import os
import re
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
REFS_OUT = Path(
    os.environ.get(
        "REFS_OUTPUT_DIR",
        "/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/refs_output",
    )
)
EXAMPLE_URL_RE = re.compile(
    r"https?://(?:www\.)?chebfun\.org/examples/([\w-]+)/([\w-]+)\.html"
)
OUTPUT_BLOCK_RE = re.compile(
    r'<pre class="mcode-output">(.*?)</pre>', re.DOTALL
)
# Complex token first (so its parts are not mistaken for two bare reals), then
# bare reals. A real is an optionally-signed decimal with optional exponent.
_REAL = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"
COMPLEX_RE = re.compile(rf"({_REAL})\s*([-+])\s*(\d+\.?\d*(?:[eE][-+]?\d+)?)i")
REAL_RE = re.compile(_REAL)


@dataclasses.dataclass(frozen=True)
class ExampleRef:
    """An example script paired with its chebfun.org source page."""

    script: str  # repo-relative path to examples/**/*.py
    category: str  # chebfun.org category (from the URL, may differ from dir)
    stem: str
    url: str


def discover_mapping() -> list[ExampleRef]:
    """Map every example script that names a chebfun.org source page."""
    refs: list[ExampleRef] = []
    for script in sorted(glob.glob(str(PROJECT / "examples/**/*.py"), recursive=True)):
        text = Path(script).read_text(encoding="utf-8", errors="replace")
        match = EXAMPLE_URL_RE.search(text)
        if not match:
            continue
        refs.append(
            ExampleRef(
                script=os.path.relpath(script, PROJECT),
                category=match.group(1),
                stem=match.group(2),
                url=match.group(0),
            )
        )
    return refs


def select_tranche(
    refs: list[ExampleRef], per_category: int | None, limit: int | None
) -> list[ExampleRef]:
    """Take a spread across categories to prove the pipeline before full runs."""
    if per_category is None and limit is None:
        return refs
    chosen: list[ExampleRef] = []
    seen_per_cat: dict[str, int] = {}
    for ref in refs:
        if per_category is not None and seen_per_cat.get(ref.category, 0) >= per_category:
            continue
        chosen.append(ref)
        seen_per_cat[ref.category] = seen_per_cat.get(ref.category, 0) + 1
        if limit is not None and len(chosen) >= limit:
            break
    return chosen


def place_value_tol(token: str) -> float:
    """Half the place value of a token's last displayed digit (absolute tol)."""
    token = token.strip()
    exp = 0
    mantissa = token
    m = re.search(r"[eE]([-+]?\d+)", token)
    if m:
        exp = int(m.group(1))
        mantissa = token[: m.start()]
    if "." in mantissa:
        decimals = len(mantissa.split(".", 1)[1])
    else:
        decimals = 0
    place = 10.0 ** (exp - decimals)
    return 0.5 * place


def parse_numbers(block: str) -> list[dict]:
    """Extract ordered numeric tokens (reals + complex parts) from an output block."""
    numbers: list[dict] = []
    remainder = block
    # Complex tokens first; blank them out so their parts are not re-parsed.
    for m in COMPLEX_RE.finditer(block):
        re_tok, sign, im_tok = m.group(1), m.group(2), m.group(3)
        im_signed = ("-" if sign == "-" else "") + im_tok
        numbers.append(
            {
                "value": float(re_tok),
                "tol": place_value_tol(re_tok),
                "raw": re_tok,
                "part": "real",
            }
        )
        numbers.append(
            {
                "value": float(im_signed),
                "tol": place_value_tol(im_tok),
                "raw": im_signed + "i",
                "part": "imag",
            }
        )
        remainder = remainder.replace(m.group(0), " ", 1)
    for m in REAL_RE.finditer(remainder):
        tok = m.group(0)
        if tok in ("+", "-", "."):
            continue
        try:
            val = float(tok)
        except ValueError:
            continue
        numbers.append(
            {"value": val, "tol": place_value_tol(tok), "raw": tok, "part": "real"}
        )
    return numbers


def fetch_text(url: str) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "chebfunjax-output-parity/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, "replace")


def harvest(ref: ExampleRef) -> dict:
    """Fetch a page and build its reference record."""
    page = fetch_text(ref.url)
    blocks: list[dict] = []
    total_numbers = 0
    for raw in OUTPUT_BLOCK_RE.findall(page):
        text = html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()
        if not text:
            continue
        # Drop lines that carry no parity-relevant number: MATLAB warnings
        # (pixel counts, iteration numbers) and "Elapsed time is ... seconds"
        # timings (timings are explicitly out of scope for output parity).
        kept = [
            ln
            for ln in text.splitlines()
            if not ln.lstrip().startswith("Warning:")
            and "Elapsed time is" not in ln
        ]
        nums = parse_numbers("\n".join(kept))
        total_numbers += len(nums)
        blocks.append({"text": text, "numbers": nums})
    return {
        "script": ref.script,
        "category": ref.category,
        "stem": ref.stem,
        "url": ref.url,
        "n_output_blocks": len(blocks),
        "n_numbers": total_numbers,
        "blocks": blocks,
    }


def write_ref(record: dict) -> Path:
    out = REFS_OUT / record["category"] / f"{record['stem']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("examples", nargs="*", help="category/Stem selectors (optional)")
    ap.add_argument("--per-category", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="harvest every mapped script")
    args = ap.parse_args()

    refs = discover_mapping()
    if args.examples:
        want = set(args.examples)
        refs = [r for r in refs if f"{r.category}/{r.stem}" in want]
    elif not args.all:
        refs = select_tranche(refs, args.per_category, args.limit)

    print(f"harvesting {len(refs)} example pages -> {REFS_OUT}")
    n_ok = n_numeric = n_empty = n_err = 0
    for ref in refs:
        try:
            record = harvest(ref)
        except Exception as exc:  # network / parse failure -> record and continue
            n_err += 1
            print(f"  ERR  {ref.category}/{ref.stem}: {exc}")
            continue
        write_ref(record)
        n_ok += 1
        if record["n_numbers"]:
            n_numeric += 1
        else:
            n_empty += 1
        print(
            f"  ok   {ref.category}/{ref.stem}: "
            f"{record['n_output_blocks']} blocks, {record['n_numbers']} numbers"
        )
    print(
        f"\ndone: {n_ok} harvested "
        f"({n_numeric} with numeric output, {n_empty} numeric-empty), {n_err} errors"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
