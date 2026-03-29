#!/usr/bin/env python3
"""Audit guide/example page parity against the Chebfun reference site.

This script compares the local docs sources with their Chebfun originals by:

1. section headings
2. referenced image count
3. image file existence

It is intended to drive page-by-page fixing of guide/example content and
plot-generation scripts without relying on manual browsing alone.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


PROJECT = Path("/scratch/gpfs/GILLES/mg6942/jaxchebfun")
DOCS_DIR = PROJECT / "docs"

GUIDE_REF_BASE = "https://www.chebfun.org/docs/guide"
GUIDE_LOCAL_BASE = DOCS_DIR / "guide"

HTML_TIMEOUT = 30
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+\.(?:png|jpg|jpeg|gif|svg))\)")
LINK_RE = re.compile(r"https?://[^\s)]+")


class _SectionParser(HTMLParser):
    """Extract headings and image URLs from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self._current_heading: str | None = None
        self._heading_chunks: list[str] = []
        self.headings: list[str] = []
        self.images: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_dict = dict(attrs)
        if tag in {"h1", "h2", "h3", "h4"}:
            self._current_heading = tag
            self._heading_chunks = []
        elif tag == "img":
            src = attrs_dict.get("src", "")
            if src:
                self.images.append(src)

    def handle_endtag(self, tag: str) -> None:
        if self._current_heading == tag:
            text = " ".join(" ".join(self._heading_chunks).split())
            if text:
                self.headings.append(text)
            self._current_heading = None
            self._heading_chunks = []

    def handle_data(self, data: str) -> None:
        if self._current_heading is not None:
            self._heading_chunks.append(data)


@dataclass
class PageAudit:
    page_id: str
    local_doc: str
    reference_url: str
    script_path: str
    script_exists: bool
    script_uses_plotting_api: bool
    script_uses_direct_matplotlib: bool
    local_headings: list[str]
    reference_headings: list[str]
    local_images: list[str]
    reference_images: list[str]
    missing_local_images: list[str]

    @property
    def heading_count_match(self) -> bool:
        return len(self.local_headings) == len(self.reference_headings)

    @property
    def image_count_match(self) -> bool:
        return len(self.local_images) == len(self.reference_images)

    @property
    def heading_mismatches(self) -> list[dict[str, str]]:
        mismatches = []
        for idx, (local, ref) in enumerate(zip(self.local_headings, self.reference_headings), start=1):
            if _norm_heading(local) != _norm_heading(ref):
                mismatches.append(
                    {
                        "index": str(idx),
                        "local": local,
                        "reference": ref,
                    }
                )
        return mismatches

    @property
    def ok(self) -> bool:
        return (
            self.heading_count_match
            and self.image_count_match
            and not self.heading_mismatches
            and not self.missing_local_images
        )


def _norm_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _fetch_text(url: str) -> str:
    with urlopen(url, timeout=HTML_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_markdown(doc_path: Path) -> tuple[list[str], list[str], list[str]]:
    text = doc_path.read_text(encoding="utf-8", errors="replace")
    headings = []
    images = []
    links = []

    for line in text.splitlines():
        if line.startswith("#"):
            headings.append(line.lstrip("#").strip())

    images.extend(IMG_RE.findall(text))
    links.extend(LINK_RE.findall(text))
    return headings, images, links


def _parse_reference_html(url: str) -> tuple[list[str], list[str]]:
    parser = _SectionParser()
    parser.feed(_fetch_text(url))
    return parser.headings, parser.images


def _resolve_images(doc_path: Path, image_refs: Iterable[str]) -> list[str]:
    missing = []
    for ref in image_refs:
        img_path = (doc_path.parent / ref).resolve()
        if not img_path.exists():
            missing.append(ref)
    return missing


def _guide_reference_url(doc_path: Path) -> str:
    return f"{GUIDE_REF_BASE}/{doc_path.stem}.html"


def _example_reference_url(doc_path: Path, links: list[str]) -> str:
    for link in links:
        if "chebfun.org/examples/" in link and link.endswith(".html"):
            return link
    raise ValueError(f"No chebfun.org example link found in {doc_path}")


def _script_path(doc_path: Path, kind: str) -> Path:
    if kind == "guide":
        return PROJECT / "scripts" / f"generate_{doc_path.stem}_plots.py"
    return PROJECT / "examples" / doc_path.parent.name / f"{doc_path.stem}.py"


def _script_audit(script_path: Path) -> tuple[bool, bool, bool]:
    if not script_path.exists():
        return False, False, False

    text = script_path.read_text(encoding="utf-8", errors="replace")
    uses_plotting_api = (
        "import chebfunjax as cj" in text
        or "from chebfunjax.plotting import" in text
        or "cj.plot(" in text
        or "plot_disk(" in text
        or "plot_sphere(" in text
        or "contour_sphere(" in text
        or "contour_disk(" in text
        or "quiver_sphere(" in text
        or "quiver_disk(" in text
        or "plot_ball_slices(" in text
        or "plot_chebfun3(" in text
    )
    uses_direct_matplotlib = any(
        token in text
        for token in (
            "plot_surface(",
            "pcolormesh(",
            ".contour(",
            ".quiver(",
            ".imshow(",
            "fig.colorbar(",
            "plt.colorbar(",
        )
    )
    return True, uses_plotting_api, uses_direct_matplotlib


def audit_doc(doc_path: Path, kind: str) -> PageAudit:
    local_headings, local_images, links = _parse_markdown(doc_path)
    if kind == "guide":
        ref_url = _guide_reference_url(doc_path)
    else:
        ref_url = _example_reference_url(doc_path, links)

    ref_headings, ref_images = _parse_reference_html(ref_url)
    missing_local_images = _resolve_images(doc_path, local_images)
    page_id = doc_path.stem if kind == "guide" else f"{doc_path.parent.name}/{doc_path.stem}"
    script_path = _script_path(doc_path, kind)
    script_exists, script_uses_plotting_api, script_uses_direct_matplotlib = _script_audit(script_path)
    return PageAudit(
        page_id=page_id,
        local_doc=str(doc_path),
        reference_url=ref_url,
        script_path=str(script_path),
        script_exists=script_exists,
        script_uses_plotting_api=script_uses_plotting_api,
        script_uses_direct_matplotlib=script_uses_direct_matplotlib,
        local_headings=local_headings,
        reference_headings=ref_headings,
        local_images=local_images,
        reference_images=ref_images,
        missing_local_images=missing_local_images,
    )


def iter_docs(kind: str, page: str | None) -> list[Path]:
    if kind == "guide":
        base = GUIDE_LOCAL_BASE
        if page:
            return [base / f"{page}.md"]
        return sorted(base.glob("guide*.md"))

    base = DOCS_DIR / "examples"
    if page:
        category, name = page.split("/", 1)
        return [base / category / f"{name}.md"]
    return sorted(path for path in base.glob("*/*.md") if path.name != "index.md")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["guide", "example"], required=True)
    parser.add_argument(
        "--page",
        help="Guide page stem like guide20 or example page path like sphere/helmholtz_decomposition",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    docs = iter_docs(args.kind, args.page)
    if not docs:
        raise SystemExit("No matching docs pages found.")

    audits: list[PageAudit] = []
    errors: list[dict[str, str]] = []
    for doc in docs:
        try:
            audits.append(audit_doc(doc, args.kind))
        except (ValueError, HTTPError, URLError) as exc:
            errors.append({"page": str(doc), "error": str(exc)})

    if args.json:
        payload = {
            "kind": args.kind,
            "audits": [asdict_with_derived(audit) for audit in audits],
            "errors": errors,
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0 if not errors else 1

    print(f"Audited {len(audits)} {args.kind} page(s)")
    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors[:10]:
            print(f"  ERROR {err['page']}: {err['error']}")

    worst = sorted(
        audits,
        key=lambda audit: (
            audit.ok,
            audit.image_count_match,
            audit.heading_count_match,
            -len(audit.heading_mismatches),
            -abs(len(audit.local_images) - len(audit.reference_images)),
        ),
    )
    for audit in worst[:20]:
        print()
        print(audit.page_id)
        print(f"  local doc: {audit.local_doc}")
        print(f"  reference: {audit.reference_url}")
        print(
            "  script:"
            f" exists={audit.script_exists}"
            f" plotting_api={audit.script_uses_plotting_api}"
            f" direct_matplotlib={audit.script_uses_direct_matplotlib}"
            f" path={audit.script_path}"
        )
        print(
            "  headings:"
            f" local={len(audit.local_headings)}"
            f" ref={len(audit.reference_headings)}"
            f" mismatches={len(audit.heading_mismatches)}"
        )
        print(
            "  images:"
            f" local={len(audit.local_images)}"
            f" ref={len(audit.reference_images)}"
            f" missing_local={len(audit.missing_local_images)}"
        )
        if audit.heading_mismatches:
            first = audit.heading_mismatches[0]
            print(
                "  first heading mismatch:"
                f" local='{first['local']}'"
                f" ref='{first['reference']}'"
            )
    return 0 if not errors else 1


def asdict_with_derived(audit: PageAudit) -> dict[str, object]:
    payload = asdict(audit)
    payload["heading_count_match"] = audit.heading_count_match
    payload["image_count_match"] = audit.image_count_match
    payload["heading_mismatches"] = audit.heading_mismatches
    payload["ok"] = audit.ok
    return payload


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
