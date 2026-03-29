#!/usr/bin/env python3
"""Audit visual parity between chebfunjax docs pages and Chebfun originals.

This script compares rendered guide/example pages on the live chebfunjax site
against their Chebfun counterparts by:

1. Fetching both pages.
2. Extracting headings and PNG image URLs in document order.
3. Comparing image counts.
4. Comparing paired images with lightweight perceptual metrics.

The first goal is to identify pages that are visibly wrong before regenerating
assets or patching plotting code.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps


PROJECT = Path(__file__).resolve().parent.parent
DOCS_EXAMPLES = PROJECT / "docs" / "examples"
GUIDE_BASE = "https://www.chebfun.org/docs/guide"
LIVE_BASE = "https://ma-gilles.github.io/chebfunjax"

PNG_RE = re.compile(r'<img[^>]+src="([^"]+\.png)"', re.IGNORECASE)
HEADING_RE = re.compile(r"<h([1-3])[^>]*>(.*?)</h\\1>", re.IGNORECASE | re.DOTALL)
CHEBFUN_EXAMPLE_RE = re.compile(
    r"https?://(?:www\.)?chebfun\.org/examples/([\w-]+)/([\w-]+)\.html"
)


@dataclass(frozen=True)
class PageSpec:
    kind: str
    slug: str
    live_url: str
    target_url: str


@dataclass(frozen=True)
class ImageMetrics:
    live_size: tuple[int, int]
    target_size: tuple[int, int]
    hash_distance: int
    rmse: float


@dataclass(frozen=True)
class PageAudit:
    spec: PageSpec
    live_headings: list[str]
    target_headings: list[str]
    live_images: list[str]
    target_images: list[str]
    compared: list[ImageMetrics]
    error: str | None = None

    @property
    def count_match(self) -> bool:
        return len(self.live_images) == len(self.target_images)

    @property
    def heading_match(self) -> bool:
        return self.live_headings[:5] == self.target_headings[:5]

    @property
    def mean_hash_distance(self) -> float:
        if not self.compared:
            return math.nan
        return sum(m.hash_distance for m in self.compared) / len(self.compared)

    @property
    def mean_rmse(self) -> float:
        if not self.compared:
            return math.nan
        return sum(m.rmse for m in self.compared) / len(self.compared)


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "chebfunjax-audit/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, "replace")


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "chebfunjax-audit/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return " ".join(text.split())


def extract_headings(html_text: str) -> list[str]:
    return [strip_html(m.group(2)) for m in HEADING_RE.finditer(html_text)]


def extract_png_urls(base_url: str, html_text: str) -> list[str]:
    urls: list[str] = []
    for src in PNG_RE.findall(html_text):
        url = urllib.parse.urljoin(base_url, html.unescape(src))
        urls.append(url)
    return urls


def average_hash(image: Image.Image, hash_size: int = 8) -> int:
    gray = ImageOps.grayscale(image).resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    pixels = list(gray.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | int(pixel >= avg)
    return bits


def bit_count(value: int) -> int:
    return value.bit_count()


def normalized_rmse(left: Image.Image, right: Image.Image, size: tuple[int, int] = (256, 256)) -> float:
    left_arr = ImageOps.grayscale(left).resize(size, Image.Resampling.BILINEAR)
    right_arr = ImageOps.grayscale(right).resize(size, Image.Resampling.BILINEAR)
    lhs = list(left_arr.getdata())
    rhs = list(right_arr.getdata())
    mse = sum((float(a) - float(b)) ** 2 for a, b in zip(lhs, rhs)) / len(lhs)
    return math.sqrt(mse) / 255.0


def compare_images(live_url: str, target_url: str) -> ImageMetrics:
    live_img = Image.open(BytesIO(fetch_bytes(live_url))).convert("RGB")
    target_img = Image.open(BytesIO(fetch_bytes(target_url))).convert("RGB")
    live_img = ImageOps.exif_transpose(live_img)
    target_img = ImageOps.exif_transpose(target_img)
    live_hash = average_hash(live_img)
    target_hash = average_hash(target_img)
    return ImageMetrics(
        live_size=live_img.size,
        target_size=target_img.size,
        hash_distance=bit_count(live_hash ^ target_hash),
        rmse=normalized_rmse(live_img, target_img),
    )


def iter_guides() -> list[PageSpec]:
    return [
        PageSpec(
            kind="guide",
            slug=f"guide{i:02d}",
            live_url=f"{LIVE_BASE}/guide/guide{i:02d}/",
            target_url=f"{GUIDE_BASE}/guide{i:02d}.html",
        )
        for i in range(1, 21)
    ]


def iter_examples() -> list[PageSpec]:
    specs: list[PageSpec] = []
    for md_path in sorted(DOCS_EXAMPLES.glob("**/*.md")):
        if md_path.name == "index.md":
            continue
        text = md_path.read_text(encoding="utf-8", errors="replace")
        match = CHEBFUN_EXAMPLE_RE.search(text)
        if not match:
            continue
        rel = md_path.relative_to(PROJECT / "docs").with_suffix("")
        specs.append(
            PageSpec(
                kind="example",
                slug=str(rel).replace("\\", "/"),
                live_url=f"{LIVE_BASE}/{rel.as_posix()}/",
                target_url=match.group(0),
            )
        )
    return specs


def audit_page(spec: PageSpec, image_limit: int | None) -> PageAudit:
    try:
        live_html = fetch_text(spec.live_url)
        target_html = fetch_text(spec.target_url)
        live_images = extract_png_urls(spec.live_url, live_html)
        target_images = extract_png_urls(spec.target_url, target_html)
        max_pairs = min(len(live_images), len(target_images))
        if image_limit is not None:
            max_pairs = min(max_pairs, image_limit)
        compared = [
            compare_images(live_images[i], target_images[i])
            for i in range(max_pairs)
        ]
        return PageAudit(
            spec=spec,
            live_headings=extract_headings(live_html),
            target_headings=extract_headings(target_html),
            live_images=live_images,
            target_images=target_images,
            compared=compared,
        )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return PageAudit(
            spec=spec,
            live_headings=[],
            target_headings=[],
            live_images=[],
            target_images=[],
            compared=[],
            error=str(exc),
        )


def write_csv(path: Path, audits: list[PageAudit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "kind",
                "slug",
                "live_url",
                "target_url",
                "live_image_count",
                "target_image_count",
                "count_match",
                "heading_match",
                "paired_images",
                "mean_hash_distance",
                "mean_rmse",
                "error",
            ]
        )
        for audit in audits:
            writer.writerow(
                [
                    audit.spec.kind,
                    audit.spec.slug,
                    audit.spec.live_url,
                    audit.spec.target_url,
                    len(audit.live_images),
                    len(audit.target_images),
                    audit.count_match,
                    audit.heading_match,
                    len(audit.compared),
                    f"{audit.mean_hash_distance:.3f}" if audit.compared else "",
                    f"{audit.mean_rmse:.6f}" if audit.compared else "",
                    audit.error or "",
                ]
            )


def print_summary(audits: list[PageAudit], limit: int) -> None:
    failed = [a for a in audits if a.error]
    count_mismatch = [a for a in audits if not a.error and not a.count_match]
    heading_mismatch = [a for a in audits if not a.error and not a.heading_match]
    print(f"Audited pages: {len(audits)}")
    print(f"Fetch failures: {len(failed)}")
    print(f"Image-count mismatches: {len(count_mismatch)}")
    print(f"Heading mismatches (first 5 headings): {len(heading_mismatch)}")
    print()
    ranked = sorted(
        [a for a in audits if not a.error],
        key=lambda a: (
            abs(len(a.live_images) - len(a.target_images)),
            a.mean_hash_distance if a.compared else -1.0,
            a.mean_rmse if a.compared else -1.0,
        ),
        reverse=True,
    )
    print(f"Top {min(limit, len(ranked))} mismatches:")
    for audit in ranked[:limit]:
        print(
            f"  {audit.spec.slug}: "
            f"images {len(audit.live_images)}/{len(audit.target_images)}, "
            f"mean_hash={audit.mean_hash_distance:.2f}, "
            f"mean_rmse={audit.mean_rmse:.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=("guides", "examples", "all"),
        default="guides",
        help="Which page set to audit.",
    )
    parser.add_argument(
        "--image-limit",
        type=int,
        default=3,
        help="Maximum number of image pairs to compare per page.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="How many worst pages to print.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT / "scripts" / "visual_parity_report.csv",
        help="Where to write the CSV report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs: list[PageSpec] = []
    if args.kind in ("guides", "all"):
        specs.extend(iter_guides())
    if args.kind in ("examples", "all"):
        specs.extend(iter_examples())
    audits = [audit_page(spec, args.image_limit) for spec in specs]
    write_csv(args.csv, audits)
    print_summary(audits, args.top)
    print()
    print(f"CSV report: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
