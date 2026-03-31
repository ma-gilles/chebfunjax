#!/usr/bin/env python3
"""Sync example pages from the Chebfun originals.

This imports the page header and content from chebfun.org example pages,
rewrites image URLs for the chebfunjax docs layout, and downloads all
referenced PNG assets into docs/images/<category>.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import html
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent
DOCS_EXAMPLES = PROJECT / "docs" / "examples"
DOCS_IMAGES = PROJECT / "docs" / "images"
EXAMPLE_URL_RE = re.compile(
    r"https?://(?:www\.)?chebfun\.org/examples/([\w-]+)/([\w-]+)\.html"
)
GITHUB_BLOB_RE = re.compile(
    r"https?://github\.com/chebfun/examples/blob/master/([\w-]+)/([\w-]+)\.m"
)


@dataclasses.dataclass(frozen=True)
class ExampleReference:
    """Discovered Chebfun example page metadata."""

    category: str
    stem: str
    url: str
    title: str
    github_source_path: str | None = None


class FragmentExtractor(HTMLParser):
    """Extract raw HTML fragments that match a tag predicate."""

    def __init__(self, predicate):
        super().__init__(convert_charrefs=False)
        self.predicate = predicate
        self.depth = 0
        self.capturing = False
        self.fragments: list[str] = []

    def _attrs_dict(self, attrs):
        return {key: value for key, value in attrs}

    def handle_starttag(self, tag, attrs):
        if not self.capturing and self.predicate(tag, self._attrs_dict(attrs)):
            self.capturing = True
            self.depth = 1
            self.fragments.append(self.get_starttag_text())
            return
        if self.capturing:
            self.depth += 1
            self.fragments.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        if self.capturing:
            self.fragments.append(f"</{tag}>")
            self.depth -= 1
            if self.depth == 0:
                self.capturing = False
                self.fragments.append("\n")

    def handle_startendtag(self, tag, attrs):
        if self.capturing or self.predicate(tag, self._attrs_dict(attrs)):
            self.fragments.append(self.get_starttag_text())

    def handle_data(self, data):
        if self.capturing:
            self.fragments.append(data)

    def handle_entityref(self, name):
        if self.capturing:
            self.fragments.append(f"&{name};")

    def handle_charref(self, name):
        if self.capturing:
            self.fragments.append(f"&#{name};")

    def handle_comment(self, data):
        if self.capturing:
            self.fragments.append(f"<!--{data}-->")

    def get_html(self) -> str:
        return "".join(self.fragments).strip()


class LinkCollector(HTMLParser):
    """Collect href attributes from HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(href)


def normalize_key(text: str) -> str:
    """Return a comparison key for titles, stems, and source-path lookups."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def extract_markdown_title(text: str) -> str | None:
    """Extract the first Markdown H1 title from a document."""
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None


def extract_github_source_path(text: str) -> str | None:
    """Extract a Chebfun examples blob path like ``sphere/Foo.m``."""
    match = GITHUB_BLOB_RE.search(text)
    if not match:
        return None
    return f"{match.group(1)}/{match.group(2)}.m"


def _clean_html_text(fragment: str) -> str:
    """Strip tags from an HTML fragment while preserving readable text."""
    fragment = html.unescape(fragment)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    return " ".join(fragment.split())


def extract_page_title(html_text: str) -> str:
    """Extract the first ``<h1>`` title from a Chebfun example page."""
    match = re.search(r"<h1[^>]*>(.*?)</h1>", html_text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return _clean_html_text(match.group(1))


@functools.lru_cache(maxsize=None)
def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "chebfunjax-example-sync/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, "replace")


@functools.lru_cache(maxsize=None)
def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "chebfunjax-example-sync/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def extract_header_and_content(html_text: str) -> tuple[str, str]:
    header = FragmentExtractor(
        lambda tag, attrs: tag == "div" and attrs.get("class") == "page-header"
    )
    content = FragmentExtractor(
        lambda tag, attrs: tag == "div" and attrs.get("id") == "content"
    )
    header.feed(html_text)
    content.feed(html_text)
    return header.get_html(), content.get_html()


def rewrite_fragment(fragment: str, category: str) -> str:
    fragment = html.unescape(fragment)
    # Example pages build to /examples/<category>/<slug>/, so images live three levels up at /images/<category>/.
    fragment = fragment.replace('src="img/', f'src="../../../images/{category}/')
    fragment = fragment.replace("src='img/", f"src='../../../images/{category}/")
    fragment = fragment.replace('class="figure"', 'class="figure chebfun-figure"')
    fragment = fragment.replace("class='figure'", "class='figure chebfun-figure'")
    fragment = fragment.replace(f'href="/examples/{category}/"', 'href="../"')
    fragment = fragment.replace(f"href='/examples/{category}/'", "href='../'")
    return fragment


def extract_image_urls(page_url: str, html_text: str) -> list[str]:
    return [
        urllib.parse.urljoin(page_url, src)
        for src in re.findall(r'<img[^>]+src="([^"]+\.png)"', html_text, re.IGNORECASE)
    ]


def extract_example_page_urls(category: str, html_text: str) -> list[str]:
    """Collect page URLs from a Chebfun category index page."""
    parser = LinkCollector()
    parser.feed(html_text)

    urls: list[str] = []
    seen: set[str] = set()
    for href in parser.hrefs:
        url = urllib.parse.urljoin(f"https://www.chebfun.org/examples/{category}/", href)
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc != "www.chebfun.org":
            continue
        if not parsed.path.startswith(f"/examples/{category}/"):
            continue
        if not parsed.path.endswith(".html"):
            continue
        if parsed.path.endswith("/index.html"):
            continue
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def build_example_catalog(categories: set[str]) -> dict[str, object]:
    """Build lookup tables for discovering Chebfun example source URLs."""
    refs: list[ExampleReference] = []
    by_github_source: dict[str, str] = {}
    by_title: dict[tuple[str, str], list[str]] = {}
    by_stem: dict[tuple[str, str], list[str]] = {}

    for category in sorted(categories):
        index_html = fetch_text(f"https://www.chebfun.org/examples/{category}/")
        page_urls = extract_example_page_urls(category, index_html)
        for page_url in page_urls:
            page_html = fetch_text(page_url)
            stem = Path(urllib.parse.urlparse(page_url).path).stem
            title = extract_page_title(page_html)
            github_source_path = extract_github_source_path(page_html)
            ref = ExampleReference(
                category=category,
                stem=stem,
                url=page_url,
                title=title,
                github_source_path=github_source_path,
            )
            refs.append(ref)
            if github_source_path is not None:
                by_github_source[github_source_path.lower()] = page_url
            by_title.setdefault((category, normalize_key(title)), []).append(page_url)
            by_stem.setdefault((category, normalize_key(stem)), []).append(page_url)

    return {
        "refs": refs,
        "by_github_source": by_github_source,
        "by_title": by_title,
        "by_stem": by_stem,
    }


def write_markdown(path: Path, header_html: str, content_html: str, source_url: str) -> None:
    body = "\n\n".join(part for part in [header_html, content_html] if part)
    text = (
        "<!-- Generated by scripts/sync_chebfun_examples.py. -->\n"
        f"<!-- Source: {source_url} -->\n\n"
        '<div class="chebfun-import chebfun-example-import">\n'
        f"{body}\n"
        "</div>\n"
    )
    path.write_text(text, encoding="utf-8")


def download_images(category: str, image_urls: list[str]) -> None:
    out_dir = DOCS_IMAGES / category
    out_dir.mkdir(parents=True, exist_ok=True)
    for url in image_urls:
        dest = out_dir / Path(urllib.parse.urlparse(url).path).name
        dest.write_bytes(fetch_bytes(url))


def discover_source_url(doc_path: Path, catalog: dict[str, object]) -> str | None:
    """Infer the Chebfun example URL for a local doc lacking a source marker."""
    text = doc_path.read_text(encoding="utf-8", errors="replace")
    category = doc_path.parent.name

    github_source_path = extract_github_source_path(text)
    if github_source_path is not None:
        url = catalog["by_github_source"].get(github_source_path.lower())
        if url is not None:
            return url

    title = extract_markdown_title(text)
    if title is not None:
        title_urls = catalog["by_title"].get((category, normalize_key(title)), [])
        if len(title_urls) == 1:
            return title_urls[0]

    stem_urls = catalog["by_stem"].get((category, normalize_key(doc_path.stem)), [])
    if len(stem_urls) == 1:
        return stem_urls[0]

    return None


def original_url_for(doc_path: Path, catalog: dict[str, object] | None = None) -> str:
    text = doc_path.read_text(encoding="utf-8", errors="replace")
    match = EXAMPLE_URL_RE.search(text)
    if not match:
        if catalog is None:
            raise ValueError(f"No Chebfun example URL found in {doc_path}")
        discovered = discover_source_url(doc_path, catalog)
        if discovered is None:
            raise ValueError(f"No Chebfun example URL found in {doc_path}")
        return discovered
    return match.group(0)


def sync_example(doc_path: Path, catalog: dict[str, object] | None = None) -> None:
    category = doc_path.parent.name
    source_url = original_url_for(doc_path, catalog=catalog)
    html_text = fetch_text(source_url)
    header_html, content_html = extract_header_and_content(html_text)
    header_html = rewrite_fragment(header_html, category)
    content_html = rewrite_fragment(content_html, category)
    write_markdown(doc_path, header_html, content_html, source_url)
    download_images(category, extract_image_urls(source_url, html_text))
    print(f"synced {doc_path.relative_to(PROJECT)}")


def iter_example_docs(paths: list[str]) -> list[Path]:
    if paths:
        return [DOCS_EXAMPLES / f"{path}.md" for path in paths]
    return sorted(path for path in DOCS_EXAMPLES.glob("*/*.md") if path.name != "index.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "examples",
        nargs="*",
        help="Example paths like approx/AAAApprox. Defaults to all example docs.",
    )
    args = parser.parse_args()

    doc_paths = iter_example_docs(args.examples)
    catalog = build_example_catalog({doc_path.parent.name for doc_path in doc_paths})
    for doc_path in doc_paths:
        try:
            sync_example(doc_path, catalog=catalog)
        except ValueError as exc:
            print(f"skip {doc_path.relative_to(PROJECT)}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
