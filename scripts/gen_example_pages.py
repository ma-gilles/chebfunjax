#!/usr/bin/env python3
"""Regenerate docs/examples/<cat>/<Stem>.md as a replica of the chebfun.org page.

The page prose, section headings, MATLAB code cells and figure placement
are taken from the published chebfun.org HTML (cached under --html-cache);
the printed outputs come from OUR translation's stdout (one file per
script under --stdout-dir, written by the regeneration runner) and the
figures are our docs/images/<cat>/<Stem>_NN.png renders.  Output blocks
are aligned with the MATLAB page by their first line (``ans =``,
``minf =`` ...); when a label is not found in our stdout the reference
block's line count is consumed instead.

Usage
-----
    python scripts/gen_example_pages.py --stdout-dir DIR [cat/Stem ...]
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DOCS = PROJECT / "docs" / "examples"
IMAGES = PROJECT / "docs" / "images"
EXAMPLES = PROJECT / "examples"
URL_RE = re.compile(r"https?://(?:www\.)?chebfun\.org/examples/([\w-]+)/([\w-]+)\.html")


def fetch_html(url: str, cache: Path) -> str:
    cache.mkdir(parents=True, exist_ok=True)
    m = URL_RE.search(url)
    dest = cache / f"{m.group(1)}_{m.group(2)}.html"
    if dest.exists():
        return dest.read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(url, headers={"User-Agent": "chebfunjax-page-gen/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode("utf-8", "replace")
    dest.write_text(text, encoding="utf-8")
    return text


class _Node:
    def __init__(self, tag, attrs):
        self.tag = tag
        self.attrs = dict(attrs)
        self.children: list = []   # str or _Node


class _TreeParser(HTMLParser):
    """Build a tree of the <div id="content"> element."""

    VOID = {"img", "br", "hr", "meta", "link", "input"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = None
        self.stack: list[_Node] = []
        self.depth_in = 0

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if self.root is None:
            if tag == "div" and d.get("id") == "content":
                self.root = _Node(tag, attrs)
                self.stack = [self.root]
            return
        if not self.stack:
            return                # past the end of the content div
        node = _Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        if self.root is None or not self.stack:
            return
        self.stack[-1].children.append(_Node(tag, attrs))

    def handle_endtag(self, tag):
        if self.root is None or tag in self.VOID or not self.stack:
            return
        if len(self.stack) > 1 and self.stack[-1].tag == tag:
            self.stack.pop()
        elif len(self.stack) == 1 and tag == "div":
            self.stack = []      # closed the content div

    def handle_data(self, data):
        if self.root is not None and self.stack:
            self.stack[-1].children.append(data)


def _inline(node, images: list) -> str:
    """Inline markdown of a node's children."""
    out = []
    for ch in node.children:
        if isinstance(ch, str):
            out.append(ch)
            continue
        t = ch.tag
        if t == "code":
            out.append("`" + _text(ch) + "`")
        elif t in ("em", "i"):
            out.append("*" + _inline(ch, images) + "*")
        elif t in ("strong", "b"):
            out.append("**" + _inline(ch, images) + "**")
        elif t == "a":
            out.append(f"[{_inline(ch, images)}]({_link(ch.attrs.get('href', ''))})")
        elif t == "img":
            images.append(ch.attrs.get("src", ""))
            out.append(f"@@IMG{len(images) - 1}@@")
        elif t == "br":
            out.append("\n")
        elif t in ("sub", "sup"):
            out.append(f"<{t}>{_inline(ch, images)}</{t}>")
        else:
            out.append(_inline(ch, images))
    return "".join(out)


_CURRENT = {"cat": ""}


def _link(href: str) -> str:
    """chebfun.org-relative links: to our page when it exists, else absolute."""
    import urllib.parse
    if not href or href.startswith("#") or href.startswith("mailto:"):
        return href
    base = f"https://www.chebfun.org/examples/{_CURRENT['cat']}/"
    url = urllib.parse.urljoin(base, href)
    m = URL_RE.match(url)
    if m and (DOCS / m.group(1) / f"{m.group(2)}.md").exists():
        tgt = f"{m.group(1)}/{m.group(2)}.md"
        return tgt.split("/", 1)[1] if m.group(1) == _CURRENT["cat"] else f"../{tgt}"
    return url


def _text(node) -> str:
    return "".join(ch if isinstance(ch, str) else _text(ch) for ch in node.children)


def _clean_para(s: str) -> str:
    s = re.sub(r"[ \t]*\n[ \t]*", " ", s.strip())
    return re.sub(r"  +", " ", s)


def render(root, cat: str, stem: str, ours: list[str] | None):
    """Walk the content tree, returning markdown; outputs come from ``ours``."""
    parts: list[str] = []
    images: list[str] = []
    ref_blocks: list[list[str]] = []
    pending: list[tuple[int, int]] = []     # (index in parts, block index)

    def img_md(src):
        name = os.path.basename(src)
        label = name[len(stem) + 1:-4] if name.startswith(stem) else name
        if not (IMAGES / cat / name).exists():
            return f"*(Figure {label} of the original page is not reproduced yet.)*"
        return f"![{stem} figure {label}](../../images/{cat}/{name})"

    def emit_inline(node):
        imgs: list = []
        text = _inline(node, imgs)
        for k, src in enumerate(imgs):
            text = text.replace(f"@@IMG{k}@@", img_md(src))
            images.append(src)
        return text

    def walk(node):
        for ch in node.children:
            if isinstance(ch, str):
                if ch.strip():
                    parts.append(_clean_para(ch))
                continue
            t = ch.tag
            cls = ch.attrs.get("class", "")
            if t in ("h1", "h2", "h3", "h4"):
                level = {"h1": "#", "h2": "##", "h3": "##", "h4": "###"}[t]
                parts.append(f"{level} {_clean_para(emit_inline(ch))}")
            elif t == "p":
                txt = emit_inline(ch)
                if txt.strip():
                    parts.append(_clean_para(txt) if not txt.lstrip().startswith("![") else txt.strip())
            elif t == "pre" and "mcode-input" in cls:
                parts.append("```matlab\n" + _text(ch).rstrip("\n") + "\n```")
            elif t == "pre" and "mcode-output" in cls:
                ref_blocks.append(_text(ch).rstrip("\n").split("\n"))
                pending.append((len(parts), len(ref_blocks) - 1))
                parts.append(None)
            elif t == "pre":
                parts.append("```\n" + _text(ch).rstrip("\n") + "\n```")
            elif t in ("ul", "ol"):
                items = []
                for k, li in enumerate(c for c in ch.children if not isinstance(c, str)):
                    bullet = "-" if t == "ul" else f"{k + 1}."
                    items.append(f"{bullet} {_clean_para(emit_inline(li))}")
                parts.append("\n".join(items))
            elif t == "blockquote":
                parts.append("\n".join("> " + ln for ln in _clean_para(emit_inline(ch)).split("\n")))
            elif t == "table":
                parts.append("```\n" + _text(ch).strip() + "\n```")
            elif t == "img":
                images.append(ch.attrs.get("src", ""))
                parts.append(img_md(ch.attrs.get("src", "")))
            elif t == "div":
                walk(ch)
            else:
                txt = emit_inline(ch)
                if txt.strip():
                    parts.append(_clean_para(txt))

    walk(root)
    chunks = align_outputs(ref_blocks, ours)
    for idx, k in pending:
        if ours is not None and not chunks[k] and ref_blocks[k] and \
                ref_blocks[k][0].lstrip().startswith("Warning:"):
            parts[idx] = None      # a MATLAB warning our run does not emit
            continue
        if ours is None:
            parts[idx] = "```text\n(output not captured for this page yet)\n```"
        else:
            parts[idx] = ("```text\n" + "\n".join(chunks[k]).rstrip("\n") + "\n```"
                          if chunks[k] else "```text\n(no matching output)\n```")
    return [p for p in parts if p is not None], images


def _key(line: str) -> str:
    """Alignment key of an output line: timings match any timing, and
    ``name = value`` lines match on ``name =``."""
    s = line.strip()
    if s.startswith("Elapsed time is"):
        return "Elapsed time is"
    if "=" in s and not s.startswith("="):
        return s.split("=", 1)[0].rstrip() + " ="
    return s


def align_outputs(ref_blocks, ours):
    """Split our stdout lines into one chunk per reference output block."""
    if ours is None:
        return [[] for _ in ref_blocks]
    lines = [ln.rstrip() for ln in ours]
    keys = [_key(ln) for ln in lines]
    starts = []
    cursor = 0
    for blk in ref_blocks:
        first = blk[0].rstrip() if blk else ""
        found = None
        if first:
            kf = _key(first)
            for i in range(cursor, len(lines)):
                if keys[i] == kf:
                    found = i
                    break
        starts.append(found)
        if found is not None:
            # skip the reference block's own lines: a block may itself
            # contain several "ans =" entries (Gibbs2D)
            cursor = found + max(1, len([ln for ln in blk if ln.strip()]))
    chunks = []
    for k, blk in enumerate(ref_blocks):
        if starts[k] is None:
            chunks.append([])
            continue
        nxt = next((s for s in starts[k + 1:] if s is not None), len(lines))
        chunk = lines[starts[k]:nxt]
        while chunk and not chunk[-1].strip():
            chunk.pop()
        chunks.append(chunk)
    # unmatched blocks: consume the reference line count after the previous chunk
    for k, blk in enumerate(ref_blocks):
        if starts[k] is None:
            prev_end = 0
            for j in range(k - 1, -1, -1):
                if starts[j] is not None:
                    prev_end = starts[j] + len(chunks[j])
                    break
            nxt = next((s for s in starts[k + 1:] if s is not None), len(lines))
            chunks[k] = lines[prev_end:min(nxt, prev_end + len(blk))]
    return chunks


def header_md(html_text: str, url: str, cat: str, stem: str, script: str | None):
    title = re.search(r"<h1[^>]*>(.*?)</h1>", html_text, re.S)
    title = html.unescape(re.sub(r"<[^>]+>", "", title.group(1))).strip() if title else stem
    h2 = re.search(r"<h2[^>]*>(.*?)</h2>", html_text, re.S)
    byline = ""
    if h2:
        byline = html.unescape(re.sub(r"<[^>]+>", " ", h2.group(1)))
        byline = re.sub(r"\s+", " ", byline).split(" in ")[0].strip()
    lines = [f"# {title}", ""]
    if byline:
        lines += [f"*{byline}*", ""]
    lines += [f"[Original MATLAB Chebfun example]({url})", ""]
    if script:
        lines += [f"Python translation: [`{script}`](https://github.com/ma-gilles/chebfunjax/blob/main/{script})", ""]
    return "\n".join(lines)


FOOTER = ("\n---\n\n*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); "
          "prose and MATLAB code from the original example, copyright The University of "
          "Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*\n")


def find_script(cat: str, stem: str) -> str | None:
    """The examples/<cat>/*.py script whose docstring links this page."""
    pat = re.compile(rf"chebfun\.org/examples/{re.escape(cat)}/{re.escape(stem)}\.html")
    for p in sorted((EXAMPLES / cat).glob("*.py")):
        if pat.search(p.read_text(encoding="utf-8", errors="replace")):
            return str(p.relative_to(PROJECT))
    return None


def page_url(cat: str, stem: str) -> str:
    md = DOCS / cat / f"{stem}.md"
    if md.exists():
        m = URL_RE.search(md.read_text(encoding="utf-8", errors="replace"))
        if m:
            return m.group(0)
    return f"https://www.chebfun.org/examples/{cat}/{stem}.html"


def generate(cat: str, stem: str, stdout_dir: Path | None, cache: Path, dry_run: bool = False):
    url = page_url(cat, stem)
    _CURRENT["cat"] = cat
    text = fetch_html(url, cache)
    parser = _TreeParser()
    parser.feed(text)
    if parser.root is None:
        raise RuntimeError(f"no content div in {url}")
    script = find_script(cat, stem)
    ours = None
    if stdout_dir is not None and script:
        f = stdout_dir / (Path(script).relative_to("examples").with_suffix("").as_posix().replace("/", "_") + ".txt")
        if f.exists():
            ours = [ln for ln in f.read_text(encoding="utf-8", errors="replace").split("\n")
                    if not ln.startswith("Warn")]
    parts, images = render(parser.root, cat, stem, ours)
    missing = [os.path.basename(s) for s in images if not (IMAGES / cat / os.path.basename(s)).exists()]
    body = "\n\n".join(parts)
    body = re.sub(r"\n{3,}", "\n\n", body)
    md = header_md(text, url, cat, stem, script) + "\n" + body + "\n" + FOOTER
    if not dry_run:
        (DOCS / cat).mkdir(parents=True, exist_ok=True)
        (DOCS / cat / f"{stem}.md").write_text(md, encoding="utf-8")
    n_ref_imgs = len(images)
    return ours is not None, missing, n_ref_imgs, script


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="*", help="cat/Stem (default: every docs page with a chebfun.org link)")
    ap.add_argument("--stdout-dir", type=Path, default=None)
    ap.add_argument("--html-cache", type=Path, default=Path(os.environ.get("CHEBFUN_PAGE_CACHE", "/tmp/chebfun_pages")))
    ap.add_argument("--dry-run", action="store_true", help="report only; do not write pages")
    args = ap.parse_args()
    pages = args.pages
    if not pages:
        for md in sorted(DOCS.glob("*/*.md")):
            if md.name != "index.md" and URL_RE.search(md.read_text(encoding="utf-8", errors="replace")):
                pages.append(f"{md.parent.name}/{md.stem}")
    n_ok = 0
    for key in pages:
        cat, stem = key.split("/")
        try:
            has_out, missing, n_imgs, script = generate(cat, stem, args.stdout_dir, args.html_cache,
                                                        dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001
            print(f"{key}: ERROR {exc}")
            continue
        n_ok += 1
        print(f"{key}: ok script={script} figures={n_imgs}"
              f"{'' if has_out else ' (no stdout)'}"
              f"{' missing images: ' + ' '.join(missing) if missing else ''}")
    print(f"{n_ok}/{len(pages)} pages written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
