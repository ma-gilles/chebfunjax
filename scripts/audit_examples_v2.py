#!/usr/bin/env python3
"""Systematic audit v2 — with proper CamelCase↔snake_case mapping.

Strategy:
1. Build a mapping from doc pages: each doc page's chebfun.org link tells us
   which MATLAB original it corresponds to.
2. For scripts without doc pages, try CamelCase→snake_case fuzzy matching.
3. Report: coverage, gaps, link issues, image issues.
"""

import os
import re
import csv
import sys
from pathlib import Path
from collections import defaultdict

PROJECT = Path("/scratch/gpfs/GILLES/mg6942/jaxchebfun")
EXAMPLES_DIR = PROJECT / "examples"
DOCS_DIR = PROJECT / "docs" / "examples"
IMAGES_DIR = PROJECT / "docs" / "images"
MATLAB_DIR = Path("/scratch/gpfs/GILLES/mg6942/chebfun_examples")

CATEGORIES = [
    "applics", "approx", "approx2", "approx3", "calc", "cheb", "complex",
    "disk", "fourier", "fun", "geom", "integro", "linalg", "ode-eig",
    "ode-linear", "ode-nonlin", "ode-random", "opt", "pde", "quad",
    "roots", "sphere", "stats", "temp", "veccalc"
]


def camel_to_snake(name):
    """Convert CamelCase to snake_case."""
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    s = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s)
    return s.lower()


def get_matlab_examples():
    """Get all MATLAB examples: {(category, name): path}."""
    result = {}
    for cat in CATEGORIES:
        cat_dir = MATLAB_DIR / cat
        if not cat_dir.exists():
            continue
        for f in cat_dir.glob("*.m"):
            result[(cat, f.stem)] = f
    return result


def get_python_scripts():
    """Get all Python scripts: {(category, name): path}."""
    result = {}
    exclude = {"__init__", "run_all", "run_all_check", "generate_all_plots"}
    for cat in CATEGORIES:
        cat_dir = EXAMPLES_DIR / cat
        if not cat_dir.exists():
            continue
        for f in cat_dir.glob("*.py"):
            if f.stem not in exclude and not f.stem.startswith("__"):
                result[(cat, f.stem)] = f
    return result


def get_doc_pages():
    """Get all doc pages: {(category, name): path}."""
    result = {}
    for cat in CATEGORIES:
        cat_dir = DOCS_DIR / cat
        if not cat_dir.exists():
            continue
        for f in cat_dir.glob("*.md"):
            if f.stem != "index":
                result[(cat, f.stem)] = f
    return result


def extract_doc_metadata(doc_path):
    """Extract metadata from a doc page."""
    content = doc_path.read_text(errors="replace")
    lines = content.splitlines()

    # Chebfun.org link
    link_match = re.search(r"https?://(?:www\.)?chebfun\.org/examples/([\w-]+)/(\w+)\.html", content)
    chebfun_cat = link_match.group(1) if link_match else None
    chebfun_name = link_match.group(2) if link_match else None
    chebfun_url = link_match.group(0) if link_match else None

    # Author
    has_author = bool(re.search(r"\*[A-Z][a-z]+ [A-Z]", content))

    # Image references
    img_refs = re.findall(r"!\[.*?\]\((.*?\.png)\)", content)

    # Check which images exist
    doc_dir = doc_path.parent
    images_missing = []
    images_found = []
    for ref in img_refs:
        # Resolve relative path
        resolved = (doc_dir / ref).resolve()
        if resolved.exists():
            images_found.append(ref)
        else:
            images_missing.append(ref)

    return {
        "chebfun_cat": chebfun_cat,
        "chebfun_name": chebfun_name,
        "chebfun_url": chebfun_url,
        "has_author": has_author,
        "img_refs": img_refs,
        "images_found": images_found,
        "images_missing": images_missing,
        "line_count": len(lines),
    }


def extract_script_metadata(script_path):
    """Extract metadata from a Python script."""
    content = script_path.read_text(errors="replace")
    return {
        "has_run_func": bool(re.search(r"^def run\(", content, re.MULTILINE)),
        "has_chebfun_style": "chebfun_style" in content,
        "has_savefig": "savefig" in content,
        "line_count": len(content.splitlines()),
    }


def build_mapping(matlab_examples, python_scripts, doc_pages):
    """Build the definitive mapping between MATLAB originals and Python translations.

    Returns:
        matched: list of (matlab_key, python_key, doc_key, match_method)
        unmatched_matlab: list of matlab_key with no Python match
        extra_python: list of python_key with no MATLAB match
        extra_docs: list of doc_key with no MATLAB match
    """
    # Index MATLAB by (cat, lowercase_name) for fuzzy matching
    matlab_by_lower = {}
    for (cat, name) in matlab_examples:
        matlab_by_lower[(cat, name.lower())] = (cat, name)

    # Index MATLAB by (cat, snake_case) for fuzzy matching
    matlab_by_snake = {}
    for (cat, name) in matlab_examples:
        snake = camel_to_snake(name)
        matlab_by_snake[(cat, snake)] = (cat, name)

    matched = []  # (matlab_key, python_key, doc_key, match_method)
    used_matlab = set()
    used_python = set()
    used_docs = set()

    # Phase 1: Match via doc page chebfun.org links
    doc_metadata = {}
    for dk, dp in doc_pages.items():
        meta = extract_doc_metadata(dp)
        doc_metadata[dk] = meta

        if meta["chebfun_name"]:
            # The link tells us the MATLAB original
            link_cat = meta["chebfun_cat"]
            link_name = meta["chebfun_name"]

            # Find the MATLAB original (case-insensitive)
            matlab_key = None
            if (link_cat, link_name) in matlab_examples:
                matlab_key = (link_cat, link_name)
            elif (link_cat, link_name.lower()) in matlab_by_lower:
                matlab_key = matlab_by_lower[(link_cat, link_name.lower())]

            # The doc page corresponds to a Python script with the same name
            doc_cat, doc_name = dk
            py_key = (doc_cat, doc_name) if (doc_cat, doc_name) in python_scripts else None

            if matlab_key and py_key:
                matched.append((matlab_key, py_key, dk, "doc_link"))
                used_matlab.add(matlab_key)
                used_python.add(py_key)
                used_docs.add(dk)
            elif matlab_key and not py_key:
                # Doc exists, MATLAB exists, but no Python script
                matched.append((matlab_key, None, dk, "doc_link_no_script"))
                used_matlab.add(matlab_key)
                used_docs.add(dk)
            elif not matlab_key and py_key:
                # Doc exists, Python exists, but link points to non-existent MATLAB
                matched.append((None, py_key, dk, "doc_link_broken"))
                used_python.add(py_key)
                used_docs.add(dk)

    # Phase 2: Match remaining Python scripts to MATLAB by exact name
    for pk in python_scripts:
        if pk in used_python:
            continue
        cat, name = pk
        if (cat, name) in matlab_examples and (cat, name) not in used_matlab:
            matched.append(((cat, name), pk, None, "exact_name"))
            used_matlab.add((cat, name))
            used_python.add(pk)

    # Phase 3: Match remaining Python scripts to MATLAB by snake_case conversion
    for pk in python_scripts:
        if pk in used_python:
            continue
        cat, name = pk
        # Try to find a MATLAB example whose snake_case matches
        for mk in matlab_examples:
            if mk in used_matlab:
                continue
            m_cat, m_name = mk
            if m_cat == cat and camel_to_snake(m_name) == name:
                matched.append((mk, pk, None, "snake_case"))
                used_matlab.add(mk)
                used_python.add(pk)
                break

    # Phase 4: Match remaining by case-insensitive name in same category
    for pk in python_scripts:
        if pk in used_python:
            continue
        cat, name = pk
        if (cat, name.lower()) in matlab_by_lower:
            mk = matlab_by_lower[(cat, name.lower())]
            if mk not in used_matlab:
                matched.append((mk, pk, None, "case_insensitive"))
                used_matlab.add(mk)
                used_python.add(pk)

    unmatched_matlab = [(cat, name) for (cat, name) in sorted(matlab_examples) if (cat, name) not in used_matlab]
    extra_python = [(cat, name) for (cat, name) in sorted(python_scripts) if (cat, name) not in used_python]
    extra_docs = [(cat, name) for (cat, name) in sorted(doc_pages) if (cat, name) not in used_docs]

    return matched, unmatched_matlab, extra_python, extra_docs, doc_metadata


def check_images(category, name):
    """Check if any image exists in docs/images/ for this example."""
    cat_dir = IMAGES_DIR / category
    if not cat_dir.exists():
        return []
    return [f.name for f in cat_dir.glob(f"{name}*.png")]


def run_audit():
    matlab = get_matlab_examples()
    python = get_python_scripts()
    docs = get_doc_pages()

    print(f"MATLAB originals:  {len(matlab)}")
    print(f"Python scripts:    {len(python)}")
    print(f"Doc pages:         {len(docs)}")
    print()

    matched, unmatched_matlab, extra_python, extra_docs, doc_metadata = build_mapping(matlab, python, docs)

    # ===== COVERAGE SUMMARY =====
    print("=" * 100)
    print("MATCHING SUMMARY")
    print("=" * 100)

    # Count match types
    method_counts = defaultdict(int)
    has_script = 0
    has_doc = 0
    has_both = 0
    for mk, pk, dk, method in matched:
        method_counts[method] += 1
        if pk: has_script += 1
        if dk: has_doc += 1
        if pk and dk: has_both += 1

    print(f"Matched MATLAB→Python+Doc: {has_both}")
    print(f"Matched MATLAB→Python only: {has_script - has_both}")
    print(f"Matched MATLAB→Doc only: {has_doc - has_both}")
    print(f"Match methods: {dict(method_counts)}")
    print(f"Unmatched MATLAB (no Python translation): {len(unmatched_matlab)}")
    print(f"Extra Python (no MATLAB original): {len(extra_python)}")
    print(f"Extra Docs (no MATLAB match): {len(extra_docs)}")

    # ===== PER-CATEGORY TABLE =====
    print("\n" + "=" * 120)
    print(f"{'Category':<15} {'MATLAB':>6} {'Matched':>7} {'Script':>6} {'Doc':>6} {'Both':>5} | {'NoTrans':>7} {'ExtraScript':>11} {'ExtraDocs':>9}")
    print("=" * 120)

    cat_stats = defaultdict(lambda: {"matlab": 0, "matched": 0, "has_script": 0, "has_doc": 0, "has_both": 0,
                                       "no_trans": 0, "extra_script": 0, "extra_docs": 0})

    for mk, pk, dk, method in matched:
        cat = mk[0] if mk else (pk[0] if pk else dk[0])
        c = cat_stats[cat]
        c["matched"] += 1
        if pk: c["has_script"] += 1
        if dk: c["has_doc"] += 1
        if pk and dk: c["has_both"] += 1

    for cat, name in unmatched_matlab:
        cat_stats[cat]["no_trans"] += 1

    for cat, name in extra_python:
        cat_stats[cat]["extra_script"] += 1

    for cat, name in extra_docs:
        cat_stats[cat]["extra_docs"] += 1

    # Count MATLAB per category
    for (cat, name) in matlab:
        cat_stats[cat]["matlab"] += 1

    for cat in CATEGORIES:
        c = cat_stats[cat]
        print(f"{cat:<15} {c['matlab']:>6} {c['matched']:>7} {c['has_script']:>6} {c['has_doc']:>6} {c['has_both']:>5} | {c['no_trans']:>7} {c['extra_script']:>11} {c['extra_docs']:>9}")

    # Totals
    t = defaultdict(int)
    for c in cat_stats.values():
        for k, v in c.items():
            t[k] += v
    print("-" * 120)
    print(f"{'TOTAL':<15} {t['matlab']:>6} {t['matched']:>7} {t['has_script']:>6} {t['has_doc']:>6} {t['has_both']:>5} | {t['no_trans']:>7} {t['extra_script']:>11} {t['extra_docs']:>9}")

    # ===== UNTRANSLATED MATLAB EXAMPLES =====
    print("\n" + "=" * 100)
    print(f"UNTRANSLATED MATLAB EXAMPLES ({len(unmatched_matlab)} total)")
    print("=" * 100)
    for cat, name in unmatched_matlab:
        print(f"  {cat}/{name}")

    # ===== EXTRA PYTHON SCRIPTS (original content, not direct translations) =====
    print("\n" + "=" * 100)
    print(f"EXTRA PYTHON SCRIPTS — no MATLAB original ({len(extra_python)} total)")
    print("=" * 100)
    for cat, name in extra_python:
        script_path = python[(cat, name)]
        meta = extract_script_metadata(script_path)
        doc_exists = (cat, name) in docs
        images = check_images(cat, name)
        print(f"  {cat}/{name}  (doc={'YES' if doc_exists else 'NO'}, img={len(images)}, lines={meta['line_count']})")

    # ===== DOC PAGES WITHOUT MATCHING SCRIPT =====
    print("\n" + "=" * 100)
    print("DOC PAGES WITHOUT MATCHING PYTHON SCRIPT")
    print("=" * 100)
    for mk, pk, dk, method in matched:
        if dk and not pk:
            cat, name = dk
            print(f"  {cat}/{name}  (matched MATLAB: {mk})")

    # ===== CHEBFUN.ORG LINK ISSUES =====
    print("\n" + "=" * 100)
    print("CHEBFUN.ORG LINK ISSUES")
    print("=" * 100)

    # Pages with no link
    no_link = []
    for dk, meta in doc_metadata.items():
        if not meta["chebfun_url"]:
            no_link.append(dk)
    if no_link:
        print(f"\n  No chebfun.org link ({len(no_link)}):")
        for cat, name in sorted(no_link):
            print(f"    {cat}/{name}")

    # Pages where link category doesn't match file category (cross-category)
    cross_cat = []
    for mk, pk, dk, method in matched:
        if dk and doc_metadata[dk]["chebfun_cat"]:
            doc_cat = dk[0]
            link_cat = doc_metadata[dk]["chebfun_cat"]
            if doc_cat != link_cat:
                cross_cat.append((dk, doc_metadata[dk]["chebfun_cat"], doc_metadata[dk]["chebfun_name"]))
    if cross_cat:
        print(f"\n  Cross-category links ({len(cross_cat)}):")
        for dk, link_cat, link_name in sorted(cross_cat):
            print(f"    {dk[0]}/{dk[1]} → {link_cat}/{link_name}")

    # ===== IMAGE ISSUES =====
    print("\n" + "=" * 100)
    print("IMAGE COVERAGE")
    print("=" * 100)

    scripts_with_img = 0
    scripts_no_img = []
    for pk, pp in sorted(python.items()):
        images = check_images(pk[0], pk[1])
        if images:
            scripts_with_img += 1
        else:
            scripts_no_img.append(pk)

    print(f"  Scripts with images in docs/images/: {scripts_with_img}/{len(python)}")
    print(f"  Scripts missing images: {len(scripts_no_img)}")
    if scripts_no_img:
        print(f"\n  Missing images (first 30):")
        for cat, name in scripts_no_img[:30]:
            print(f"    {cat}/{name}")
        if len(scripts_no_img) > 30:
            print(f"    ... and {len(scripts_no_img) - 30} more")

    # Broken image refs in docs
    broken_img_docs = []
    for dk, meta in sorted(doc_metadata.items()):
        if meta["images_missing"]:
            broken_img_docs.append((dk, meta["images_missing"]))
    if broken_img_docs:
        print(f"\n  Doc pages with broken image refs ({len(broken_img_docs)}):")
        for dk, missing in broken_img_docs:
            print(f"    {dk[0]}/{dk[1]}: {missing}")
    else:
        print(f"\n  No broken image references in doc pages.")

    # ===== DETAILED VERIFICATION SAMPLE =====
    # Check a few chebfun.org links actually work
    print("\n" + "=" * 100)
    print("CHEBFUN.ORG LINK VALIDATION (unique URLs)")
    print("=" * 100)
    unique_urls = set()
    for dk, meta in doc_metadata.items():
        if meta["chebfun_url"]:
            unique_urls.add(meta["chebfun_url"])
    print(f"  Total unique chebfun.org URLs referenced: {len(unique_urls)}")
    print(f"  (URL validation requires network access — run separately)")

    # ===== CSV OUTPUT =====
    csv_path = PROJECT / "scripts" / "audit_results_v2.csv"
    rows = []

    # Matched entries
    for mk, pk, dk, method in matched:
        row = {
            "matlab_cat": mk[0] if mk else "",
            "matlab_name": mk[1] if mk else "",
            "python_cat": pk[0] if pk else "",
            "python_name": pk[1] if pk else "",
            "doc_cat": dk[0] if dk else "",
            "doc_name": dk[1] if dk else "",
            "match_method": method,
            "has_script": bool(pk),
            "has_doc": bool(dk),
            "chebfun_url": doc_metadata[dk]["chebfun_url"] if dk and dk in doc_metadata else "",
            "doc_images_missing": "; ".join(doc_metadata[dk]["images_missing"]) if dk and dk in doc_metadata else "",
        }
        if pk:
            smeta = extract_script_metadata(python[pk])
            row["script_lines"] = smeta["line_count"]
            row["has_savefig"] = smeta["has_savefig"]
            row["has_chebfun_style"] = smeta["has_chebfun_style"]
            row["has_run_func"] = smeta["has_run_func"]
            imgs = check_images(pk[0], pk[1])
            row["image_count"] = len(imgs)
        else:
            row["script_lines"] = 0
            row["has_savefig"] = False
            row["has_chebfun_style"] = False
            row["has_run_func"] = False
            row["image_count"] = 0
        rows.append(row)

    # Unmatched MATLAB
    for cat, name in unmatched_matlab:
        rows.append({
            "matlab_cat": cat, "matlab_name": name,
            "python_cat": "", "python_name": "",
            "doc_cat": "", "doc_name": "",
            "match_method": "UNTRANSLATED",
            "has_script": False, "has_doc": False,
            "chebfun_url": f"https://www.chebfun.org/examples/{cat}/{name}.html",
            "doc_images_missing": "",
            "script_lines": 0, "has_savefig": False,
            "has_chebfun_style": False, "has_run_func": False,
            "image_count": 0,
        })

    # Extra Python
    for cat, name in extra_python:
        smeta = extract_script_metadata(python[(cat, name)])
        imgs = check_images(cat, name)
        doc_key = (cat, name) if (cat, name) in docs else None
        rows.append({
            "matlab_cat": "", "matlab_name": "",
            "python_cat": cat, "python_name": name,
            "doc_cat": cat if doc_key else "", "doc_name": name if doc_key else "",
            "match_method": "EXTRA_PYTHON",
            "has_script": True, "has_doc": bool(doc_key),
            "chebfun_url": doc_metadata[doc_key]["chebfun_url"] if doc_key and doc_key in doc_metadata else "",
            "doc_images_missing": "",
            "script_lines": smeta["line_count"],
            "has_savefig": smeta["has_savefig"],
            "has_chebfun_style": smeta["has_chebfun_style"],
            "has_run_func": smeta["has_run_func"],
            "image_count": len(imgs),
        })

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written to: {csv_path}")

    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    total_matlab = len(matlab)
    translated = sum(1 for mk, pk, dk, m in matched if pk and mk)
    documented = sum(1 for mk, pk, dk, m in matched if dk and mk)
    fully_done = sum(1 for mk, pk, dk, m in matched if pk and dk and mk)
    print(f"  MATLAB originals:                    {total_matlab}")
    print(f"  Translated (have Python script):      {translated} ({100*translated/total_matlab:.0f}%)")
    print(f"  Documented (have doc page):           {documented} ({100*documented/total_matlab:.0f}%)")
    print(f"  Fully done (script + doc):            {fully_done} ({100*fully_done/total_matlab:.0f}%)")
    print(f"  Untranslated:                         {len(unmatched_matlab)}")
    print(f"  Extra Python scripts (original work): {len(extra_python)}")
    print(f"  Scripts with images:                  {scripts_with_img}/{len(python)}")
    print(f"  Scripts missing images:               {len(scripts_no_img)}")


if __name__ == "__main__":
    os.chdir(PROJECT)
    run_audit()
