#!/usr/bin/env python3
"""Systematic audit of all jaxchebfun examples against Chebfun originals.

Checks for each original MATLAB example:
1. Python script exists
2. Doc page exists
3. Doc page has chebfun.org link (and it's correct)
4. Image exists in docs/images/
5. Doc page references images that exist
6. Script uses chebfun_style()
7. Script has a run() function
8. Script has a docstring with title

Outputs: CSV report + summary tables.
"""

import os
import re
import csv
import sys
from pathlib import Path
from collections import defaultdict

# Paths
PROJECT = Path("/scratch/gpfs/GILLES/mg6942/jaxchebfun")
EXAMPLES_DIR = PROJECT / "examples"
DOCS_DIR = PROJECT / "docs" / "examples"
IMAGES_DIR = PROJECT / "docs" / "images"
MATLAB_EXAMPLES = Path("/scratch/gpfs/GILLES/mg6942/chebfun_examples")

CATEGORIES = [
    "applics", "approx", "approx2", "approx3", "calc", "cheb", "complex",
    "disk", "fourier", "fun", "geom", "integro", "linalg", "ode-eig",
    "ode-linear", "ode-nonlin", "ode-random", "opt", "pde", "quad",
    "roots", "sphere", "stats", "temp", "veccalc"
]


def get_matlab_examples(category):
    """Get list of MATLAB example names (without .m) for a category."""
    cat_dir = MATLAB_EXAMPLES / category
    if not cat_dir.exists():
        return []
    return sorted(f.stem for f in cat_dir.glob("*.m"))


def get_python_scripts(category):
    """Get list of Python script names (without .py) for a category."""
    cat_dir = EXAMPLES_DIR / category
    if not cat_dir.exists():
        return []
    exclude = {"__init__", "run_all", "run_all_check", "generate_all_plots"}
    return sorted(
        f.stem for f in cat_dir.glob("*.py")
        if f.stem not in exclude and not f.stem.startswith("__")
    )


def get_doc_pages(category):
    """Get list of doc page names (without .md) for a category."""
    cat_dir = DOCS_DIR / category
    if not cat_dir.exists():
        return []
    return sorted(
        f.stem for f in cat_dir.glob("*.md")
        if f.stem != "index"
    )


def get_images(category):
    """Get list of image base names (without .png) for a category."""
    cat_dir = IMAGES_DIR / category
    if not cat_dir.exists():
        return []
    return sorted(f.stem for f in cat_dir.glob("*.png"))


def check_script(category, name):
    """Check properties of a Python script."""
    script_path = EXAMPLES_DIR / category / f"{name}.py"
    result = {
        "script_exists": script_path.exists(),
        "has_run_func": False,
        "has_chebfun_style": False,
        "has_docstring": False,
        "has_savefig": False,
        "script_lines": 0,
    }
    if not script_path.exists():
        return result

    content = script_path.read_text(errors="replace")
    result["script_lines"] = len(content.splitlines())
    result["has_run_func"] = bool(re.search(r"^def run\(", content, re.MULTILINE))
    result["has_chebfun_style"] = "chebfun_style" in content
    result["has_docstring"] = content.strip().startswith('"""') or content.strip().startswith("'''")
    result["has_savefig"] = "savefig" in content or "save_fig" in content or "plt.savefig" in content

    return result


def check_doc(category, name):
    """Check properties of a doc page."""
    doc_path = DOCS_DIR / category / f"{name}.md"
    result = {
        "doc_exists": doc_path.exists(),
        "has_chebfun_link": False,
        "chebfun_link": "",
        "expected_link": f"https://www.chebfun.org/examples/{category}/{name}.html",
        "link_matches": False,
        "has_author": False,
        "has_images_ref": False,
        "image_refs": [],
        "images_exist": [],
        "images_missing": [],
        "doc_lines": 0,
    }
    if not doc_path.exists():
        return result

    content = doc_path.read_text(errors="replace")
    result["doc_lines"] = len(content.splitlines())

    # Check for chebfun.org link
    link_match = re.search(r"https?://(?:www\.)?chebfun\.org/examples/\S+\.html", content)
    if link_match:
        result["has_chebfun_link"] = True
        result["chebfun_link"] = link_match.group(0)
        result["link_matches"] = (
            result["chebfun_link"] == result["expected_link"]
        )

    # Check for author
    result["has_author"] = bool(re.search(r"\*[A-Z][a-z]+ [A-Z]", content))

    # Check image references
    img_refs = re.findall(r"!\[.*?\]\((.*?\.png)\)", content)
    result["has_images_ref"] = len(img_refs) > 0
    result["image_refs"] = img_refs

    for ref in img_refs:
        # Resolve relative to doc location
        img_path = (DOCS_DIR / category / ref).resolve()
        if not img_path.exists():
            # Try from docs/images/
            alt_path = PROJECT / "docs" / ref.lstrip("./")
            if ref.startswith("../../images/"):
                alt_path = DOCS_DIR / category / Path(ref)
                alt_path = alt_path.resolve()
            if alt_path.exists():
                result["images_exist"].append(ref)
            else:
                result["images_missing"].append(ref)
        else:
            result["images_exist"].append(ref)

    return result


def check_image_in_docs(category, name):
    """Check if any image exists in docs/images/ for this example."""
    cat_dir = IMAGES_DIR / category
    if not cat_dir.exists():
        return {"has_image": False, "image_count": 0, "image_files": []}

    # Images might be named: name.png, name_01.png, name_fig1.png, etc.
    images = list(cat_dir.glob(f"{name}*.png"))
    return {
        "has_image": len(images) > 0,
        "image_count": len(images),
        "image_files": [f.name for f in images],
    }


def run_audit():
    """Run the full audit."""
    results = []

    for category in CATEGORIES:
        matlab = set(get_matlab_examples(category))
        python = set(get_python_scripts(category))
        docs = set(get_doc_pages(category))

        # Union of all names
        all_names = sorted(matlab | python | docs)

        for name in all_names:
            row = {
                "category": category,
                "name": name,
                "in_matlab": name in matlab,
                "in_python": name in python,
                "in_docs": name in docs,
            }

            # Script checks
            script_info = check_script(category, name)
            row.update(script_info)

            # Doc checks
            doc_info = check_doc(category, name)
            row.update(doc_info)

            # Image checks
            img_info = check_image_in_docs(category, name)
            row.update(img_info)

            results.append(row)

    return results


def print_summary(results):
    """Print summary tables."""
    # Per-category summary
    cat_stats = defaultdict(lambda: {
        "matlab": 0, "python": 0, "docs": 0,
        "missing_script": 0, "missing_doc": 0, "missing_image": 0,
        "extra_script": 0, "extra_doc": 0,
        "no_chebfun_link": 0, "wrong_link": 0,
        "no_run_func": 0, "no_chebfun_style": 0,
        "missing_img_refs": 0,
    })

    for r in results:
        c = cat_stats[r["category"]]
        if r["in_matlab"]:
            c["matlab"] += 1
        if r["in_python"]:
            c["python"] += 1
        if r["in_docs"]:
            c["docs"] += 1

        if r["in_matlab"] and not r["in_python"]:
            c["missing_script"] += 1
        if r["in_matlab"] and not r["in_docs"]:
            c["missing_doc"] += 1
        if r["in_python"] and not r["in_matlab"]:
            c["extra_script"] += 1
        if r["in_docs"] and not r["in_matlab"]:
            c["extra_doc"] += 1

        if r["in_python"] and not r["has_image"]:
            c["missing_image"] += 1
        if r["in_docs"] and not r["has_chebfun_link"]:
            c["no_chebfun_link"] += 1
        if r["in_docs"] and r["has_chebfun_link"] and not r["link_matches"]:
            c["wrong_link"] += 1
        if r["in_python"] and not r["has_run_func"]:
            c["no_run_func"] += 1
        if r["in_python"] and not r["has_chebfun_style"]:
            c["no_chebfun_style"] += 1
        if r["in_docs"] and r["images_missing"]:
            c["missing_img_refs"] += 1

    # Print per-category table
    print("=" * 130)
    print(f"{'Category':<15} {'MATLAB':>6} {'Python':>6} {'Docs':>6} | {'MissScript':>10} {'MissDoc':>8} {'MissImg':>8} {'ExtraScript':>11} | {'NoLink':>6} {'BadLink':>7} {'NoRun':>5} {'BrkImg':>6}")
    print("=" * 130)

    totals = defaultdict(int)
    for cat in CATEGORIES:
        c = cat_stats[cat]
        print(f"{cat:<15} {c['matlab']:>6} {c['python']:>6} {c['docs']:>6} | {c['missing_script']:>10} {c['missing_doc']:>8} {c['missing_image']:>8} {c['extra_script']:>11} | {c['no_chebfun_link']:>6} {c['wrong_link']:>7} {c['no_run_func']:>5} {c['missing_img_refs']:>6}")
        for k, v in c.items():
            totals[k] += v

    print("-" * 130)
    c = totals
    print(f"{'TOTAL':<15} {c['matlab']:>6} {c['python']:>6} {c['docs']:>6} | {c['missing_script']:>10} {c['missing_doc']:>8} {c['missing_image']:>8} {c['extra_script']:>11} | {c['no_chebfun_link']:>6} {c['wrong_link']:>7} {c['no_run_func']:>5} {c['missing_img_refs']:>6}")
    print()

    # Detailed issues
    print("\n" + "=" * 80)
    print("MISSING SCRIPTS (MATLAB example exists, no Python script)")
    print("=" * 80)
    for r in results:
        if r["in_matlab"] and not r["in_python"]:
            print(f"  {r['category']}/{r['name']}")

    print("\n" + "=" * 80)
    print("MISSING DOCS (Python script exists, no doc page)")
    print("=" * 80)
    for r in results:
        if r["in_python"] and not r["in_docs"]:
            print(f"  {r['category']}/{r['name']}")

    print("\n" + "=" * 80)
    print("EXTRA SCRIPTS (Python script exists, no MATLAB original)")
    print("=" * 80)
    for r in results:
        if r["in_python"] and not r["in_matlab"]:
            print(f"  {r['category']}/{r['name']}")

    print("\n" + "=" * 80)
    print("EXTRA DOCS (doc page exists, no MATLAB original)")
    print("=" * 80)
    for r in results:
        if r["in_docs"] and not r["in_matlab"]:
            print(f"  {r['category']}/{r['name']}")

    print("\n" + "=" * 80)
    print("MISSING IMAGES (Python script exists, no image in docs/images/)")
    print("=" * 80)
    for r in results:
        if r["in_python"] and not r["has_image"]:
            print(f"  {r['category']}/{r['name']}")

    print("\n" + "=" * 80)
    print("DOC PAGES WITH NO CHEBFUN.ORG LINK")
    print("=" * 80)
    for r in results:
        if r["in_docs"] and not r["has_chebfun_link"]:
            print(f"  {r['category']}/{r['name']}")

    print("\n" + "=" * 80)
    print("DOC PAGES WITH WRONG CHEBFUN.ORG LINK")
    print("=" * 80)
    for r in results:
        if r["in_docs"] and r["has_chebfun_link"] and not r["link_matches"]:
            print(f"  {r['category']}/{r['name']}")
            print(f"    expected: {r['expected_link']}")
            print(f"    found:    {r['chebfun_link']}")

    print("\n" + "=" * 80)
    print("SCRIPTS WITHOUT run() FUNCTION")
    print("=" * 80)
    for r in results:
        if r["in_python"] and not r["has_run_func"]:
            print(f"  {r['category']}/{r['name']}")

    print("\n" + "=" * 80)
    print("DOC PAGES WITH BROKEN IMAGE REFERENCES")
    print("=" * 80)
    for r in results:
        if r["in_docs"] and r["images_missing"]:
            print(f"  {r['category']}/{r['name']}")
            for img in r["images_missing"]:
                print(f"    missing: {img}")

    # Scripts without savefig (no plot generation)
    print("\n" + "=" * 80)
    print("SCRIPTS WITHOUT savefig (no plot generation)")
    print("=" * 80)
    for r in results:
        if r["in_python"] and not r["has_savefig"] and not r["has_image"]:
            print(f"  {r['category']}/{r['name']} ({r['script_lines']} lines)")

    return totals


def write_csv(results, path):
    """Write results to CSV."""
    if not results:
        return

    # Flatten complex fields
    flat_results = []
    for r in results:
        fr = dict(r)
        fr["image_refs"] = "; ".join(r.get("image_refs", []))
        fr["images_exist"] = "; ".join(r.get("images_exist", []))
        fr["images_missing"] = "; ".join(r.get("images_missing", []))
        fr["image_files"] = "; ".join(r.get("image_files", []))
        flat_results.append(fr)

    fieldnames = list(flat_results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_results)
    print(f"\nCSV written to: {path}")


if __name__ == "__main__":
    print("Running comprehensive example audit...")
    print(f"  Project:  {PROJECT}")
    print(f"  MATLAB:   {MATLAB_EXAMPLES}")
    print()

    results = run_audit()
    totals = print_summary(results)

    csv_path = PROJECT / "scripts" / "audit_results.csv"
    write_csv(results, csv_path)

    # Final tally
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"  Total MATLAB originals: {totals['matlab']}")
    print(f"  Total Python scripts:   {totals['python']}")
    print(f"  Total doc pages:        {totals['docs']}")
    print(f"  Missing scripts:        {totals['missing_script']}")
    print(f"  Missing docs:           {totals['missing_doc']}")
    print(f"  Missing images:         {totals['missing_image']}")
    print(f"  Extra scripts (no MATLAB): {totals['extra_script']}")
    print(f"  No chebfun.org link:    {totals['no_chebfun_link']}")
    print(f"  Wrong chebfun.org link: {totals['wrong_link']}")
    print(f"  No run() function:      {totals['no_run_func']}")
    print(f"  Broken image refs:      {totals['missing_img_refs']}")
