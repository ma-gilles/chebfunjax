#!/usr/bin/env python3
"""Batch-patch example scripts:

1. Insert `chebfun_style()` call after the matplotlib import block.
2. Fix save-path for scripts that save to `_here` instead of docs/images/<cat>/.
"""
from __future__ import annotations
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")

# Scripts whose save path needs to be fixed from _here to docs/images/<cat>/
# Format: (category, script_basename, image_stem, [extra_image_stems])
FIX_PATHS = [
    # geom
    ("geom", "curves_and_lengths.py", "curves_and_lengths"),
    ("geom", "parametric_surfaces.py", "parametric_surfaces"),
    # sphere
    ("sphere", "sphere_operations.py", "sphere_operations"),
    ("sphere", "spherical_harmonics.py", "spherical_harmonics"),
    # applics
    ("applics", "finance_applications.py", "finance_applications"),
    # integro
    ("integro", "integro_diff.py", "integro_diff"),
    # stats
    ("stats", "probability_distributions.py", "probability_distributions"),
    # fun
    ("fun", "audible_chebfuns.py", "audible_chebfuns"),
    ("fun", "fun_examples.py", "fun_examples"),
]

SKIP_FILES = {"__init__.py", "generate_all_plots.py", "run_all_check.py", "run_all.py"}


def has_chebfun_style(text: str) -> bool:
    return "chebfun_style" in text


def add_chebfun_style(text: str, fpath: str) -> str:
    """Insert chebfun_style() call after the matplotlib/sys.path block."""
    # Strategy: find the last matplotlib.use("Agg") or import matplotlib.pyplot line,
    # then add after the import block (after the last sys.path.insert / import chebfunjax line)

    # If already has chebfun_style, skip
    if has_chebfun_style(text):
        return text

    lines = text.split("\n")

    # Find the best insertion point:
    # After `import chebfunjax as cj` line, or after matplotlib.pyplot import
    insert_after = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Good insertion points (in order of preference)
        if stripped.startswith("import chebfunjax") or "import chebfunjax as" in stripped:
            insert_after = i
            break
        if "matplotlib.pyplot" in stripped and stripped.startswith("import"):
            insert_after = i

    if insert_after == -1:
        # Fallback: after matplotlib.use("Agg")
        for i, line in enumerate(lines):
            if "matplotlib.use" in line:
                insert_after = i
                break

    if insert_after == -1:
        print(f"  WARNING: Could not find insertion point for {fpath}")
        return text

    # Build import + call lines
    inject = [
        "from chebfunjax.plotting import chebfun_style",
        "chebfun_style()",
        "",
    ]

    new_lines = lines[:insert_after + 1] + inject + lines[insert_after + 1:]
    return "\n".join(new_lines)


def fix_save_path(text: str, cat: str, stem: str) -> str:
    """Replace _here-based savefig with outdir-based savefig and add outdir setup."""
    if "outdir" in text and f"docs/images/{cat}" in text:
        # Already correct
        return text

    # Replace: _here = os.path.dirname... + savefig(_here, "stem.png")
    # With: outdir = ... docs/images/cat + os.makedirs + savefig(outdir, ...)

    lines = text.split("\n")
    new_lines = []
    outdir_added = False

    for i, line in enumerate(lines):
        # Replace _here assignment if it's the only place _here is used for savefig
        # Actually, just replace the savefig line and add outdir before run()
        new_lines.append(line)

    # Re-do more carefully
    result = text

    # 1. Replace the savefig call
    old_save = f'os.path.join(_here, "{stem}.png")'
    new_save = f'os.path.join(outdir, "{stem}.png")'
    if old_save not in result:
        # Try with double-quoted stem
        old_save = f"os.path.join(_here, '{stem}.png')"
        new_save = f"os.path.join(outdir, '{stem}.png')"

    if old_save in result:
        result = result.replace(old_save, new_save)

    # 2. Find the run() function definition and add outdir assignment after it
    # Look for "def run():" and add outdir as first statement
    outdir_code = (
        f'    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),\n'
        f'                          \'../../docs/images/{cat}\')\n'
        f'    os.makedirs(outdir, exist_ok=True)\n'
    )

    # Add outdir after "def run():\n" if not already present
    if "outdir" not in result:
        # Find def run(): and add after it
        run_match = re.search(r'(def run\(\)[^:]*:)\n', result)
        if run_match:
            pos = run_match.end()
            result = result[:pos] + outdir_code + result[pos:]

    return result


def process_file(fpath: str, fix_path_info=None) -> bool:
    """Process a single file. Returns True if modified."""
    with open(fpath, 'r', encoding='utf-8') as f:
        original = f.read()

    text = original

    # Add chebfun_style
    text = add_chebfun_style(text, fpath)

    # Fix save path if needed
    if fix_path_info is not None:
        cat, stem = fix_path_info
        text = fix_save_path(text, cat, stem)

    if text != original:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    return False


def main():
    modified = []
    skipped = []
    errors = []

    # Build fix-path lookup
    fix_lookup = {}
    for cat, basename, stem in FIX_PATHS:
        fix_lookup[os.path.join(EXAMPLES, cat, basename)] = (cat, stem)

    # Process all example scripts
    for root, dirs, files in os.walk(EXAMPLES):
        # Skip __pycache__ etc.
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for fn in sorted(files):
            if not fn.endswith('.py') or fn in SKIP_FILES:
                continue
            fpath = os.path.join(root, fn)

            # Only process scripts that generate plots (have savefig or matplotlib)
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            if 'savefig' not in content and 'matplotlib' not in content:
                continue

            fix_path_info = fix_lookup.get(fpath)

            try:
                changed = process_file(fpath, fix_path_info)
                if changed:
                    rel = os.path.relpath(fpath, ROOT)
                    modified.append(rel)
                    action = "fixed+styled" if fix_path_info else "styled"
                    print(f"  [{action}] {rel}")
            except Exception as e:
                errors.append((fpath, str(e)))
                print(f"  [ERROR] {fpath}: {e}")

    print(f"\nModified {len(modified)} files, {len(errors)} errors")
    return len(errors) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
