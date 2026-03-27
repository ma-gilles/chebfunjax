"""Generate all missing PNG images for the chebfunjax documentation site.

Usage (from repo root):
    pixi run python scripts/generate_all_images.py

This script:
1. Fixes broken image paths in all docs/examples/**/*.md files.
2. Copies PNGs that already exist in examples/{cat}/ to docs/images/{cat}/.
3. Runs each example script to generate any remaining missing images.
4. Reports a final summary of success/failure.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Ensure src/ is importable when running example modules
sys.path.insert(0, str(REPO / "src"))

# ---------------------------------------------------------------------------
# Step 1 — Fix broken image paths in markdown files
# ---------------------------------------------------------------------------

IMG_PATTERN = re.compile(r'(!\[.*?\])\(([^)]+\.png)\)')


def _correct_img_ref(md_file: Path, img_ref: str) -> str | None:
    """Return the corrected image reference, or None if already correct."""
    md_dir = md_file.parent
    target = (md_dir / img_ref).resolve()

    # Already points to docs/images/ and exists → fine
    if target.exists():
        return None

    # Determine the canonical category and image name
    cat = md_dir.name  # e.g. 'approx', 'calc', etc.
    img_name = target.name  # e.g. 'mean_value_theorem.png'

    # Case 1: ../../../images/{cat}/{img}  → wrong (goes to repo-root/images/)
    # Case 2: ../../../docs/images/{cat}/{img} → non-standard (but resolves OK)
    # Both should become ../../images/{cat}/{img}
    if '../../../images/' in img_ref or '../../../docs/images/' in img_ref:
        return f'../../images/{cat}/{img_name}'

    # Case 3: top-level docs/examples/*.md uses ../../examples/{cat}/{img}
    # These are actually fine (they work), skip them
    if '../../examples/' in img_ref:
        return None  # leave as-is, those reference examples/ PNGs directly

    # No fix needed
    return None


def fix_markdown_paths(dry_run: bool = False) -> dict[str, int]:
    """Fix broken image paths in all docs/examples/**/*.md files.

    Returns a dict with counts: fixed_files, fixed_refs.
    """
    stats = {"fixed_files": 0, "fixed_refs": 0}

    md_files = sorted(REPO.glob("docs/examples/**/*.md"))
    for md_file in md_files:
        if md_file.name == "index.md":
            continue

        original = md_file.read_text(encoding="utf-8")
        updated = original
        file_fixed = 0

        for m in IMG_PATTERN.finditer(original):
            bracket_part = m.group(1)  # e.g. '![alt text]'
            img_ref = m.group(2)       # e.g. '../../../images/calc/foo.png'
            corrected = _correct_img_ref(md_file, img_ref)
            if corrected is not None and corrected != img_ref:
                old_full = f"{bracket_part}({img_ref})"
                new_full = f"{bracket_part}({corrected})"
                updated = updated.replace(old_full, new_full, 1)
                file_fixed += 1
                stats["fixed_refs"] += 1

        if file_fixed > 0:
            stats["fixed_files"] += 1
            rel = md_file.relative_to(REPO)
            print(f"  [PATH FIX] {rel}  ({file_fixed} ref(s))")
            if not dry_run:
                md_file.write_text(updated, encoding="utf-8")

    return stats


# ---------------------------------------------------------------------------
# Step 2 — Copy existing PNGs from examples/{cat}/ to docs/images/{cat}/
# ---------------------------------------------------------------------------

def copy_existing_pngs() -> int:
    """Copy PNGs from examples/{cat}/ to docs/images/{cat}/ when missing."""
    copied = 0
    for png in sorted(REPO.glob("examples/*/*.png")):
        cat = png.parent.name
        dest_dir = REPO / "docs" / "images" / cat
        dest = dest_dir / png.name
        if not dest.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png, dest)
            print(f"  [COPY] examples/{cat}/{png.name} → docs/images/{cat}/{png.name}")
            copied += 1
    return copied


# ---------------------------------------------------------------------------
# Step 3 — Run example scripts to generate remaining missing images
# ---------------------------------------------------------------------------

def _load_and_run(py_file: Path) -> None:
    """Import and run a single example script's run() function."""
    spec = importlib.util.spec_from_file_location(
        py_file.stem, str(py_file)
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if hasattr(mod, "run"):
        mod.run()


def _find_script_for_image(img_name: str, cat: str) -> Path | None:
    """Locate the example script that should generate img_name.png."""
    # Primary: examples/{cat}/{img_name}.py
    primary = REPO / "examples" / cat / f"{img_name}.py"
    if primary.exists():
        return primary
    return None


def collect_missing_images() -> list[tuple[str, str, Path]]:
    """Return list of (cat, img_name, target_path) for missing doc images."""
    missing = []
    IMG_PAT = re.compile(r'!\[.*?\]\(([^)]+\.png)\)')

    for md_file in sorted(REPO.glob("docs/examples/**/*.md")):
        if md_file.name == "index.md":
            continue
        text = md_file.read_text(encoding="utf-8")
        for m in IMG_PAT.finditer(text):
            img_ref = m.group(1)
            target = (md_file.parent / img_ref).resolve()
            if not target.exists():
                cat = md_file.parent.name
                img_name = target.stem
                missing.append((cat, img_name, target))

    # Deduplicate by target path
    seen: set[Path] = set()
    unique = []
    for cat, img_name, target in missing:
        if target not in seen:
            seen.add(target)
            unique.append((cat, img_name, target))
    return unique


def run_example_scripts(missing: list[tuple[str, str, Path]]) -> dict[str, list]:
    """Run example scripts to generate missing images.

    Returns a dict: passed, failed, skipped.
    """
    results: dict[str, list] = {"passed": [], "failed": [], "skipped": []}

    # Group by script to avoid running the same script twice
    scripts_needed: dict[Path, list[tuple[str, str, Path]]] = {}
    for cat, img_name, target in missing:
        script = _find_script_for_image(img_name, cat)
        if script is None:
            # Try parent script (e.g. smooth_functions_2d generates multiple PNGs)
            base = img_name.rsplit("_", 1)[0]  # strip suffix
            script = _find_script_for_image(base, cat)
        if script is None:
            results["skipped"].append((cat, img_name, "no script found"))
            continue
        scripts_needed.setdefault(script, []).append((cat, img_name, target))

    print(f"\nRunning {len(scripts_needed)} unique example scripts...")

    for script, targets in sorted(scripts_needed.items()):
        rel = script.relative_to(REPO)
        t0 = time.time()
        try:
            _load_and_run(script)
            elapsed = time.time() - t0
            # Check if targets are now satisfied (directly or via copy)
            for cat, img_name, target in targets:
                if not target.exists():
                    # Try copying from examples/ if the script saved there
                    src = REPO / "examples" / cat / target.name
                    if src.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, target)
            print(f"  [OK]   {rel}  ({elapsed:.1f}s)")
            results["passed"].append(str(rel))
        except Exception as exc:
            elapsed = time.time() - t0
            results["failed"].append((str(rel), str(exc)))
            print(f"  [FAIL] {rel}  ({elapsed:.1f}s)  Error: {exc}")

    return results


# ---------------------------------------------------------------------------
# Step 4 — Verify and report
# ---------------------------------------------------------------------------

def verify_images() -> tuple[list[Path], list[tuple[Path, str]]]:
    """Re-scan docs to find still-missing images."""
    IMG_PAT = re.compile(r'!\[.*?\]\(([^)]+\.png)\)')
    still_missing = []
    ok_count = 0

    for md_file in sorted(REPO.glob("docs/examples/**/*.md")):
        if md_file.name == "index.md":
            continue
        text = md_file.read_text(encoding="utf-8")
        for m in IMG_PAT.finditer(text):
            img_ref = m.group(1)
            target = (md_file.parent / img_ref).resolve()
            if not target.exists():
                still_missing.append((md_file.relative_to(REPO), img_ref))
            else:
                ok_count += 1

    return ok_count, still_missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> bool:
    print(f"\n{'='*70}")
    print("  chebfunjax: Generate all missing documentation images")
    print(f"{'='*70}\n")

    # Step 1: Fix paths
    print("Step 1: Fixing broken image paths in markdown files...")
    path_stats = fix_markdown_paths()
    print(f"  Fixed {path_stats['fixed_refs']} refs in "
          f"{path_stats['fixed_files']} files.\n")

    # Step 2: Copy existing PNGs
    print("Step 2: Copying existing PNGs from examples/ to docs/images/...")
    n_copied = copy_existing_pngs()
    print(f"  Copied {n_copied} PNG files.\n")

    # Step 3: Collect remaining missing and run scripts
    print("Step 3: Collecting still-missing images...")
    missing = collect_missing_images()
    print(f"  Found {len(missing)} missing images.")

    run_results: dict[str, list] = {"passed": [], "failed": [], "skipped": []}
    if missing:
        run_results = run_example_scripts(missing)
    else:
        print("  All images present — nothing to generate.")

    # Step 4: Final verification
    print("\nStep 4: Verification...")
    ok_count, still_missing = verify_images()
    print(f"  OK: {ok_count} image refs resolved.")
    if still_missing:
        print(f"  STILL MISSING: {len(still_missing)}")
        for md_rel, img_ref in still_missing:
            print(f"    {md_rel} -> {img_ref}")
    else:
        print("  All image refs resolved successfully!")

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Markdown path fixes:   {path_stats['fixed_refs']} refs in "
          f"{path_stats['fixed_files']} files")
    print(f"  PNGs copied:           {n_copied}")
    print(f"  Scripts ran OK:        {len(run_results['passed'])}")
    print(f"  Scripts failed:        {len(run_results['failed'])}")
    print(f"  Skipped (no script):   {len(run_results['skipped'])}")
    print(f"  Still missing images:  {len(still_missing)}")
    print(f"{'='*70}\n")

    if run_results["failed"]:
        print("FAILED scripts:")
        for name, err in run_results["failed"]:
            print(f"  - {name}: {err}")
    if run_results["skipped"]:
        print("SKIPPED (no script found):")
        for cat, img_name, reason in run_results["skipped"]:
            print(f"  - {cat}/{img_name}: {reason}")

    return len(still_missing) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
