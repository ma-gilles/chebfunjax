#!/usr/bin/env python3
"""Generate deterministic, source-pinned Chebfun parity inventory (no execution)."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

NATIVE = Path("/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref")
EXAMPLES = Path("/scratch/gpfs/GILLES/mg6942/chebfun_examples")
PAGES = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/chebfunjax_handoff_20261002/pages_html")
PIN = "7574c77680d7e82b79626300bf255498271a72df"
FUNC = re.compile(r"^\s*function\s+(?:(?:\[[^\]]*\]|\w+)\s*=\s*)?(\w+)\s*(?:\(|$)")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def declarations(path: Path) -> list[dict[str, str]]:
    found = []
    for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        m = FUNC.match(line)
        if m:
            found.append({"name": m.group(1), "line": str(line_no),
                          "kind": "first_detected_declaration" if not found else "subsequent_detected_declaration",
                          "implementation_status": "unverified",
                          "semantic_assertion_coverage_status": "unverified",
                          "evidence": []})
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python-root", type=Path, required=True,
                    help="explicit Chebfunjax checkout used for candidate-path hints")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--matlab-root", type=Path, default=NATIVE)
    ap.add_argument("--cached-pages", type=Path, default=PAGES)
    ap.add_argument("--matlab-examples", type=Path, default=EXAMPLES)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit("Refusing to overwrite a ledger: generate a fresh baseline and reconcile reviewed evidence.")
    native, pages_root, matlab_examples = a.matlab_root, a.cached_pages, a.matlab_examples
    ref = git("rev-parse", "HEAD", cwd=native)
    if ref != PIN:
        raise SystemExit(f"native reference pin mismatch: expected {PIN}, got {ref}")
    repo_ref = git("rev-parse", "HEAD", cwd=a.python_root)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "*.m"], cwd=native, check=True, capture_output=True)
    mapping_roots = ["src", "tests/test_matlab_port", "docs/examples", "examples"]
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *mapping_roots], cwd=a.python_root, check=True, capture_output=True)
    if git("ls-files", "--others", "--exclude-standard", "--", *mapping_roots, cwd=a.python_root):
        raise SystemExit("Candidate mapping roots contain untracked files; use a clean source checkout.")
    tracked = git("ls-files", "-z", "*.m", cwd=native).split("\0")
    native_files = [native / p for p in tracked if p]
    tests, library = [], []
    for p in native_files:
        rel = p.relative_to(native).as_posix()
        is_test = rel.startswith("tests/")
        row = {"source": rel, "sha256": sha(p), "functions": declarations(p),
               "source_status": "unverified", "api_assertion_status": "unverified",
               "evidence": []}
        (tests if is_test else library).append(row)

    # Candidate mappings are navigational only; presence is never parity evidence.
    src = a.python_root / "src"
    pyfiles = sorted(p for p in src.rglob("*.py") if p.is_file())
    by_stem: dict[str, list[str]] = {}
    for p in pyfiles:
        by_stem.setdefault(p.stem.lower(), []).append(p.relative_to(a.python_root).as_posix())
    for row in library:
        row["candidate_port_paths"] = by_stem.get(Path(row["source"]).stem.lower(), [])
    port_tests = sorted((a.python_root / "tests/test_matlab_port").rglob("*.py"))
    port_tests_by_stem: dict[str, list[str]] = {}
    for p in port_tests:
        port_tests_by_stem.setdefault(p.stem.lower(), []).append(
            p.relative_to(a.python_root).as_posix())
    for row in tests:
        stem = Path(row["source"]).stem.lower()
        row["candidate_port_paths"] = sorted(port_tests_by_stem.get(stem, []) +
                                              port_tests_by_stem.get(stem + "_matlab", []))

    pages = []
    python_examples = sorted((a.python_root / "examples").rglob("*.py"))
    htmls = sorted(pages_root.glob("*.html"))
    docs = sorted((a.python_root / "docs/examples").rglob("*.md"))
    examples = sorted(matlab_examples.glob("*/*.m"))
    for p in htmls:
        page_id = p.stem
        stem = page_id.split("_", 1)[-1].lower()
        doc_hits = [x for x in docs if re.sub(r"[^a-z0-9]", "", x.stem.lower()) ==
                    re.sub(r"[^a-z0-9]", "", stem)]
        ex_hits = [x for x in examples if re.sub(r"[^a-z0-9]", "", x.stem.lower()) ==
                   re.sub(r"[^a-z0-9]", "", stem)]
        py_hits = [x for x in python_examples if re.sub(r"[^a-z0-9]", "", x.stem.lower()) ==
                   re.sub(r"[^a-z0-9]", "", stem)]
        pages.append({"page_id": page_id, "reference_html": p.name, "sha256": sha(p),
                      "local_doc_candidates": [x.relative_to(a.python_root).as_posix() for x in doc_hits],
                      "matlab_example_candidates": [x.relative_to(matlab_examples).as_posix() for x in ex_hits],
                      "python_script_candidates": [x.relative_to(a.python_root).as_posix() for x in py_hits],
                      "prose_status": "unverified", "computation_status": "unverified",
                      "stdout_status": "unverified", "figure_content_status": "unverified",
                      "figure_size_status": "unverified", "performance_status": "unverified",
                      "evidence": []})

    for row in tests:
        row.update({"assertion_coverage_status": "unverified", "runtime_status": "unverified",
                    "tolerance_status": "unverified"})
    data = {"schema_version": 1, "generated_by_sha256": sha(Path(__file__)), "generated_by": "scripts/build_parity_inventory.py",
            "generation_mode": "source-only; generator only, no MATLAB/Python package code or numerical execution",
            "function_parser": {
                "method": "line-based regular expression over MATLAB function declarations",
                "limitations": "not a MATLAB parser; multiline/unusual declarations, scope, visibility, class dispatch and public API membership are not determined",
                "counts_are_public_api_counts": False},
            "native_source": str(native), "native_commit": ref,
            "repository": str(a.python_root), "repository_commit": repo_ref,
            "example_source": str(matlab_examples), "cached_page_source": str(pages_root),
            "counts": {"native_library_m_files": len(library),
                       "native_library_function_entries": sum(len(x["functions"]) for x in library),
                       "original_test_m_files": len(tests),
                       "original_test_function_entries": sum(len(x["functions"]) for x in tests),
                       "cached_example_pages": len(pages)},
            "coverage_contract": {"port_presence_is_evidence": False,
                "test_file_count_is_assertion_coverage": False,
                "library_file_and_function_inventory_are_distinct_from_public_api": True,
                "api_assertion_coverage_requires_reviewed_evidence": True,
                "page_computations_stdout_figures_performance_are_independent": True},
            "library_sources": sorted(library, key=lambda x: x["source"]),
            "original_tests": sorted(tests, key=lambda x: x["source"]),
            "example_pages": pages}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps(data["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
