"""Regenerate a static MATLAB test, example and guide file inventory.

File presence and literal pytest markers do not establish runtime or assertion
parity. Run from the repo root; numerical qualification lives in gate evidence.
"""
from __future__ import annotations

import ast
import collections
import glob
import os

ML = "/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref/tests"
PT = "tests/test_matlab_port"
MARK = {"PRESENT": "[x]", "MASKED": "[ ]", "SKIPPED": "[ ]", "MISSING": "[ ]"}


def _mark_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _mark_name(node.value) + "." + node.attr
    return ""


def _static_status(txt: str) -> tuple[str, str]:
    tree = ast.parse(txt)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        ):
            if isinstance(node.value, ast.Call) and _mark_name(node.value.func) == "pytest.mark.skip":
                reason = next((kw.value.value for kw in node.value.keywords
                               if kw.arg == "reason" and isinstance(kw.value, ast.Constant)
                               and isinstance(kw.value.value, str)), "module skip")
                return "SKIPPED", reason[:90]
    marks = {_mark_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    if marks & {"pytest.mark.skip", "pytest.mark.skipif", "pytest.mark.xfail", "pytest.skip", "pytest.xfail"}:
        return "MASKED", "literal skip/skipif/xfail present; inspect runtime evidence"
    return "PRESENT", ""


def main() -> None:
    rows = []
    for d in sorted(os.listdir(ML)):
        md = os.path.join(ML, d)
        if not os.path.isdir(md):
            continue
        for m in sorted(glob.glob(md + "/test_*.m")):
            name = os.path.basename(m)[:-2]
            port = os.path.join(PT, d, name + "_matlab.py")
            status, reason = "MISSING", ""
            if os.path.exists(port):
                txt = open(port).read()
                status, reason = _static_status(txt)
            rows.append((d, name, status, reason))
    c = collections.Counter(s for _, _, s, _ in rows)
    total = len(rows)
    out = ["# chebfunjax — Full MATLAB Test Checklist\n",
           "One line per MATLAB Chebfun test file (commit 7574c77). "
           "`[x]` means a Python file exists without a literal skip/xfail marker; "
           "`[ ]` means a file is missing or contains a skip/xfail marker. "
           "This static inventory does not run tests, check original assertion "
           "coverage or bounds, detect every dynamically assigned marker, or "
           "establish MATLAB parity. Runtime qualification requires gate evidence. "
           "All source suites, including chebgui and adchebfun, count toward "
           "the full-parity goal; existing policy skips remain open gaps.\n",
           f"\n**Totals: {total} MATLAB test files — {c['PRESENT']} present without "
           f"literal masks, {c['MASKED']} with skip/xfail markers, {c['SKIPPED']} "
           f"module-skipped, {c['MISSING']} missing.**\n"]
    bydir = collections.defaultdict(list)
    for d, n, s, r in rows:
        bydir[d].append((n, s, r))
    for d in sorted(bydir):
        items = bydir[d]
        cc = collections.Counter(s for _, s, _ in items)
        out.append(f"\n## {d}  ({len(items) - cc.get('MISSING', 0)}"
                   f"/{len(items)} Python files present)\n")
        for n, s, r in items:
            line = f"- {MARK[s]} `{n}`"
            if s == "MISSING":
                line += " — no port file"
            elif s == "SKIPPED" and r:
                line += f" — {r.strip()}"
            elif s == "MASKED":
                line += " — " + r
            out.append(line)
    out.append("\n\n# Examples (chebfun.org reproductions)\n")
    out.append("Scripts listed below are present. File presence does not "
               "qualify prose, computations, printed output, figure pixels or "
               "reference sizes; see page/figure audit evidence for those gaps.\n")
    for cat in sorted(os.listdir("examples")):
        p = os.path.join("examples", cat)
        scripts = sorted(glob.glob(p + "/*.py")) if os.path.isdir(p) else []
        if not scripts:
            continue
        out.append(f"\n## examples/{cat} ({len(scripts)} scripts)\n")
        out.extend(f"- [x] `{os.path.basename(sc)}`" for sc in scripts)
    out.append("\n\n# Guide (docs/guide)\n")
    gs = sorted(glob.glob("docs/guide/guide*.md"))
    out.append(f"{len(gs)} chapter files are present. Figure counts below "
               "count Markdown image references and do not establish image parity.\n")
    for g in gs:
        nfig = open(g).read().count("![")
        out.append(f"- [x] `{os.path.basename(g)}` ({nfig} figures)")
    open("TEST_CHECKLIST.md", "w").write("\n".join(out) + "\n")
    print(f"TEST_CHECKLIST.md: {total} tests, {dict(c)}")


if __name__ == "__main__":
    main()
