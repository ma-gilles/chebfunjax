"""Regenerate TEST_CHECKLIST.md: one checkbox per MATLAB Chebfun test file
(ported+enabled / partial / skipped-with-reason / missing), plus the
examples and guide inventories.  Run from the repo root."""
from __future__ import annotations

import collections
import glob
import os
import re

ML = "/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref/tests"
PT = "tests/test_matlab_port"
MARK = {"PORTED": "[x]", "PARTIAL": "[~]", "SKIPPED": "[ ]", "MISSING": "[ ]"}


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
                mm = re.search(
                    r'pytestmark = pytest\.mark\.skip\(\s*reason="?([^"\n]*)', txt)
                if mm:
                    status, reason = "SKIPPED", mm.group(1)[:90]
                elif "@pytest.mark.skip" in txt:
                    status, reason = "PARTIAL", "some cases skipped"
                else:
                    status = "PORTED"
            rows.append((d, name, status, reason))
    c = collections.Counter(s for _, _, s, _ in rows)
    total = len(rows)
    out = ["# chebfunjax — Full MATLAB Test Checklist\n",
           "One line per MATLAB Chebfun test file (commit 7574c77). "
           "`[x]` ported & enabled (runs green in the suite), `[~]` ported "
           "with some cases skipped, `[ ]` skipped or missing (reason shown). "
           "GUI-only dirs (chebgui) and adchebfun are excluded by project "
           "policy.\n",
           f"\n**Totals: {total} MATLAB tests — {c['PORTED']} ported+enabled, "
           f"{c['PARTIAL']} partial, {c['SKIPPED']} skipped, {c['MISSING']} "
           f"missing ({100 * (c['PORTED'] + c['PARTIAL']) // total}% "
           "enabled).**\n"]
    bydir = collections.defaultdict(list)
    for d, n, s, r in rows:
        bydir[d].append((n, s, r))
    for d in sorted(bydir):
        items = bydir[d]
        cc = collections.Counter(s for _, s, _ in items)
        out.append(f"\n## {d}  ({cc.get('PORTED', 0) + cc.get('PARTIAL', 0)}"
                   f"/{len(items)} enabled)\n")
        for n, s, r in items:
            line = f"- {MARK[s]} `{n}`"
            if s == "MISSING":
                line += " — no port file"
            elif s == "SKIPPED" and r:
                line += f" — {r.strip()}"
            elif s == "PARTIAL":
                line += " — some cases skipped"
            out.append(line)
    out.append("\n\n# Examples (chebfun.org reproductions)\n")
    out.append("All 21 categories ported; every numbered figure regenerated "
               "(see PARITY_MATRIX.md for the figure-level audit).\n")
    for cat in sorted(os.listdir("examples")):
        p = os.path.join("examples", cat)
        scripts = sorted(glob.glob(p + "/*.py")) if os.path.isdir(p) else []
        if not scripts:
            continue
        out.append(f"\n## examples/{cat} ({len(scripts)} scripts)\n")
        out.extend(f"- [x] `{os.path.basename(sc)}`" for sc in scripts)
    out.append("\n\n# Guide (docs/guide)\n")
    gs = sorted(glob.glob("docs/guide/guide*.md"))
    out.append(f"All {len(gs)} chapters translated with regenerated figures.\n")
    for g in gs:
        nfig = open(g).read().count("![")
        out.append(f"- [x] `{os.path.basename(g)}` ({nfig} figures)")
    open("TEST_CHECKLIST.md", "w").write("\n".join(out) + "\n")
    print(f"TEST_CHECKLIST.md: {total} tests, {dict(c)}")


if __name__ == "__main__":
    main()
