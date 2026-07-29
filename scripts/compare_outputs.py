#!/usr/bin/env python3
"""Compare example-script numeric output against chebfun.org published output.

Consumes the per-example reference files written by ``harvest_example_outputs.py``
(the ``<pre class="mcode-output">`` numbers MATLAB echoed on each chebfun.org
example page), runs the corresponding ``examples/**/*.py`` port, captures the
numbers it prints, and reports how faithfully the port reproduces the published
results -- to the precision MATLAB displayed.

This is a MEASUREMENT tool, not a fixer: it records the state of each script.

Classification (per script)
---------------------------
- ``BLOCKED``   the port failed to run (exception / timeout). The metric names
                the terminating error so a missing feature is visible.
- ``NO_OUTPUT`` the reference page printed no verifiable numeric output (nothing
                to compare against).
- ``PASS``      the port ran and reproduced every published number to displayed
                precision.
- ``SOFT_PASS`` the port ran and reproduced every published number to at least 8
                significant figures, but not to MATLAB's full ``format long``
                display -- a display-precision gap, not a numeric disagreement.
- ``DIFF``      the port ran but at least one published number is genuinely not
                reproduced; the metric reports recall and the worst missed value.

Matching model
--------------
Every published number carries an absolute tolerance equal to half the place
value of its last displayed digit (see ``harvest_example_outputs.place_value_tol``).
A reference number is *matched* if the port printed some number within that
tolerance. Values displayed as zero (``|v| <= tol``) are unverifiable and are
excluded from the recall denominator. Matching is greedy and multiset-aware so a
printed value is consumed by at most one reference number.

Usage::

    python scripts/compare_outputs.py                 # every harvested reference
    python scripts/compare_outputs.py --timeout 300
    python scripts/compare_outputs.py approx/AAAApprox # a single example
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harvest_example_outputs import parse_numbers  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
REFS_OUT = Path(
    os.environ.get(
        "REFS_OUTPUT_DIR",
        "/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/refs_output",
    )
)
RESULTS_CSV = Path(
    os.environ.get(
        "OUTPUT_PARITY_CSV",
        "/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/parity_matrix/output_parity.csv",
    )
)


def load_references(selectors: set[str] | None) -> list[dict]:
    refs = []
    for path in sorted(REFS_OUT.glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = f"{record['category']}/{record['stem']}"
        if selectors is not None and key not in selectors:
            continue
        refs.append(record)
    return refs


def run_script(script: str, timeout: int) -> tuple[int, str, str]:
    """Run an example script in isolation; return (returncode, stdout, stderr)."""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env["MPLBACKEND"] = "Agg"
    try:
        proc = subprocess.run(
            [sys.executable, str(PROJECT / script)],
            cwd=str(PROJECT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    return proc.returncode, proc.stdout, proc.stderr


# A block whose leading line is a bare ``<name> =`` assignment with a
# wall-clock-timing variable name (``time``, ``time_in_seconds``,
# ``total_time_in_seconds``, ``time_elapsed_in_seconds``, ...) carries a MATLAB
# tic/toc reading, not a parity signal: no port can reproduce another machine's
# elapsed time.  Its numbers are dropped from the reference set.  The rule is
# deliberately narrow -- it requires the label to END in ``=`` -- so it never
# touches a data table that merely has a "computation time" *column* (e.g.
# approx/WeierstrassFunction, whose header is not an assignment).
_TIMING_HEAD_RE = re.compile(
    r"^\s*\w*(?:time|elapsed|second)\w*\s*=\s*$", re.IGNORECASE
)


def _is_timing_block(block: dict) -> bool:
    head = block["text"].strip().splitlines()[0] if block["text"].strip() else ""
    return bool(_TIMING_HEAD_RE.match(head))


def reference_numbers(record: dict) -> list[dict]:
    nums: list[dict] = []
    for block in record["blocks"]:
        if _is_timing_block(block):
            continue
        nums.extend(block["numbers"])
    return nums


# Reference values below this magnitude are numerically zero: MATLAB's
# ``format long`` prints residuals like 1.7e-15 or 4.2e-24 at full precision,
# but they carry no parity signal (a port that computes a different residual
# near machine epsilon is not "wrong"). Excluded from the recall denominator.
ABS_FLOOR = 1e-11


def _verifiable(n: dict) -> bool:
    return abs(n["value"]) > n["tol"] and abs(n["value"]) >= ABS_FLOOR


# A published number that our port reproduces to at least this relative
# accuracy -- but not to MATLAB's full ``format long`` display -- is a
# SOFT_PASS: the computation agrees to >=8 significant figures, the miss is only
# display precision, not a real numeric disagreement. Evidence: the first tranche
# showed a cluster of ports agreeing to 8-16 figs (EuropeanCall 5e-11, geom/Area
# 2e-9) cleanly separated from real gaps (calc/Integrals relerr 0.58).
SOFT_TOL = 1e-8


def match_recall(ref_nums: list[dict], our_vals: list[float]) -> dict:
    """Greedily match verifiable reference numbers against printed values.

    Returns match count plus, for every *unmatched* verifiable reference number,
    its best-available relative error -- so the caller can tell a display-only
    near-miss (SOFT_PASS) from a genuine disagreement or an absent value.
    """
    verifiable = [n for n in ref_nums if _verifiable(n)]
    remaining = list(our_vals)
    matched = 0
    unmatched = []  # list of (abs_value, ref_raw, ref_value, rel_err_or_inf)
    for n in verifiable:
        best_i, best_d = None, None
        for i, v in enumerate(remaining):
            d = abs(v - n["value"])
            if best_d is None or d < best_d:
                best_d, best_i = d, i
        if best_i is not None and best_d <= n["tol"]:
            matched += 1
            remaining.pop(best_i)
        else:
            rel = float("inf") if best_d is None else best_d / max(abs(n["value"]), 1e-300)
            unmatched.append((abs(n["value"]), n["raw"], n["value"], rel))
    # Worst = the *most significant* published number missed (largest magnitude),
    # reported with its best relative error -- more informative than largest
    # relerr, which fixates on near-zero residuals.
    worst = max(unmatched, key=lambda u: u[0]) if unmatched else None
    worst_relerr = max((u[3] for u in unmatched), default=0.0)
    return {
        "n_verifiable": len(verifiable),
        "matched": matched,
        "worst": worst,
        "worst_relerr": worst_relerr,
    }


def last_error_line(stderr: str) -> str:
    lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    if not lines:
        return "no stderr"
    for ln in reversed(lines):
        if ":" in ln and any(
            tok in ln for tok in ("Error", "error", "Exception", "timeout")
        ):
            return ln.strip()[:200]
    return lines[-1].strip()[:200]


def classify(record: dict, timeout: int) -> dict:
    ref_nums = reference_numbers(record)
    verifiable = [n for n in ref_nums if _verifiable(n)]
    if not verifiable:
        return dict(state="NO_OUTPUT", metric="reference prints no numeric output", note="")

    rc, out, err = run_script(record["script"], timeout)
    if rc != 0:
        return dict(state="BLOCKED", metric=last_error_line(err),
                    note=f"exit={rc}")

    our_vals = [n["value"] for n in parse_numbers(out)]
    res = match_recall(ref_nums, our_vals)
    matched, total = res["matched"], res["n_verifiable"]
    if matched == total:
        return dict(state="PASS", metric=f"recall={matched}/{total}", note="")
    worst = res["worst"]
    if worst[3] == float("inf"):
        worst_str = f"{worst[1]}=absent"
    else:
        worst_str = f"{worst[1]} relerr={worst[3]:.2e}"
    # SOFT_PASS: every missed number was still reproduced to >=8 sig figs --
    # the only gap is MATLAB's format-long display, not a numeric disagreement.
    if res["worst_relerr"] <= SOFT_TOL:
        return dict(state="SOFT_PASS",
                    metric=f"recall={matched}/{total} max_relerr={res['worst_relerr']:.2e}",
                    note=f"worst[{worst_str}]")
    return dict(state="DIFF",
                metric=f"recall={matched}/{total} worst[{worst_str}]",
                note=f"printed {len(our_vals)} numbers")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("examples", nargs="*", help="category/Stem selectors (optional)")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    selectors = set(args.examples) if args.examples else None
    records = load_references(selectors)
    if not records:
        print("no reference files found under", REFS_OUT)
        return 1

    rows = []
    counts = {"PASS": 0, "SOFT_PASS": 0, "DIFF": 0, "NO_OUTPUT": 0, "BLOCKED": 0}
    print(f"comparing {len(records)} example scripts (timeout={args.timeout}s)\n")
    for record in records:
        result = classify(record, args.timeout)
        counts[result["state"]] += 1
        rows.append(dict(script=record["script"], category=record["category"],
                         stem=record["stem"], url=record["url"], **result))
        print(f"  {result['state']:<9} {record['category']}/{record['stem']}: {result['metric']}")

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Selector runs MERGE into an existing csv rather than clobbering the
    # full-sweep results: keep prior rows for scripts not re-run here.
    if selectors is not None and RESULTS_CSV.exists():
        rerun = {(r["category"], r["stem"]) for r in rows}
        with open(RESULTS_CSV, newline="") as fh:
            kept = [r for r in csv.DictReader(fh)
                    if (r["category"], r["stem"]) not in rerun]
        rows = sorted(kept + rows, key=lambda r: (r["category"], r["stem"]))
    with open(RESULTS_CSV, "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["script", "category", "stem", "url", "state", "metric", "note"]
        )
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {RESULTS_CSV}")
    total = len(rows)
    print(
        f"PASS={counts['PASS']} SOFT_PASS={counts['SOFT_PASS']} DIFF={counts['DIFF']} "
        f"NO_OUTPUT={counts['NO_OUTPUT']} BLOCKED={counts['BLOCKED']} total={total}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
