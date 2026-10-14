"""Run diagnostic-only default/Zen LU captures on the same CPU host.

No numerical algorithm or scientific assertion is changed. Actual OpenBLAS
selection comes from its startup log, not inference from the requested core.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = Path("tests/references/piecewise_variable_native_lu.npz").resolve()
    fixture_hash = hashlib.sha256(fixture.read_bytes()).hexdigest()
    cpu_text = Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0]
    flags_line = next((line.partition(":")[2] for line in cpu_text.splitlines()
                       if line.partition(":")[0].strip() == "flags"), "")
    flags = set(flags_line.split())
    safe_zen = {"avx", "avx2", "fma"} <= flags
    summary = {
        "fixture": str(fixture), "fixture_sha256": fixture_hash,
        "cpu_first_record": cpu_text,
        "scope": "Diagnostic only; the unchanged pytest assertion runs separately.",
        "zen_supported": safe_zen, "arms": {},
    }
    records = {}
    for arm in ("default", "zen"):
        if arm == "zen" and not safe_zen:
            summary["arms"][arm] = {"skipped": "Required AVX/AVX2/FMA flags absent"}
            continue
        env = os.environ.copy()
        env.pop("OPENBLAS_CORETYPE", None)
        env.update(OPENBLAS_VERBOSE="2", OPENBLAS_NUM_THREADS="1",
                   OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
                   JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true")
        if arm == "zen":
            env["OPENBLAS_CORETYPE"] = "ZEN"
        # Preserve the original default-arm filename for existing artifact readers.
        output = args.output / ("fresh-process.json" if arm == "default" else "zen-process.json")
        command = [sys.executable, "-m", "tests.piecewise_lu_diagnostic",
                   "--fixture", str(fixture), "--output", str(output)]
        with (args.output / (arm + ".stdout")).open("w") as stdout, \
                (args.output / (arm + ".stderr")).open("w") as stderr:
            try:
                process = subprocess.run(command, env=env, stdout=stdout,
                                         stderr=stderr, timeout=90, check=False)
                status = {"returncode": process.returncode}
            except subprocess.TimeoutExpired:
                status = {"timeout_seconds": 90}
        stderr_text = (args.output / (arm + ".stderr")).read_text()
        cores = re.findall(r"^Core:\s*(.+?)\s*$", stderr_text, re.MULTILINE)
        status.update(actual_core_log=cores, requested_core=env.get("OPENBLAS_CORETYPE"),
                      fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest())
        if output.exists():
            records[arm] = json.loads(output.read_text())
            status["capture_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
            status["stages"] = [{key: row[key] for key in
                                 ("origin", "arrays", "native_max_abs_difference",
                                  "unchanged_bound", "within_original_bound")}
                                for row in records[arm]]
        summary["arms"][arm] = status
        (args.output / "dispatch-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    default_cores = summary["arms"]["default"].get("actual_core_log", [])
    zen_cores = summary["arms"].get("zen", {}).get("actual_core_log", [])
    summary["avx512_discrimination_available"] = bool(
        "avx512f" in flags and default_cores and zen_cores
        and all(core.lower() in {"cooperlake", "skylakex", "sapphirerapids"}
                for core in default_cores)
        and all(core.lower() == "zen" for core in zen_cores))
    if set(records) == {"default", "zen"}:
        summary["stage_hash_equal"] = {
            left["origin"]: {key: value["sha256"] == right["arrays"][key]["sha256"]
                             for key, value in left["arrays"].items()}
            for left, right in zip(records["default"], records["zen"], strict=True)
        }
    summary["fixture_unchanged"] = all(
        arm.get("fixture_sha256", fixture_hash) == fixture_hash
        for arm in summary["arms"].values())
    (args.output / "dispatch-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("avx512_discrimination_available", "fixture_unchanged")}))
    # Capture failures are infrastructure errors; numerical native mismatch is
    # intentionally left to the original assertion in the next workflow step.
    if not summary["fixture_unchanged"] or any(
            "skipped" not in arm and arm.get("returncode") != 0
            for arm in summary["arms"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
