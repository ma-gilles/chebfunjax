#!/usr/bin/env python3
# uses-numpy: table formatting uses numpy
"""Summarize benchmark results and generate RESULTS.md.

Reads:
  - benchmarks/matlab_results.json
  - benchmarks/python_results_cpu.json   (or python_results.json)
  - benchmarks/python_results_gpu.json   (optional)
  - benchmarks/python_results_gpu_vmap.json  (optional)

Prints a comparison table and writes ``benchmarks/RESULTS.md``.

Usage::

    pixi run python benchmarks/summarize_results.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BENCH_DIR = Path(__file__).parent


def _load(path: Path) -> dict:
    if path.exists():
        with path.open() as fh:
            return json.load(fh)
    return {}


def _warm(d: dict) -> float | None:
    """Extract warm mean ms from a timing dict."""
    if not d or "error" in d:
        return None
    return d.get("warm_mean_ms") or d.get("mean_ms")


def _fmt_ms(v: float | None, precision: int = 3) -> str:
    if v is None:
        return "N/A"
    if v < 0.001:
        return f"{v*1000:.2f} µs"
    if v < 1.0:
        return f"{v:.4f} ms"
    return f"{v:.{precision}f} ms"


def _fmt_ratio(py: float | None, matlab: float | None) -> str:
    if py is None or matlab is None:
        return "N/A"
    r = matlab / py
    if r >= 10:
        return f"{r:.0f}x faster"
    elif r >= 1.0:
        return f"{r:.1f}x faster"
    else:
        return f"{1/r:.1f}x slower"


def main() -> None:
    matlab = _load(BENCH_DIR / "matlab_results.json")
    cpu_path = (BENCH_DIR / "python_results_cpu.json")
    if not cpu_path.exists():
        cpu_path = BENCH_DIR / "python_results.json"
    cpu = _load(cpu_path)
    gpu = _load(BENCH_DIR / "python_results_gpu.json")
    vmap = _load(BENCH_DIR / "python_results_gpu_vmap.json")

    # ---- 1D benchmark rows ----
    rows_1d = [
        ("construct sin(x)", "construct_sin_ms"),
        ("eval f(0.5) single pt", "eval_single_ms"),
        ("eval f(1000 pts)", "eval_1000pts_ms"),
        ("diff sin(10x)", "diff_ms"),
        ("sum exp(-x²)", "sum_ms"),
        ("roots sin(5πx)", "roots_ms"),
    ]

    rows_2d = [
        ("construct cos(x+y)", "chebfun2_construct_ms"),
        ("eval (1000 pts)", "chebfun2_eval_ms"),
        ("diff (x-direction)", "chebfun2_diff_ms"),
        ("sum2", "chebfun2_sum2_ms"),
        ("norm", "chebfun2_norm_ms"),
    ]

    rows_3d = [
        ("construct cos(x+y+z)", "chebfun3_construct_ms"),
        ("eval (500 pts)", "chebfun3_eval_ms"),
    ]

    has_gpu = bool(gpu)

    def table_header():
        if has_gpu:
            return (
                "| Operation | MATLAB (ms) | Python CPU (ms) | Python GPU (ms) |"
                " CPU speedup | GPU speedup |\n"
                "|-----------|------------|----------------|----------------|"
                "-------------|-------------|\n"
            )
        else:
            return (
                "| Operation | MATLAB (ms) | Python CPU (ms) | CPU speedup |\n"
                "|-----------|------------|----------------|-------------|\n"
            )

    def table_row(label: str, key: str):
        m_ms = matlab.get(key)
        c_ms = _warm(cpu.get(key))
        g_ms = _warm(gpu.get(key)) if has_gpu else None
        if has_gpu:
            return (
                f"| {label} | {_fmt_ms(m_ms)} | {_fmt_ms(c_ms)} | {_fmt_ms(g_ms)} |"
                f" {_fmt_ratio(c_ms, m_ms)} | {_fmt_ratio(g_ms, m_ms)} |\n"
            )
        else:
            return (
                f"| {label} | {_fmt_ms(m_ms)} | {_fmt_ms(c_ms)} |"
                f" {_fmt_ratio(c_ms, m_ms)} |\n"
            )

    # Build scaling table
    scale_section = ""
    scale_cpu = (cpu.get("scaling") or {})
    scale_gpu = (gpu.get("scaling") or {}) if has_gpu else {}
    scale_matlab_keys = {"10": "construct_n10_ms", "100": "construct_n100_ms",
                         "1000": "construct_n1000_ms"}
    if scale_cpu:
        scale_section = "\n### 1D Construction Scaling (fixed degree)\n\n"
        if has_gpu:
            scale_section += (
                "| n (degree) | MATLAB (ms) | CPU (ms) | GPU (ms) | CPU speedup | GPU speedup |\n"
                "|-----------|------------|---------|---------|-------------|-------------|\n"
            )
        else:
            scale_section += (
                "| n (degree) | MATLAB (ms) | CPU (ms) | CPU speedup |\n"
                "|-----------|------------|---------|-------------|\n"
            )
        for n_str in ["10", "100", "1000", "10000"]:
            m_key = scale_matlab_keys.get(n_str)
            m_ms = matlab.get(m_key) if m_key else None
            c_ms = _warm(scale_cpu.get(n_str))
            g_ms = _warm(scale_gpu.get(n_str)) if has_gpu else None
            if has_gpu:
                scale_section += (
                    f"| n={n_str:<6} | {_fmt_ms(m_ms)} | {_fmt_ms(c_ms)} | {_fmt_ms(g_ms)} |"
                    f" {_fmt_ratio(c_ms, m_ms)} | {_fmt_ratio(g_ms, m_ms)} |\n"
                )
            else:
                scale_section += (
                    f"| n={n_str:<6} | {_fmt_ms(m_ms)} | {_fmt_ms(c_ms)} |"
                    f" {_fmt_ratio(c_ms, m_ms)} |\n"
                )

    # vmap/GPU section
    vmap_section = ""
    if vmap:
        eval_scaling = vmap.get("eval_scaling", {})
        if eval_scaling:
            vmap_section += "\n### GPU Evaluation Throughput (N points)\n\n"
            vmap_section += "| N points | Mean (ms) | Throughput (pts/s) |\n"
            vmap_section += "|---------|---------|-------------------|\n"
            for N_str in ["100", "1000", "10000", "100000", "1000000"]:
                entry = eval_scaling.get(N_str, {})
                if "error" in entry:
                    continue
                m_ms = entry.get("mean_ms")
                tp = entry.get("throughput_pts_per_sec")
                tp_str = f"{tp:.2e}" if tp else "N/A"
                vmap_section += f"| {N_str} | {_fmt_ms(m_ms)} | {tp_str} |\n"

        vmap_batch = vmap.get("vmap_batch", {})
        if vmap_batch and "speedup" in vmap_batch:
            B = vmap_batch["B"]
            vm = vmap_batch.get("vmap", {})
            lp = vmap_batch.get("loop", {})
            sp = vmap_batch["speedup"]
            vmap_section += f"\n### vmap Batch Eval ({B} functions, 1000 pts each)\n\n"
            vmap_section += "| Mode | Mean (ms) | Speedup |\n"
            vmap_section += "|------|---------|--------|\n"
            vmap_section += f"| loop (sequential) | {_fmt_ms(_warm(lp))} | 1.0x |\n"
            vmap_section += f"| vmap (batched) | {_fmt_ms(_warm(vm))} | {sp:.1f}x |\n"

        grid_eval = vmap.get("chebfun2_grid", {})
        if grid_eval:
            vmap_section += "\n### 2D Chebfun2 Grid Evaluation Throughput\n\n"
            vmap_section += "| Grid | N total pts | Mean (ms) | Throughput (pts/s) |\n"
            vmap_section += "|------|------------|---------|-------------------|\n"
            for g_str in ["50", "100", "200", "500"]:
                entry = grid_eval.get(g_str, {})
                if "error" in entry:
                    continue
                g = int(g_str)
                m_ms = entry.get("mean_ms")
                tp = entry.get("throughput_pts_per_sec")
                tp_str = f"{tp:.2e}" if tp else "N/A"
                vmap_section += f"| {g}×{g} | {g*g} | {_fmt_ms(m_ms)} | {tp_str} |\n"

    # ---- Device info ----
    cpu_device = cpu.get("device", "CPU")
    gpu_device = gpu.get("device", "GPU") if has_gpu else None

    # ---- Compose RESULTS.md ----
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = f"""# chebfunjax Benchmark Results

> Generated: {now}

## System

- **Python/JAX CPU device**: `{cpu_device}`
"""
    if gpu_device:
        md += f"- **GPU device**: `{gpu_device}`\n"
    md += """
## Notes

- **MATLAB** timings: warm loop averages from `matlab_timing.m` (R2025b,
  Chebfun commit 7574c77 equivalent).
- **Python CPU** timings: warm mean (after JIT compilation) from
  `bench_comparison.py --device cpu`.
- **Python GPU** timings: warm mean from `bench_comparison.py --device gpu`.
- *Speedup* = MATLAB_time / Python_warm_mean. Values > 1 mean chebfunjax is
  faster; values < 1 mean MATLAB is faster.
- First-call (JIT compile) times are **not** shown here; see the raw JSON
  files for first-call overhead.

## 1D Chebfun Results

"""
    md += table_header()
    for label, key in rows_1d:
        md += table_row(label, key)

    md += scale_section

    md += "\n## 2D Chebfun2 Results\n\n"
    md += table_header()
    for label, key in rows_2d:
        md += table_row(label, key)

    md += "\n## 3D Chebfun3 Results\n\n"
    md += table_header()
    for label, key in rows_3d:
        md += table_row(label, key)

    if vmap_section:
        md += "\n## GPU & vmap Benchmarks\n"
        md += vmap_section

    md += """
## How to Reproduce

```bash
# 1. MATLAB timings (requires MATLAB R2025b)
cd /scratch/gpfs/GILLES/mg6942/jaxchebfun
module load matlab/R2025b
matlab -batch "run('benchmarks/matlab_timing.m')"

# 2. Python CPU benchmarks
pixi run python benchmarks/bench_comparison.py --device cpu

# 3. GPU benchmarks (Slurm)
sbatch benchmarks/bench_gpu.sh
# outputs saved to benchmarks/python_results_{cpu,gpu}.json
# and benchmarks/python_results_gpu_vmap.json

# 4. Re-generate this table
pixi run python benchmarks/summarize_results.py
```

## Raw Data

- [`matlab_results.json`](matlab_results.json) — MATLAB reference timings
- [`python_results_cpu.json`](python_results_cpu.json) — Python/JAX CPU (warm + first-call)
- [`python_results_gpu.json`](python_results_gpu.json) — Python/JAX GPU (warm + first-call)
- [`python_results_gpu_vmap.json`](python_results_gpu_vmap.json) — GPU vmap scaling
"""

    out_path = BENCH_DIR / "RESULTS.md"
    out_path.write_text(md)
    print(f"RESULTS.md written to {out_path}")

    # Also print a brief summary to stdout
    print("\n=== Summary ===")
    print(f"{'Operation':<40s}  {'MATLAB':>10s}  {'CPU':>10s}", end="")
    if has_gpu:
        print(f"  {'GPU':>10s}", end="")
    print()
    print("-" * (65 + (12 if has_gpu else 0)))
    for label, key in rows_1d + rows_2d + rows_3d:
        m_ms = matlab.get(key)
        c_ms = _warm(cpu.get(key))
        g_ms = _warm(gpu.get(key)) if has_gpu else None
        print(
            f"  {label:<38s}  {_fmt_ms(m_ms):>10s}  {_fmt_ms(c_ms):>10s}",
            end="",
        )
        if has_gpu:
            print(f"  {_fmt_ms(g_ms):>10s}", end="")
        print()


if __name__ == "__main__":
    main()
