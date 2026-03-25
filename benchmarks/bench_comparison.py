#!/usr/bin/env python3
# uses-numpy: timing statistics and result tables use numpy/scipy
"""Comparison benchmarks: chebfunjax vs MATLAB Chebfun reference timings.

Measures the same operations benchmarked in ``matlab_timing.m``:

  1D Chebfun
  - construction of sin(x)
  - evaluation at a single point
  - evaluation at 1000 points
  - differentiation (diff)
  - integration (sum)
  - rootfinding

  1D Scaling
  - construction with fixed degree n in {10, 100, 1000, 10000}

  2D Chebfun2
  - construction, eval, diff, sum2, norm

  3D Chebfun3
  - construction, eval

For each operation this script reports:
  - first-call time (includes JIT compilation overhead)
  - warm (steady-state) time

If ``benchmarks/matlab_results.json`` exists, a comparison table is printed
showing speedup/slowdown relative to MATLAB.

Run::

    pixi run python benchmarks/bench_comparison.py
    pixi run python benchmarks/bench_comparison.py --device cpu   # force CPU
    pixi run python benchmarks/bench_comparison.py --device gpu   # force GPU

Output is also saved to ``benchmarks/python_results.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Parse arguments before JAX import so --device can set platform
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="chebfunjax comparison benchmarks")
parser.add_argument(
    "--device",
    choices=["cpu", "gpu", "auto"],
    default="auto",
    help="Force JAX device backend (default: auto)",
)
parser.add_argument(
    "--out",
    default=None,
    help="Path for JSON output (default: benchmarks/python_results.json)",
)
args, _unknown = parser.parse_known_args()

if args.device == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
elif args.device == "gpu":
    os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax  # noqa: E402 (must come after env setup)
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def timeit_first_and_warm(
    fn: Callable,
    n_warmup: int = 1,
    n_rep: int = 10,
) -> dict:
    """Return first-call and warm timings in milliseconds.

    Parameters
    ----------
    fn :
        Zero-argument callable to time.
    n_warmup :
        Number of calls used to reach steady state (not counted in warm stats).
    n_rep :
        Number of timed repetitions for the warm measurement.

    Returns
    -------
    dict with keys:
        first_ms  – wall time of the very first call (JIT compile included)
        warm_mean_ms, warm_std_ms, warm_min_ms
        reps
    """
    # First call (cold, includes JIT)
    t0 = time.perf_counter()
    fn()
    jax.effects_barrier()
    first_ms = (time.perf_counter() - t0) * 1000.0

    # Additional warm-up calls
    for _ in range(max(n_warmup - 1, 0)):
        fn()
    jax.effects_barrier()

    # Timed warm calls
    times = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        jax.effects_barrier()
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times)
    return {
        "first_ms": float(first_ms),
        "warm_mean_ms": float(np.mean(arr)),
        "warm_std_ms": float(np.std(arr)),
        "warm_min_ms": float(np.min(arr)),
        "reps": n_rep,
    }


def _fmt_row(
    label: str,
    res: dict,
    matlab_ms: float | None = None,
) -> str:
    """Format a timing row for the summary table."""
    cols = [
        f"{label:<48s}",
        f"first={res['first_ms']:9.2f} ms",
        f"warm={res['warm_mean_ms']:9.3f} ms",
        f"(min={res['warm_min_ms']:.3f}, std={res['warm_std_ms']:.3f})",
    ]
    if matlab_ms is not None:
        ratio = matlab_ms / max(res["warm_mean_ms"], 1e-9)
        cols.append(f"  MATLAB={matlab_ms:.3f} ms  speedup={ratio:.1f}x")
    return "  ".join(cols)


# ---------------------------------------------------------------------------
# 1D Chebfun benchmarks
# ---------------------------------------------------------------------------


def bench_1d_construct_sin() -> dict:
    from chebfunjax.chebfun1d.chebfun import chebfun
    return timeit_first_and_warm(
        lambda: chebfun(jnp.sin, domain=(-1.0, 1.0)),
        n_warmup=3, n_rep=15,
    )


def bench_1d_eval_single() -> dict:
    from chebfunjax.chebfun1d.chebfun import chebfun
    f = chebfun(jnp.sin, domain=(-1.0, 1.0))
    x = jnp.array(0.5, dtype=jnp.float64)
    return timeit_first_and_warm(lambda: f(x), n_warmup=3, n_rep=200)


def bench_1d_eval_1000pts() -> dict:
    from chebfunjax.chebfun1d.chebfun import chebfun
    f = chebfun(jnp.sin, domain=(-1.0, 1.0))
    xs = jnp.linspace(-1.0, 1.0, 1000, dtype=jnp.float64)
    return timeit_first_and_warm(lambda: f(xs), n_warmup=3, n_rep=50)


def bench_1d_diff() -> dict:
    from chebfunjax.chebfun1d.chebfun import chebfun
    f = chebfun(lambda x: jnp.sin(10.0 * x), domain=(-1.0, 1.0))
    return timeit_first_and_warm(lambda: f.diff(), n_warmup=3, n_rep=30)


def bench_1d_sum() -> dict:
    from chebfunjax.chebfun1d.chebfun import chebfun
    f = chebfun(lambda x: jnp.exp(-(x ** 2)), domain=(-1.0, 1.0))
    return timeit_first_and_warm(lambda: f.sum(), n_warmup=3, n_rep=100)


def bench_1d_roots() -> dict:
    from chebfunjax.chebfun1d.chebfun import chebfun
    f = chebfun(lambda x: jnp.sin(5.0 * jnp.pi * x), domain=(-1.0, 1.0))
    return timeit_first_and_warm(lambda: f.roots(), n_warmup=2, n_rep=15)


# ---------------------------------------------------------------------------
# 1D Scaling benchmarks (fixed-degree construction)
# ---------------------------------------------------------------------------

_SCALE_DEGREES = [10, 100, 1000, 10000]


def _bench_1d_construct_fixed_n(n: int) -> dict:
    """Construct a chebfun with exactly n+1 coefficients (trunc to n+1)."""
    from chebfunjax.chebfun1d.chebfun import chebfun

    def fn():
        return chebfun(jnp.sin, domain=(-1.0, 1.0), n=n)

    n_rep = max(3, 30 // max(1, n // 100))
    return timeit_first_and_warm(fn, n_warmup=2, n_rep=n_rep)


def bench_1d_scaling() -> dict[int, dict]:
    results = {}
    for n in _SCALE_DEGREES:
        try:
            results[n] = _bench_1d_construct_fixed_n(n)
        except Exception as exc:
            results[n] = {"error": str(exc)}
    return results


# ---------------------------------------------------------------------------
# 2D Chebfun2 benchmarks
# ---------------------------------------------------------------------------


def bench_2d_construct() -> dict:
    from chebfunjax.chebfun2d.chebfun2 import chebfun2
    return timeit_first_and_warm(
        lambda: chebfun2(lambda x, y: jnp.cos(x + y)),
        n_warmup=2, n_rep=8,
    )


def bench_2d_eval_1000pts() -> dict:
    from chebfunjax.chebfun2d.chebfun2 import chebfun2
    f = chebfun2(lambda x, y: jnp.cos(x + y))
    xs = jnp.linspace(-1.0, 1.0, 1000, dtype=jnp.float64)
    ys = jnp.linspace(-1.0, 1.0, 1000, dtype=jnp.float64)
    return timeit_first_and_warm(lambda: f(xs, ys), n_warmup=3, n_rep=30)


def bench_2d_diff() -> dict:
    from chebfunjax.chebfun2d.chebfun2 import chebfun2
    f = chebfun2(lambda x, y: jnp.exp(x * y))
    return timeit_first_and_warm(lambda: f.diff(dim=2), n_warmup=2, n_rep=30)


def bench_2d_sum2() -> dict:
    from chebfunjax.chebfun2d.chebfun2 import chebfun2
    f = chebfun2(lambda x, y: jnp.cos(x + y))
    return timeit_first_and_warm(lambda: f.sum2(), n_warmup=3, n_rep=100)


def bench_2d_norm() -> dict:
    from chebfunjax.chebfun2d.chebfun2 import chebfun2
    f = chebfun2(lambda x, y: jnp.cos(x + y))
    return timeit_first_and_warm(lambda: f.norm(), n_warmup=3, n_rep=50)


# ---------------------------------------------------------------------------
# 3D Chebfun3 benchmarks
# ---------------------------------------------------------------------------


def bench_3d_construct() -> dict:
    from chebfunjax.chebfun3d.chebfun3 import chebfun3
    return timeit_first_and_warm(
        lambda: chebfun3(lambda x, y, z: jnp.cos(x + y + z)),
        n_warmup=1, n_rep=5,
    )


def bench_3d_eval_500pts() -> dict:
    from chebfunjax.chebfun3d.chebfun3 import chebfun3
    f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
    xs = jnp.linspace(-1.0, 1.0, 500, dtype=jnp.float64)
    ys = jnp.linspace(-1.0, 1.0, 500, dtype=jnp.float64)
    zs = jnp.linspace(-1.0, 1.0, 500, dtype=jnp.float64)
    return timeit_first_and_warm(lambda: f(xs, ys, zs), n_warmup=2, n_rep=15)


# ---------------------------------------------------------------------------
# vmap speedup benchmark (1D eval)
# ---------------------------------------------------------------------------


def bench_vmap_eval() -> dict:
    """Time batched evaluation via jax.vmap over 10000 points."""
    from chebfunjax.chebfun1d.chebfun import chebfun

    f = chebfun(jnp.sin, domain=(-1.0, 1.0))
    xs_large = jnp.linspace(-1.0, 1.0, 10_000, dtype=jnp.float64)

    # Standard batched eval
    std = timeit_first_and_warm(lambda: f(xs_large), n_warmup=3, n_rep=30)

    return {"standard_10k": std}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 90)
    print("chebfunjax comparison benchmarks")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX version: {jax.__version__}")
    print("=" * 90)

    # Load MATLAB reference if available
    matlab_path = Path(__file__).parent / "matlab_results.json"
    matlab: dict = {}
    if matlab_path.exists():
        with matlab_path.open() as fh:
            matlab = json.load(fh)
        print(f"[MATLAB reference loaded from {matlab_path}]")
    else:
        print("[No matlab_results.json found — speedup column will be omitted]")
    print()

    all_results: dict = {"device": str(jax.devices()[0])}

    # ---- 1D ----
    print("--- 1D Chebfun ---")
    benchmarks_1d = [
        ("construct sin(x)", bench_1d_construct_sin, "construct_sin_ms"),
        ("eval f(0.5)", bench_1d_eval_single, "eval_single_ms"),
        ("eval f(1000 pts)", bench_1d_eval_1000pts, "eval_1000pts_ms"),
        ("diff sin(10x)", bench_1d_diff, "diff_ms"),
        ("sum exp(-x^2)", bench_1d_sum, "sum_ms"),
        ("roots sin(5*pi*x)", bench_1d_roots, "roots_ms"),
    ]
    for label, fn, matlab_key in benchmarks_1d:
        try:
            res = fn()
            matlab_ms = matlab.get(matlab_key)
            print(_fmt_row(f"  1D {label}", res, matlab_ms))
            all_results[matlab_key] = res
        except Exception as exc:
            print(f"  1D {label:<44s}  ERROR: {exc}")
    print()

    # ---- 1D Scaling ----
    print("--- 1D Scaling (fixed-degree construction) ---")
    try:
        scale_results = bench_1d_scaling()
        for n, res in scale_results.items():
            if "error" in res:
                print(f"  1D construct (n={n:5d})  ERROR: {res['error']}")
                continue
            matlab_key = f"construct_n{n}_ms"
            matlab_ms = matlab.get(matlab_key)
            print(_fmt_row(f"  1D construct (n={n:5d})", res, matlab_ms))
        all_results["scaling"] = {str(n): r for n, r in scale_results.items()}
    except Exception as exc:
        print(f"  ERROR in scaling benchmarks: {exc}")
    print()

    # ---- 2D ----
    print("--- 2D Chebfun2 ---")
    benchmarks_2d = [
        ("construct cos(x+y)", bench_2d_construct, "chebfun2_construct_ms"),
        ("eval (1000 pts)", bench_2d_eval_1000pts, "chebfun2_eval_ms"),
        ("diff (x-direction)", bench_2d_diff, "chebfun2_diff_ms"),
        ("sum2", bench_2d_sum2, "chebfun2_sum2_ms"),
        ("norm", bench_2d_norm, "chebfun2_norm_ms"),
    ]
    for label, fn, matlab_key in benchmarks_2d:
        try:
            res = fn()
            matlab_ms = matlab.get(matlab_key)
            print(_fmt_row(f"  2D {label}", res, matlab_ms))
            all_results[matlab_key] = res
        except Exception as exc:
            print(f"  2D {label:<44s}  ERROR: {exc}")
    print()

    # ---- 3D ----
    print("--- 3D Chebfun3 ---")
    benchmarks_3d = [
        ("construct cos(x+y+z)", bench_3d_construct, "chebfun3_construct_ms"),
        ("eval (500 pts)", bench_3d_eval_500pts, "chebfun3_eval_ms"),
    ]
    for label, fn, matlab_key in benchmarks_3d:
        try:
            res = fn()
            matlab_ms = matlab.get(matlab_key)
            print(_fmt_row(f"  3D {label}", res, matlab_ms))
            all_results[matlab_key] = res
        except Exception as exc:
            print(f"  3D {label:<44s}  ERROR: {exc}")
    print()

    # ---- vmap ----
    print("--- vmap speedup ---")
    try:
        vmap_res = bench_vmap_eval()
        std10k = vmap_res["standard_10k"]
        print(_fmt_row("  1D eval (10000 pts, standard)", std10k, None))
        all_results["vmap"] = vmap_res
    except Exception as exc:
        print(f"  ERROR in vmap benchmarks: {exc}")
    print()

    # ---- Save JSON ----
    out_path = args.out or str(Path(__file__).parent / "python_results.json")
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"Results saved to {out_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
