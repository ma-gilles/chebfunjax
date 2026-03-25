#!/usr/bin/env python3
# uses-numpy: timing statistics use numpy
"""GPU-specific benchmarks: CPU vs GPU speedup, vmap scaling.

Tests:
  - Evaluation at N points for N in {100, 1000, 10000, 100000, 1000000}
    on both CPU and GPU, measuring throughput in points/second.
  - vmap over batch of functions (1D evaluation).
  - 2D Chebfun2 evaluation grid sweep.

Run via the Slurm script ``bench_gpu.sh``, or directly::

    pixi run python benchmarks/bench_gpu_vmap.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description="GPU vmap scaling benchmarks")
parser.add_argument("--out", default=None, help="Output JSON path")
args, _unknown = parser.parse_known_args()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _barrier_timeit(fn, n_warmup: int = 2, n_rep: int = 10) -> dict:
    for _ in range(n_warmup):
        fn()
    jax.effects_barrier()
    times = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        jax.effects_barrier()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "reps": n_rep,
    }


# ---------------------------------------------------------------------------
# 1D eval scaling: N points on current device
# ---------------------------------------------------------------------------


def bench_eval_scaling() -> dict:
    from chebfunjax.chebfun1d.chebfun import chebfun

    f = chebfun(jnp.sin, domain=(-1.0, 1.0))
    Ns = [100, 1_000, 10_000, 100_000, 1_000_000]
    results = {}
    for N in Ns:
        xs = jnp.linspace(-1.0, 1.0, N, dtype=jnp.float64)
        try:
            res = _barrier_timeit(lambda xs=xs: f(xs), n_warmup=3, n_rep=15)
            res["N"] = N
            res["throughput_pts_per_sec"] = N / (res["mean_ms"] * 1e-3)
            results[str(N)] = res
            dev = str(jax.devices()[0]).split(":")[-1]
            print(
                f"  eval N={N:8d}  mean={res['mean_ms']:9.3f} ms"
                f"  throughput={res['throughput_pts_per_sec']:.2e} pts/s"
                f"  [{dev}]"
            )
        except Exception as exc:
            results[str(N)] = {"error": str(exc)}
            print(f"  eval N={N:8d}  ERROR: {exc}")
    return results


# ---------------------------------------------------------------------------
# vmap over batch of functions (batched coefficient arrays)
# ---------------------------------------------------------------------------


def bench_vmap_batch_eval() -> dict:
    """Evaluate a batch of Chebfun-like coefficient arrays at xs using vmap.

    Simulates the use case: you have B independent Chebfun approximations
    (different functions) all on the same grid. Evaluate all of them at
    the same xs using jax.vmap.
    """
    from chebfunjax.tech.chebtech import Chebtech2

    # Build B=32 independent random coefficient arrays (degree 64)
    rng = np.random.default_rng(42)
    B = 32
    n_coeffs = 65  # degree 64
    xs = jnp.linspace(-1.0, 1.0, 1000, dtype=jnp.float64)

    coeffs_batch = jnp.array(
        rng.standard_normal((B, n_coeffs)), dtype=jnp.float64
    )

    # Single function eval (no vmap)
    def eval_single(coeffs: jax.Array) -> jax.Array:
        tech = Chebtech2.from_coeffs(coeffs)
        return tech(xs)

    # Batched via vmap
    eval_batched = jax.vmap(eval_single)

    # Warm up
    _ = eval_batched(coeffs_batch)
    jax.effects_barrier()

    res_vmap = _barrier_timeit(
        lambda: eval_batched(coeffs_batch), n_warmup=3, n_rep=30
    )

    # Sequential baseline (loop over B)
    def eval_loop():
        outs = []
        for i in range(B):
            outs.append(eval_single(coeffs_batch[i]))
        return outs

    _ = eval_loop()
    res_loop = _barrier_timeit(eval_loop, n_warmup=2, n_rep=10)

    speedup = res_loop["mean_ms"] / max(res_vmap["mean_ms"], 1e-9)
    print(f"  vmap B={B} funcs at 1000pts:  vmap={res_vmap['mean_ms']:.3f} ms"
          f"  loop={res_loop['mean_ms']:.3f} ms  speedup={speedup:.1f}x")

    return {
        "B": B,
        "n_coeffs": n_coeffs,
        "n_eval_pts": 1000,
        "vmap": res_vmap,
        "loop": res_loop,
        "speedup": float(speedup),
    }


# ---------------------------------------------------------------------------
# 2D Chebfun2 eval on grids of increasing size
# ---------------------------------------------------------------------------


def bench_chebfun2_grid_eval() -> dict:
    from chebfunjax.chebfun2d.chebfun2 import chebfun2

    f = chebfun2(lambda x, y: jnp.cos(x + y))
    grid_sizes = [50, 100, 200, 500]
    results = {}
    for g in grid_sizes:
        xs = jnp.linspace(-1.0, 1.0, g, dtype=jnp.float64)
        ys = jnp.linspace(-1.0, 1.0, g, dtype=jnp.float64)
        try:
            res = _barrier_timeit(lambda xs=xs, ys=ys: f(xs, ys), n_warmup=2, n_rep=20)
            res["grid"] = g
            res["N"] = g * g
            res["throughput_pts_per_sec"] = (g * g) / (res["mean_ms"] * 1e-3)
            results[str(g)] = res
            print(
                f"  Chebfun2 grid {g:4d}x{g:<4d}  mean={res['mean_ms']:9.3f} ms"
                f"  throughput={res['throughput_pts_per_sec']:.2e} pts/s"
            )
        except Exception as exc:
            results[str(g)] = {"error": str(exc)}
            print(f"  Chebfun2 grid {g}x{g}  ERROR: {exc}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 80)
    print("chebfunjax GPU vmap scaling benchmarks")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 80)

    all_results: dict = {"device": str(jax.devices()[0])}

    print("\n--- 1D eval scaling (N points) ---")
    all_results["eval_scaling"] = bench_eval_scaling()

    print("\n--- vmap batch eval (B functions) ---")
    try:
        all_results["vmap_batch"] = bench_vmap_batch_eval()
    except Exception as exc:
        print(f"  ERROR: {exc}")
        all_results["vmap_batch"] = {"error": str(exc)}

    print("\n--- 2D Chebfun2 grid eval ---")
    try:
        all_results["chebfun2_grid"] = bench_chebfun2_grid_eval()
    except Exception as exc:
        print(f"  ERROR: {exc}")
        all_results["chebfun2_grid"] = {"error": str(exc)}

    out_path = args.out or str(Path(__file__).parent / "python_results_gpu_vmap.json")
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults saved to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
