"""Diagnostic-only CPU LU staging; never changes scientific acceptance bounds.

Run ``python -m tests.piecewise_lu_diagnostic --output PATH`` in a fresh
process. The existing failing test can call ``capture`` inside its worker.
Separate staged primitives diagnose, but do not replace, the production solve.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from chebfunjax.operators.chebop import (
    _piecewise_row_scaled_solve,
    _piecewise_row_scaled_system,
)


def _runtime():
    selected_env = (
        "XLA_FLAGS", "JAX_ENABLE_X64", "JAX_PLATFORMS",
        "JAX_DEFAULT_MATMUL_PRECISION", "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "OPENBLAS_CORETYPE", "MKL_NUM_THREADS",
        "MKL_CBWR", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
        "PYTEST_XDIST_WORKER", "PYTHONNOUSERSITE",
    )
    cpu = {}
    try:
        first = Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0]
        for line in first.splitlines():
            key, _, value = line.partition(":")
            if key.strip() in ("vendor_id", "model name", "cpu family", "model",
                               "stepping", "flags", "microcode"):
                cpu[key.strip()] = value.strip()
    except OSError as error:
        cpu["read_error"] = str(error)
    libraries = []
    try:
        libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps")
                            .read_text().splitlines()
                            if "/" in line and any(s in line.lower() for s in
                            ("openblas", "libblas", "liblapack", "libmkl", "blis", "_lapack"))})
    except OSError:
        pass
    try:
        from threadpoolctl import threadpool_info
        threadpools = threadpool_info()
    except ImportError:
        threadpools = {"unavailable": "threadpoolctl not installed"}
    binary_hashes = {}
    for name in libraries:
        try:
            digest = hashlib.sha256()
            with Path(name).open("rb") as stream:
                for chunk in iter(lambda: stream.read(65536), b""):
                    digest.update(chunk)
            binary_hashes[name] = digest.hexdigest()
        except OSError as error:
            binary_hashes[name] = {"read_error": str(error)}
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "cpu": cpu,
        "jax": jax.__version__, "jaxlib": jaxlib.__version__,
        "numpy": np.__version__, "scipy": getattr(sys.modules.get("scipy"), "__version__", None),
        "scipy_distribution_version": importlib.metadata.version("scipy"),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "config": {name: getattr(jax.config, name, None) for name in
                   ("jax_enable_x64", "jax_default_matmul_precision", "jax_disable_jit",
                    "jax_numpy_dtype_promotion", "jax_numpy_rank_promotion")},
        "environment": {key: os.environ[key] for key in selected_env if key in os.environ},
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "loaded_blas_libraries": libraries, "blas_binary_sha256": binary_hashes,
        "module_origins": {name: getattr(sys.modules.get(name), "__file__", None)
                           for name in ("jax", "jaxlib", "numpy", "scipy",
                                        "scipy.linalg.cython_lapack")},
        "threadpools": threadpools,
    }


def capture(matrix, rhs, expected, *, solution=None, origin="native", phase="worker"):
    """Record unchanged production output and independently staged LU primitives."""
    a, b = _piecewise_row_scaled_system(matrix, rhs)
    if solution is None:
        solution = _piecewise_row_scaled_solve(matrix, rhs)
    packed, pivots, permutation = jax.lax.linalg.lu(a)
    permuted_rhs = b[permutation].reshape((b.shape[0], -1))
    lower_result = jax.lax.linalg.triangular_solve(
        packed, permuted_rhs, left_side=True, lower=True, unit_diagonal=True)
    upper_result = jax.lax.linalg.triangular_solve(
        packed, lower_result, left_side=True, lower=False).reshape(b.shape)
    arrays = {key: np.asarray(value) for key, value in {
        "matrix": matrix, "rhs": rhs, "expected": expected,
        "scaled_matrix": a, "scaled_rhs": b, "solution": solution,
        "packed_lu": packed, "pivots": pivots, "permutation": permutation,
        "permuted_rhs": permuted_rhs, "lower_result": lower_result,
        "staged_solution": upper_result,
    }.items()}
    archive = io.BytesIO()
    np.savez_compressed(archive, **arrays)
    error = float(np.max(np.abs(arrays["solution"] - arrays["expected"])))
    bound = float(100 * np.finfo(np.float64).eps)
    return {
        "phase": phase, "origin": origin, "runtime": _runtime(),
        "native_max_abs_difference": error, "unchanged_bound": bound,
        "within_original_bound": error <= bound,
        "staged_matches_production": bool(np.array_equal(arrays["staged_solution"], arrays["solution"])),
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                          "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
                   for key, value in arrays.items()},
        "npz_base64": base64.b64encode(archive.getvalue()).decode("ascii"),
        "scope": "Staged primitives may compile separately from production; equality is recorded, not assumed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).parent /
                        "references/piecewise_variable_native_lu.npz")
    args = parser.parse_args()
    with np.load(args.fixture) as data:
        records = [capture(jnp.asarray(data[origin + "_matrix"]),
                           jnp.asarray(data[origin + "_rhs"]),
                           data[origin + "_native_solution"], origin=origin,
                           phase="fresh-process") for origin in ("native", "jax")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    for record in records:
        print(json.dumps({k: record[k] for k in
                          ("phase", "origin", "native_max_abs_difference",
                           "unchanged_bound", "within_original_bound",
                           "staged_matches_production")}))
    # Diagnostic only: unchanged pytest assertion remains authoritative.


if __name__ == "__main__":
    main()
