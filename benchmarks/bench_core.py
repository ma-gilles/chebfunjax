# uses-numpy: construction benchmarks use numpy for timing and statistics
"""U102 — Core performance benchmarks for chebfunjax.

Times the key operations:
  - Chebfun construction (adaptive)
  - Chebfun evaluation (single point)
  - Chebfun diff (spectral differentiation)
  - Chebfun sum (definite integral)
  - Chebfun roots (eigenvalue solve)
  - Chebfun2 construction
  - Chebfun2 evaluation
  - Chebfun2 diff / sum2 / norm
  - Chebfun3 construction
  - Chebfun3 evaluation

Run from the repo root::

    pixi run python benchmarks/bench_core.py

or with ``pytest-benchmark`` (if installed)::

    pixi run pytest benchmarks/bench_core.py --benchmark-only

The script also runs as a standalone timing harness without pytest.

Provenance
----------
No direct MATLAB source (benchmarking infrastructure).
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

# Ensure float64
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _timeit(fn: Callable, n_warmup: int = 1, n_rep: int = 5) -> dict:
    """Time a callable, returning mean and std in milliseconds.

    Parameters
    ----------
    fn : callable
        Zero-argument callable to time.
    n_warmup : int
        Number of warm-up calls (to trigger JIT compilation etc.).
    n_rep : int
        Number of timed repetitions.

    Returns
    -------
    dict with keys 'mean_ms', 'std_ms', 'min_ms', 'reps'.
    """
    for _ in range(n_warmup):
        fn()
    # Block until previous JAX work completes
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


def _fmt(result: dict, label: str) -> str:
    """Format a timing result as a human-readable line."""
    return (
        f"{label:<45s}  "
        f"{result['mean_ms']:8.3f} ms  "
        f"(min={result['min_ms']:.3f} ms, "
        f"std={result['std_ms']:.3f} ms, "
        f"n={result['reps']})"
    )


# ---------------------------------------------------------------------------
# Individual benchmark functions
# ---------------------------------------------------------------------------


def bench_chebfun_construct_sin():
    """Construction of sin(x) Chebfun on [-1, 1]."""
    from chebfunjax.chebfun1d.chebfun import chebfun
    return _timeit(lambda: chebfun(jnp.sin, domain=(-1.0, 1.0)), n_warmup=2, n_rep=10)


def bench_chebfun_construct_runge():
    """Construction of Runge function 1/(1+25*x^2) — medium degree."""
    from chebfunjax.chebfun1d.chebfun import chebfun
    f_runge = lambda x: 1.0 / (1.0 + 25.0 * x ** 2)
    return _timeit(lambda: chebfun(f_runge, domain=(-1.0, 1.0)), n_warmup=2, n_rep=10)


def bench_chebfun_eval():
    """Evaluation of a sin Chebfun at 1000 points (JIT-compiled)."""
    from chebfunjax.chebfun1d.chebfun import chebfun
    f = chebfun(jnp.sin, domain=(-1.0, 1.0))
    xs = jnp.linspace(-1.0, 1.0, 1000, dtype=jnp.float64)
    # Warm up JIT
    _ = f(xs)
    jax.effects_barrier()
    return _timeit(lambda: f(xs), n_warmup=2, n_rep=20)


def bench_chebfun_diff():
    """Spectral differentiation of a sin Chebfun (returns new Chebfun)."""
    from chebfunjax.chebfun1d.chebfun import chebfun
    f = chebfun(lambda x: jnp.sin(10.0 * x), domain=(-1.0, 1.0))
    return _timeit(lambda: f.diff(), n_warmup=2, n_rep=20)


def bench_chebfun_sum():
    """Definite integral of a Chebfun on [-1, 1]."""
    from chebfunjax.chebfun1d.chebfun import chebfun
    f = chebfun(lambda x: jnp.exp(-x ** 2), domain=(-1.0, 1.0))
    return _timeit(lambda: f.sum(), n_warmup=2, n_rep=20)


def bench_chebfun_roots():
    """Rootfinding for a Chebfun with ~10 roots."""
    from chebfunjax.chebfun1d.chebfun import chebfun
    # sin(5*pi*x) has 10 roots on [-1, 1]
    f = chebfun(lambda x: jnp.sin(5.0 * jnp.pi * x), domain=(-1.0, 1.0))
    return _timeit(lambda: f.roots(), n_warmup=2, n_rep=10)


def bench_chebfun2_construct():
    """Construction of cos(x+y) Chebfun2 on [-1,1]^2."""
    from chebfunjax.chebfun2d.chebfun2 import chebfun2
    return _timeit(
        lambda: chebfun2(lambda x, y: jnp.cos(x + y)),
        n_warmup=1, n_rep=5,
    )


def bench_chebfun2_eval():
    """Evaluation of a Chebfun2 at 1000 points."""
    from chebfunjax.chebfun2d.chebfun2 import chebfun2
    f = chebfun2(lambda x, y: jnp.cos(x + y))
    xs = jnp.linspace(-1.0, 1.0, 1000, dtype=jnp.float64)
    ys = jnp.linspace(-1.0, 1.0, 1000, dtype=jnp.float64)
    _ = f(xs, ys)  # warm up JIT
    jax.effects_barrier()
    return _timeit(lambda: f(xs, ys), n_warmup=2, n_rep=20)


def bench_chebfun2_diff():
    """Partial derivative of a Chebfun2 (x-direction)."""
    from chebfunjax.chebfun2d.chebfun2 import chebfun2
    f = chebfun2(lambda x, y: jnp.exp(x * y))
    return _timeit(lambda: f.diff(dim=2), n_warmup=2, n_rep=20)


def bench_chebfun2_sum2():
    """Double integral of a Chebfun2 on [-1,1]^2."""
    from chebfunjax.chebfun2d.chebfun2 import chebfun2
    f = chebfun2(lambda x, y: jnp.cos(x + y))
    return _timeit(lambda: f.sum2(), n_warmup=2, n_rep=20)


def bench_chebfun2_norm():
    """Frobenius norm of a Chebfun2."""
    from chebfunjax.chebfun2d.chebfun2 import chebfun2
    f = chebfun2(lambda x, y: jnp.cos(x + y))
    return _timeit(lambda: f.norm(), n_warmup=2, n_rep=20)


def bench_chebfun3_construct():
    """Construction of cos(x+y+z) Chebfun3 on [-1,1]^3."""
    from chebfunjax.chebfun3d.chebfun3 import chebfun3
    return _timeit(
        lambda: chebfun3(lambda x, y, z: jnp.cos(x + y + z)),
        n_warmup=1, n_rep=3,
    )


def bench_chebfun3_eval():
    """Evaluation of a Chebfun3 at 500 points."""
    from chebfunjax.chebfun3d.chebfun3 import chebfun3
    f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
    xs = jnp.linspace(-1.0, 1.0, 500, dtype=jnp.float64)
    ys = jnp.linspace(-1.0, 1.0, 500, dtype=jnp.float64)
    zs = jnp.linspace(-1.0, 1.0, 500, dtype=jnp.float64)
    _ = f(xs, ys, zs)  # warm up JIT
    jax.effects_barrier()
    return _timeit(lambda: f(xs, ys, zs), n_warmup=2, n_rep=10)


# ---------------------------------------------------------------------------
# pytest-benchmark wrappers (optional)
# ---------------------------------------------------------------------------

# These functions are discovered by pytest when --benchmark-only is passed.
# Each calls a function from above and reports via pytest-benchmark.

try:
    import pytest

    @pytest.mark.benchmark(group="chebfun1d")
    def test_bench_chebfun_construct_sin(benchmark):
        from chebfunjax.chebfun1d.chebfun import chebfun
        benchmark(lambda: chebfun(jnp.sin, domain=(-1.0, 1.0)))

    @pytest.mark.benchmark(group="chebfun1d")
    def test_bench_chebfun_eval(benchmark):
        from chebfunjax.chebfun1d.chebfun import chebfun
        f = chebfun(jnp.sin, domain=(-1.0, 1.0))
        xs = jnp.linspace(-1.0, 1.0, 1000, dtype=jnp.float64)
        _ = f(xs)
        benchmark(lambda: f(xs))

    @pytest.mark.benchmark(group="chebfun1d")
    def test_bench_chebfun_diff(benchmark):
        from chebfunjax.chebfun1d.chebfun import chebfun
        f = chebfun(lambda x: jnp.sin(10.0 * x), domain=(-1.0, 1.0))
        benchmark(lambda: f.diff())

    @pytest.mark.benchmark(group="chebfun1d")
    def test_bench_chebfun_sum(benchmark):
        from chebfunjax.chebfun1d.chebfun import chebfun
        f = chebfun(lambda x: jnp.exp(-x ** 2), domain=(-1.0, 1.0))
        benchmark(lambda: f.sum())

    @pytest.mark.benchmark(group="chebfun1d")
    def test_bench_chebfun_roots(benchmark):
        from chebfunjax.chebfun1d.chebfun import chebfun
        f = chebfun(lambda x: jnp.sin(5.0 * jnp.pi * x), domain=(-1.0, 1.0))
        benchmark(lambda: f.roots())

    @pytest.mark.benchmark(group="chebfun2d")
    def test_bench_chebfun2_construct(benchmark):
        from chebfunjax.chebfun2d.chebfun2 import chebfun2
        benchmark(lambda: chebfun2(lambda x, y: jnp.cos(x + y)))

    @pytest.mark.benchmark(group="chebfun2d")
    def test_bench_chebfun2_eval(benchmark):
        from chebfunjax.chebfun2d.chebfun2 import chebfun2
        f = chebfun2(lambda x, y: jnp.cos(x + y))
        xs = jnp.linspace(-1.0, 1.0, 1000, dtype=jnp.float64)
        ys = jnp.linspace(-1.0, 1.0, 1000, dtype=jnp.float64)
        _ = f(xs, ys)
        benchmark(lambda: f(xs, ys))

    @pytest.mark.benchmark(group="chebfun3d")
    def test_bench_chebfun3_construct(benchmark):
        from chebfunjax.chebfun3d.chebfun3 import chebfun3
        benchmark(lambda: chebfun3(lambda x, y, z: jnp.cos(x + y + z)))

    @pytest.mark.benchmark(group="chebfun3d")
    def test_bench_chebfun3_eval(benchmark):
        from chebfunjax.chebfun3d.chebfun3 import chebfun3
        f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
        xs = jnp.linspace(-1.0, 1.0, 500, dtype=jnp.float64)
        ys = jnp.linspace(-1.0, 1.0, 500, dtype=jnp.float64)
        zs = jnp.linspace(-1.0, 1.0, 500, dtype=jnp.float64)
        _ = f(xs, ys, zs)
        benchmark(lambda: f(xs, ys, zs))

except ImportError:
    # pytest not available — benchmarks run in standalone mode only
    pass


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

BENCHMARKS = [
    ("Chebfun1D: construct sin(x)", bench_chebfun_construct_sin),
    ("Chebfun1D: construct Runge 1/(1+25x^2)", bench_chebfun_construct_runge),
    ("Chebfun1D: eval at 1000 pts", bench_chebfun_eval),
    ("Chebfun1D: diff()", bench_chebfun_diff),
    ("Chebfun1D: sum()", bench_chebfun_sum),
    ("Chebfun1D: roots()", bench_chebfun_roots),
    ("Chebfun2D: construct cos(x+y)", bench_chebfun2_construct),
    ("Chebfun2D: eval at 1000 pts", bench_chebfun2_eval),
    ("Chebfun2D: diff(dim=2)", bench_chebfun2_diff),
    ("Chebfun2D: sum2()", bench_chebfun2_sum2),
    ("Chebfun2D: norm()", bench_chebfun2_norm),
    ("Chebfun3D: construct cos(x+y+z)", bench_chebfun3_construct),
    ("Chebfun3D: eval at 500 pts", bench_chebfun3_eval),
]


if __name__ == "__main__":
    print("=" * 80)
    print("chebfunjax core benchmarks")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 80)
    for label, fn in BENCHMARKS:
        try:
            result = fn()
            print(_fmt(result, label))
        except Exception as exc:
            print(f"  {label:<43s}  ERROR: {exc}")
    print("=" * 80)
