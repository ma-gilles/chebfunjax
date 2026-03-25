"""Speed and accuracy of Chebfun rootfinding.

Tests the colleague-matrix rootfinding algorithm on functions with known roots,
measuring accuracy as a function of polynomial degree.

Credit: Inspired by Chebfun example roots/RootsSpeed.m
(Jared Aurentz and Nick Trefethen, May 2014).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj


def run():
    print("=" * 60)
    print("Speed and accuracy of Chebfun rootfinding")
    print("=" * 60)

    pi = float(jnp.pi)

    # Test 1: sin(n*pi*x) has n roots on [-1,1] at x_k = k/n, k=-n..n
    print("\nTest: sin(n*pi*x) on [-1,1]")
    print(f"  {'n':>6}  {'roots found':>12}  {'max err':>12}  {'time (ms)':>10}")

    ns = [5, 10, 20, 50, 100]
    errs = []
    times = []
    for n in ns:
        f = cj.chebfun(lambda x, _n=n: jnp.sin(_n * pi * x))
        t0 = time.time()
        r = f.roots()
        elapsed = (time.time() - t0) * 1000  # ms
        r_arr = np.sort(np.array(r))
        # Exact roots: k/n for k = -n..n (includes endpoints ±1 if n is integer)
        exact_all = np.arange(-n, n + 1) / n
        # Chebfun may or may not return endpoint roots; check all returned roots against exact
        n_found = len(r_arr)
        # Every returned root should match some exact root
        max_err = 0.0
        for ri in r_arr:
            min_dist = np.min(np.abs(exact_all - ri))
            max_err = max(max_err, min_dist)
        errs.append(max_err)
        times.append(elapsed)
        # The number of found roots should be at least 2n-1 (interior) and at most 2n+1
        n_interior = 2 * n - 1  # roots strictly inside (-1,1)
        print(f"  {n:>6}  {n_found:>12}  {max_err:>12.2e}  {elapsed:>10.1f}")
        assert n_found >= n_interior, f"Too few roots for n={n}: {n_found} < {n_interior}"
        assert max_err < 1e-8, f"Root error {max_err} for n={n}"

    # Test 2: roots of T_n (Chebyshev polynomial of degree n)
    print("\nTest: Chebyshev T_n (n roots at cos((2k-1)*pi/(2n)))")
    for n in [10, 30, 50]:
        coeffs = jnp.zeros(n + 1).at[n].set(1.0)
        f = cj.Chebfun.from_coeffs(coeffs)
        t0 = time.time()
        r = f.roots()
        elapsed = (time.time() - t0) * 1000
        r_arr = np.sort(np.array(r))
        exact = np.sort(np.cos((2.0 * np.arange(1, n + 1) - 1.0) * np.pi / (2.0 * n)))
        err = np.max(np.abs(r_arr - exact)) if len(r_arr) == n else np.inf
        print(f"  T_{n}: {len(r_arr)} roots, max err = {err:.2e}, time = {elapsed:.1f}ms")
        assert len(r_arr) == n and err < 1e-10

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: accuracy vs n
    finite_errs = [(n, e) for n, e in zip(ns, errs) if np.isfinite(e)]
    if finite_errs:
        ns_plot, errs_plot = zip(*finite_errs)
        axes[0].semilogy(ns_plot, errs_plot, 'b.-', markersize=8, linewidth=1.5)
        axes[0].set_xlabel("n")
        axes[0].set_ylabel("max root error")
        axes[0].set_title("Rootfinding accuracy: $\\sin(n\\pi x)$")
        axes[0].grid(True, alpha=0.4)

    # Right: example sin(20*pi*x) with roots
    n_ex = 20
    f_ex = cj.chebfun(lambda x: jnp.sin(n_ex * pi * x))
    xs_ex = np.linspace(-1.0, 1.0, 1000)
    ys_ex = np.array(f_ex(jnp.array(xs_ex)))
    r_ex = np.array(f_ex.roots())
    axes[1].plot(xs_ex, ys_ex, color="#1e77b4", linewidth=1.2)
    axes[1].plot(r_ex, np.zeros_like(r_ex), "ro", markersize=4, label=f"{len(r_ex)} roots")
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set_title(f"$\\sin({n_ex}\\pi x)$ with {len(r_ex)} roots")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.4)

    fig.suptitle("Chebfun rootfinding: speed and accuracy", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "roots_speed.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
