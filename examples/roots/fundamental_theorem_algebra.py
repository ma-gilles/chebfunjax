"""Does a Chebfun of degree n have n roots?

Explores the analogue of the fundamental theorem of algebra for Chebyshev
expansions: how many of the n roots of a degree-n Chebfun lie on the real
interval versus in the complex plane?

Credit: Inspired by Chebfun example roots/FundamentalTheoremOfAlgebra.m
(Alex Townsend, October 2013).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj


def run():
    print("=" * 60)
    print("Fundamental theorem of algebra for Chebfuns")
    print("=" * 60)

    # For a degree-n polynomial represented in the Chebyshev basis, the
    # colleague matrix approach gives all n eigenvalues.  The real ones in
    # [-1,1] are the real roots; the rest are complex.

    # Example 1: x^n - 1  (Chebyshev expansion)
    # The real roots of x^n - 1 on [-1,1] are ±1 for even n, just 1 for odd n.
    print("\nExample 1: f(x) = x^n - 1 (monomial, converted to Chebfun)")
    for n in [3, 4, 8]:
        f = cj.chebfun(lambda x, _n=n: x**_n - 1.0)
        r = f.roots()
        r_arr = np.array(r)
        print(f"  n={n}: {len(r_arr)} real root(s) on [-1,1]: {r_arr}")
        # x^n = 1 on [-1,1]: solution is x=1 always; x=-1 if n is even
        expected = [1.0] if n % 2 == 1 else [-1.0, 1.0]
        for x_exp in expected:
            assert any(abs(r_arr - x_exp) < 1e-8), f"Missing root {x_exp} for n={n}"

    # Example 2: Chebyshev polynomial T_n itself has n real roots on [-1,1]
    # T_n(x) = cos(n*arccos(x)), roots at x_k = cos((2k-1)*pi/(2n))
    print("\nExample 2: Chebyshev polynomial T_n (n real roots on [-1,1])")
    for n in [5, 10, 20]:
        # T_n as a Chebfun: coefficients = [0,0,...,0,1] (length n+1)
        coeffs = jnp.zeros(n + 1).at[n].set(1.0)
        f = cj.Chebfun.from_coeffs(coeffs)
        r = f.roots()
        r_arr = np.sort(np.array(r))
        exact = np.cos((2.0 * np.arange(1, n + 1) - 1.0) * np.pi / (2.0 * n))[::-1]
        err = np.max(np.abs(r_arr - exact)) if len(r_arr) == n else np.inf
        print(f"  T_{n}: {len(r_arr)} roots found, max error vs exact = {err:.2e}")
        assert len(r_arr) == n, f"T_{n} should have {n} roots, got {len(r_arr)}"
        assert err < 1e-8, f"T_{n} root error: {err}"

    # Example 3: A random Chebfun and root count
    rng = np.random.default_rng(42)
    n = 10
    coeffs = jnp.array(rng.standard_normal(n + 1) * 0.9**np.arange(n + 1))
    f_rand = cj.Chebfun.from_coeffs(coeffs)
    r_rand = f_rand.roots()
    print(f"\nRandom degree-{n} Chebfun: {len(np.array(r_rand))} real roots on [-1,1]")
    assert len(np.array(r_rand)) <= n

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: T_10 and its roots
    n_plot = 10
    coeffs_plot = jnp.zeros(n_plot + 1).at[n_plot].set(1.0)
    f_plot = cj.Chebfun.from_coeffs(coeffs_plot)
    xs = np.linspace(-1.0, 1.0, 500)
    ys = np.array(f_plot(jnp.array(xs)))
    r_plot = np.sort(np.array(f_plot.roots()))
    axes[0].plot(xs, ys, color="#1e77b4", linewidth=1.8)
    axes[0].axhline(0, color="k", linewidth=0.5)
    axes[0].plot(r_plot, np.zeros(len(r_plot)), "ro", markersize=5, label=f"{n_plot} roots")
    axes[0].set_title(f"$T_{{10}}(x)$: exactly {n_plot} real roots")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.4)

    # Right: x^3 - 1 and root
    f_cubic = cj.chebfun(lambda x: x**3 - 1.0)
    xs2 = np.linspace(-1.0, 1.0, 500)
    ys2 = np.array(f_cubic(jnp.array(xs2)))
    r_cubic = np.array(f_cubic.roots())
    axes[1].plot(xs2, ys2, color="#d62728", linewidth=1.8)
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].plot(r_cubic, np.zeros(len(r_cubic)), "ro", markersize=7, label="root x=1")
    axes[1].set_title("$x^3 - 1$: 1 real root on $[-1,1]$")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.4)

    fig.suptitle("Fundamental theorem of algebra for Chebfuns", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "fundamental_theorem_algebra.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
