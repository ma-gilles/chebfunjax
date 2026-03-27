"""Gauss-Legendre quadrature (Golub-Welsch algorithm).

Demonstrates computing Gauss-Legendre nodes and weights via the
Golub-Welsch algorithm (eigenvalues of a tridiagonal matrix),
and verifies that the rule integrates polynomials exactly.

Credit: Inspired by Chebfun examples quad/Gauss.m.
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.plotting import plot


def golub_welsch(n):
    """Compute n-point Gauss-Legendre nodes and weights via Golub-Welsch."""
    k = np.arange(1, n)
    beta = 0.5 / np.sqrt(1.0 - (2.0 * k)**(-2))
    T = np.diag(beta, 1) + np.diag(beta, -1)
    eigvals, eigvecs = np.linalg.eigh(T)
    idx = np.argsort(eigvals)
    x = eigvals[idx]
    w = 2.0 * eigvecs[0, idx]**2
    return x, w


def run():
    print("=" * 60)
    print("Gauss-Legendre quadrature (Golub-Welsch)")
    print("=" * 60)

    # --- Compute n-point rules for n = 2, 3, 5, 10 ------------------
    print("\nGauss-Legendre nodes and weights:")
    for n in [2, 3, 5]:
        x, w = golub_welsch(n)
        print(f"\n  n={n}: nodes = {x}")
        print(f"         weights = {w}")
        print(f"         sum(w) = {sum(w):.15f}  (exact: 2.0)")
        assert abs(sum(w) - 2.0) < 1e-14

    # --- Exactness: n-point rule integrates degree-(2n-1) exactly ----
    print("\nExactness: n-point GL integrates degree <= 2n-1 polynomials exactly")
    for n in [3, 5, 10]:
        x, w = golub_welsch(n)
        max_degree = 2 * n - 1
        # Test on x^k for k = 0, 1, ..., 2n-1
        errors = []
        for k in range(max_degree + 1):
            computed = float(np.dot(w, x**k))
            # Exact: integral of x^k from -1 to 1
            exact = 2.0 / (k + 1) if k % 2 == 0 else 0.0
            errors.append(abs(computed - exact))
        max_err = max(errors)
        print(f"  n={n}: max error for deg 0..{max_degree}: {max_err:.2e}")
        assert max_err < 1e-12, f"n={n}: max error too large: {max_err}"

    # --- Compare with numpy.polynomial.legendre.leggauss ------------
    print("\nComparison with numpy.polynomial.legendre.leggauss:")
    for n in [5, 10, 20]:
        x_gw, w_gw = golub_welsch(n)
        x_np, w_np = np.polynomial.legendre.leggauss(n)
        err_x = np.max(np.abs(np.sort(x_gw) - np.sort(x_np)))
        err_w = np.max(np.abs(np.sort(w_gw) - np.sort(w_np)))
        print(f"  n={n}: node err={err_x:.2e}, weight err={err_w:.2e}")
        assert err_x < 1e-12
        assert err_w < 1e-12

    # --- Apply 5-pt rule to exp(x) ----------------------------------
    n = 5
    x5, w5 = golub_welsch(n)
    computed = float(np.dot(w5, np.exp(x5)))
    exact = float(jnp.exp(jnp.array(1.0)) - jnp.exp(jnp.array(-1.0)))
    print(f"\n5-pt GL quadrature of exp(x):")
    print(f"  Computed: {computed:.15f}")
    print(f"  Exact:    {exact:.15f}")
    print(f"  Error:    {abs(computed - exact):.2e}")
    assert abs(computed - exact) < 1e-7  # 5-pt GL exact for poly up to degree 9

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as _plt
    import numpy as _np
    fig, ax = _plt.subplots(figsize=(6, 3.5))
    for _n, _col in [(5, "#4169E1"), (10, "#E04040"), (20, "#228B22")]:
        _nodes, _weights = golub_welsch(_n)
        ax.plot(_nodes, _np.zeros_like(_nodes), "o", color=_col,
                markersize=5, label=f"n={_n}")
    ax.set_title("Gauss-Legendre nodes on [-1, 1]", fontsize=11)
    ax.set_xlabel("x", fontsize=10)
    ax.set_yticks([])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "gauss_quadrature.png"),
                dpi=150, bbox_inches="tight")
    _plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
