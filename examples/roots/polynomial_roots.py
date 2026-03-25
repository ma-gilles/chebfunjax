"""Roots of polynomials and trigonometric functions.

Demonstrates finding roots of Chebyshev polynomials (known analytically),
trigonometric functions, and equations like cos(x) = x^2.

Credit: Inspired by Chebfun examples roots/ChebPts.m.
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
from chebfunjax.plotting import plot


def run():
    print("=" * 60)
    print("Roots of polynomials and trigonometric functions")
    print("=" * 60)

    # --- T_n(x) has roots at cos((2k-1)*pi/(2n)) for k=1,...,n ------
    # T_2(x) = 2x^2 - 1, roots at x = ±1/sqrt(2) = ±cos(pi/4)
    f_T2 = cj.chebfun(lambda x: 2.0*x**2 - 1.0)
    roots_T2 = np.sort(np.array(f_T2.roots()))
    exact_T2 = np.sort([-1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)])
    print(f"\nT_2(x) = 2x^2 - 1:")
    print(f"  Roots: {roots_T2}")
    print(f"  Exact: {exact_T2}")
    assert len(roots_T2) == 2
    assert abs(roots_T2[0] - exact_T2[0]) < 1e-12
    assert abs(roots_T2[1] - exact_T2[1]) < 1e-12

    # T_4(x) = 8x^4 - 8x^2 + 1, roots at cos((2k-1)*pi/8) for k=1,2,3,4
    f_T4 = cj.chebfun(lambda x: 8.0*x**4 - 8.0*x**2 + 1.0)
    roots_T4 = np.sort(np.array(f_T4.roots()))
    exact_T4 = np.sort([np.cos((2*k - 1)*np.pi/8) for k in range(1, 5)])
    print(f"\nT_4(x): {len(roots_T4)} roots, max error = "
          f"{np.max(np.abs(roots_T4 - exact_T4)):.2e}")
    assert len(roots_T4) == 4
    assert np.max(np.abs(roots_T4 - exact_T4)) < 1e-12

    # --- sin(x) on [-pi, pi] has roots at 0, ±pi --------------------
    pi = float(jnp.pi)
    dom_sin = (-pi, pi)
    f_sin = cj.chebfun(lambda x: jnp.sin(x), domain=dom_sin)
    roots_sin = np.sort(np.array(f_sin.roots()))
    print(f"\nsin(x) on [-pi, pi]:")
    print(f"  Roots: {roots_sin}")
    # Interior roots: 0; boundary roots: -pi, pi (may or may not be included)
    # Filter to interior
    interior_sin = roots_sin[np.abs(roots_sin) < pi - 1e-6]
    print(f"  Interior roots (should be 0): {interior_sin}")
    assert any(abs(r) < 1e-10 for r in roots_sin), "Root at 0 not found"

    # --- cos(x) = x^2 — equation solving via roots(f - g) -----------
    # On [-1, 1]: cos(x) > x^2 for some region
    f_mixed = cj.chebfun(lambda x: jnp.cos(x) - x**2)
    roots_mixed = np.sort(np.array(f_mixed.roots()))
    print(f"\ncos(x) = x^2 on [-1, 1]:")
    print(f"  Roots: {roots_mixed}")
    # There are 2 roots (symmetric since both functions are even)
    assert len(roots_mixed) == 2
    # cos(x) = x^2 at x ~ +-0.824
    for ri in roots_mixed:
        lhs = float(jnp.cos(jnp.array(ri)))
        rhs = ri**2
        assert abs(lhs - rhs) < 1e-13, f"|cos(r) - r^2| = {abs(lhs-rhs)}"
    # Symmetry check
    assert abs(roots_mixed[0] + roots_mixed[1]) < 1e-10

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f_T2, title="Chebyshev polynomial roots", label="T₂(x)")
    plot(f_T4, ax=ax, color="#E04040", label="T₄(x)")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "polynomial_roots.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
