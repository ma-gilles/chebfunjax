"""Rank of bivariate functions.

Demonstrates the notion of numerical rank for bivariate functions:
separable functions are rank-1, sums of separable functions have
low rank, and general smooth functions may have higher rank.

Credit: Inspired by Chebfun2 approx2/Rank.m examples.
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
from chebfunjax.plotting import surf, contour


def run():
    print("=" * 60)
    print("Rank of bivariate functions")
    print("=" * 60)

    # --- Rank-1: f(x,y) = g(x)*h(y) ---------------------------------
    f1 = cj.chebfun2(lambda x, y: jnp.cos(x) * jnp.exp(y))
    print(f"\ncos(x)*exp(y): rank = {f1.rank}  (exact: 1)")
    assert f1.rank == 1

    f2 = cj.chebfun2(lambda x, y: jnp.sin(x) * jnp.sin(y))
    print(f"sin(x)*sin(y): rank = {f2.rank}  (exact: 1)")
    assert f2.rank == 1

    # --- Rank-2: x^2 + y^2 = x^2*1 + 1*y^2 -------------------------
    f3 = cj.chebfun2(lambda x, y: x**2 + y**2)
    print(f"\nx^2 + y^2: rank = {f3.rank}  (exact: 2)")
    assert f3.rank == 2

    # --- Higher rank: general smooth functions -----------------------
    f4 = cj.chebfun2(lambda x, y: jnp.exp(jnp.cos(x + y)))
    print(f"\nexp(cos(x+y)): rank = {f4.rank}  (low but >1)")
    # This is not separable, so rank > 1
    assert f4.rank > 1

    # --- Verify accuracy of low-rank approximations ----------------
    # Test that rank-1 approximation of cos(x)*exp(y) is correct
    x_test, y_test = jnp.array(0.7), jnp.array(-0.3)
    val1 = float(f1(x_test, y_test))
    exact1 = float(jnp.cos(x_test) * jnp.exp(y_test))
    print(f"\ncos(x)*exp(y) at (0.7,-0.3): error = {abs(val1 - exact1):.2e}")
    assert abs(val1 - exact1) < 1e-12

    val3 = float(f3(x_test, y_test))
    exact3 = float(x_test**2 + y_test**2)
    print(f"x^2+y^2 at (0.7,-0.3): error = {abs(val3 - exact3):.2e}")
    assert abs(val3 - exact3) < 1e-12

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = surf(f1, title="cos(x)·exp(y) on [-1,1]²")
    fig.savefig(os.path.join(_here, "rank_of_functions.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig2, ax2 = contour(f2, title="sin(x)·sin(y) on [-1,1]²")
    fig2.savefig(os.path.join(_here, "rank_of_functions_contour.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
