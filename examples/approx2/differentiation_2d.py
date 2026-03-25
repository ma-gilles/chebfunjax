"""Partial derivatives of Chebfun2 functions.

Demonstrates computing partial derivatives of bivariate functions
using Chebfun2's diff method.

Credit: Inspired by Chebfun2 approx2/Differentiation.m.
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
from chebfunjax.plotting import surf, contour


def run():
    print("=" * 60)
    print("Partial derivatives of Chebfun2 functions")
    print("=" * 60)

    x_t, y_t = jnp.array(0.3), jnp.array(0.7)

    # --- d/dx exp(x+y) = exp(x+y) ------------------------------------
    # Note: in chebfunjax, dim=2 differentiates w.r.t. x, dim=1 w.r.t. y
    f1 = cj.chebfun2(lambda x, y: jnp.exp(x + y))
    df1_dx = f1.diff(dim=2)
    val = float(df1_dx(x_t, y_t))
    exact = float(jnp.exp(x_t + y_t))
    print(f"\nd/dx exp(x+y) at (0.3, 0.7):")
    print(f"  Computed: {val:.15f}")
    print(f"  Exact:    {exact:.15f}")
    print(f"  Error:    {abs(val - exact):.2e}")
    assert abs(val - exact) < 1e-12

    # --- d/dy exp(x+y) = exp(x+y) ------------------------------------
    df1_dy = f1.diff(dim=1)
    val_dy = float(df1_dy(x_t, y_t))
    print(f"\nd/dy exp(x+y) at (0.3, 0.7):")
    print(f"  Computed: {val_dy:.15f}")
    print(f"  Error:    {abs(val_dy - exact):.2e}")
    assert abs(val_dy - exact) < 1e-12

    # --- d/dx (x^2*y + sin(x)) = 2x*y + cos(x) ----------------------
    f2 = cj.chebfun2(lambda x, y: x**2 * y + jnp.sin(x))
    df2_dx = f2.diff(dim=2)  # dim=2 => d/dx
    val2 = float(df2_dx(x_t, y_t))
    exact2 = float(2.0 * x_t * y_t + jnp.cos(x_t))
    print(f"\nd/dx (x^2*y + sin(x)) at (0.3, 0.7):")
    print(f"  Computed: {val2:.15f}")
    print(f"  Exact:    {exact2:.15f}")
    print(f"  Error:    {abs(val2 - exact2):.2e}")
    assert abs(val2 - exact2) < 1e-12

    # --- d/dy (x^2*y + sin(x)) = x^2 --------------------------------
    df2_dy = f2.diff(dim=1)  # dim=1 => d/dy
    val2_dy = float(df2_dy(x_t, y_t))
    exact2_dy = float(x_t**2)
    print(f"\nd/dy (x^2*y + sin(x)) at (0.3, 0.7):")
    print(f"  Computed: {val2_dy:.15f}")
    print(f"  Exact (x^2 = {exact2_dy:.6f}):    {exact2_dy:.15f}")
    print(f"  Error:    {abs(val2_dy - exact2_dy):.2e}")
    assert abs(val2_dy - exact2_dy) < 1e-12

    # --- Mixed partial: verify d^2/dxdy (x*y) = 1 -------------------
    f3 = cj.chebfun2(lambda x, y: x * y)
    df3_dx = f3.diff(dim=2)  # d/dx (x*y) = y
    # d/dy (y) = 1
    df3_dxdy = df3_dx.diff(dim=1)  # d/dy
    val3 = float(df3_dxdy(x_t, y_t))
    print(f"\nd^2/dxdy (x*y) at (0.3, 0.7):")
    print(f"  Computed: {val3:.15f}  (exact: 1.0)")
    assert abs(val3 - 1.0) < 1e-12

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = surf(f1, title="exp(x+y): 2-D differentiation")
    fig.savefig(os.path.join(_here, "differentiation_2d.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig2, ax2 = contour(f2, title="x²y + sin(x)")
    fig2.savefig(os.path.join(_here, "differentiation_2d_f2.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
