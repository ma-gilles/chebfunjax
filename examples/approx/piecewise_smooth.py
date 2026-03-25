"""Piecewise smooth functions in Chebfun.

Demonstrates constructing Chebfun approximations of piecewise smooth
functions by specifying breakpoints in the domain.

Credit: Inspired by Chebfun approx/PiecewiseSmooth.m.
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
    print("Piecewise smooth functions")
    print("=" * 60)

    # --- |x| with breakpoint at 0 ------------------------------------
    # MATLAB: f = chebfun('abs(x)', [-1, 0, 1]);
    dom_split = (-1.0, 0.0, 1.0)  # breakpoint at 0
    f_abs = cj.chebfun(lambda x: jnp.abs(x), domain=dom_split)
    print(f"\n|x| on [-1,1] with breakpoint at 0:")
    print(f"  Number of coefficients: {len(f_abs)}")
    integral_abs = float(f_abs.sum())
    print(f"  Integral = {integral_abs:.15f}  (exact: 1.0)")
    assert abs(integral_abs - 1.0) < 1e-14

    # Evaluate at a few points
    for x_val in [-0.5, 0.0, 0.5]:
        val = float(f_abs(jnp.array(x_val)))
        print(f"  f({x_val}) = {val:.15f}  (exact: {abs(x_val):.15f})")
        assert abs(val - abs(x_val)) < 1e-14

    # --- Step function -----------------------------------------------
    # MATLAB: f = chebfun(@(x) sign(x), [-1, 0, 1]);
    f_step = cj.chebfun(
        lambda x: jnp.where(x < 0, -jnp.ones_like(x), jnp.ones_like(x)),
        domain=(-1.0, 0.0, 1.0)
    )
    print(f"\nStep function sign(x) on [-1,1]:")
    print(f"  Number of coefficients: {len(f_step)}")
    # Integral of sign(x) over [-1,1] = 0
    integral_step = float(f_step.sum())
    print(f"  Integral = {integral_step:.2e}  (exact: 0)")
    assert abs(integral_step) < 1e-9

    # --- Hat function ------------------------------------------------
    # f(x) = 1 - |x| on [-1, 1]
    dom_hat = (-1.0, 0.0, 1.0)
    f_hat = cj.chebfun(lambda x: 1.0 - jnp.abs(x), domain=dom_hat)
    print(f"\nHat function 1 - |x| on [-1,1]:")
    print(f"  Number of coefficients: {len(f_hat)}")
    # Integral = 1.0 (area of triangle with base 2, height 1)
    integral_hat = float(f_hat.sum())
    print(f"  Integral = {integral_hat:.15f}  (exact: 1.0)")
    assert abs(integral_hat - 1.0) < 1e-14
    print(f"  f(0) = {float(f_hat(jnp.array(0.0))):.15f}  (exact: 1.0)")
    assert abs(float(f_hat(jnp.array(0.0))) - 1.0) < 1e-14

    # --- Piecewise defined: different functions on each piece ---------
    # f = x^2 for x < 0, sin(x) for x >= 0
    f_piecewise = cj.chebfun(
        lambda x: jnp.where(x < 0, x**2, jnp.sin(x)),
        domain=(-1.0, 0.0, 1.0)
    )
    print(f"\nPiecewise: x^2 for x<0, sin(x) for x>=0:")
    print(f"  Number of coefficients: {len(f_piecewise)}")
    # Check values
    assert abs(float(f_piecewise(jnp.array(-0.5))) - 0.25) < 1e-12
    assert abs(float(f_piecewise(jnp.array(0.5))) - float(jnp.sin(jnp.array(0.5)))) < 1e-12

    # Integral = integral(-1,0) x^2 dx + integral(0,1) sin(x) dx
    #          = [x^3/3](-1,0) + [-cos(x)](0,1)
    #          = 0 - (-1/3) + (-cos(1) + 1)
    #          = 1/3 + 1 - cos(1)
    exact_piecewise = 1.0/3.0 + 1.0 - float(jnp.cos(jnp.array(1.0)))
    integral_piecewise = float(f_piecewise.sum())
    print(f"  Integral = {integral_piecewise:.15f}  (exact: {exact_piecewise:.15f})")
    assert abs(integral_piecewise - exact_piecewise) < 1e-13

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f_abs, title="Piecewise smooth functions", label="|x|")
    plot(f_hat, ax=ax, color="#E04040", label="1-|x|")
    plot(f_piecewise, ax=ax, color="#228B22", label="x² / sin(x)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "piecewise_smooth.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
