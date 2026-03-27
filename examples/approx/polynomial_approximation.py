"""Polynomial approximation basics with chebfunjax.

Demonstrates constructing Chebfun approximations of smooth functions,
checking lengths (polynomial degrees), and measuring errors.

Credit: Inspired by Chebfun examples approx/Entire.m and related examples.
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

from chebfunjax.plotting import plot, plotcoeffs

def run():
    print("=" * 60)
    print("Polynomial approximation basics")
    print("=" * 60)

    # --- Construct a Chebfun for exp(x) on [-1, 1] --------------------
    # MATLAB: f = chebfun('exp(x)')
    f = cj.chebfun(lambda x: jnp.exp(x))
    print(f"\nexp(x) on [-1,1]:")
    print(f"  Number of coefficients (length): {len(f)}")
    # Should be around 18 (degree ~17 suffices for machine precision)

    # Evaluate at a point: f(0) should equal 1
    val_at_0 = float(f(jnp.array(0.0)))
    print(f"  f(0) = {val_at_0:.15f}  (exact: 1.0)")

    # Evaluate at x = 0.5; exact = exp(0.5)
    val_at_half = float(f(jnp.array(0.5)))
    exact = float(jnp.exp(jnp.array(0.5)))
    print(f"  f(0.5) = {val_at_half:.15f}  (exact: {exact:.15f})")
    print(f"  Error at 0.5: {abs(val_at_half - exact):.2e}")

    # --- Chebfun for cos(pi*x) on [-1, 1] ----------------------------
    # MATLAB: g = chebfun('cos(pi*x)')
    g = cj.chebfun(lambda x: jnp.cos(jnp.pi * x))
    print(f"\ncos(pi*x) on [-1,1]:")
    print(f"  Number of coefficients: {len(g)}")
    # Exact value at x=0 is 1
    print(f"  g(0) = {float(g(jnp.array(0.0))):.15f}  (exact: 1.0)")
    # Integral of cos(pi*x) over [-1,1] = 0
    integral_g = float(g.sum())
    print(f"  Integral over [-1,1] = {integral_g:.2e}  (exact: 0)")

    # --- Composition of Chebfuns -------------------------------------
    # MATLAB: h = sin(cos(x)), where x = chebfun('x')
    x = cj.chebfun(lambda t: t)   # identity chebfun
    h = cj.chebfun(lambda t: jnp.sin(jnp.cos(t)))
    print(f"\nsin(cos(x)) on [-1,1]:")
    print(f"  Number of coefficients: {len(h)}")
    # Integral of sin(cos(x)) over [-1,1]
    integral_h = float(h.sum())
    print(f"  Integral = {integral_h:.10f}")
    # Reference value from scipy.integrate.quad:
    # scipy.integrate.quad(lambda x: np.sin(np.cos(x)), -1, 1) = 1.47728599607378
    ref = 1.47728599607378
    print(f"  Reference = {ref:.10f}")
    print(f"  Error: {abs(integral_h - ref):.2e}")

    # --- Non-default domain -------------------------------------------
    # MATLAB: f = chebfun('sin(x)', [0, pi])
    f_pi = cj.chebfun(lambda x: jnp.sin(x), domain=(0.0, float(jnp.pi)))
    print(f"\nsin(x) on [0, pi]:")
    print(f"  Number of coefficients: {len(f_pi)}")
    integral_sin = float(f_pi.sum())
    print(f"  Integral of sin(x) from 0 to pi = {integral_sin:.15f}  (exact: 2.0)")

    # --- Norm of the error in a Chebfun interpolation ----------------
    # Compare chebfun representation against exact function at many points
    xx = jnp.linspace(-1.0, 1.0, 1000)
    f_vals = f(xx)
    exact_vals = jnp.exp(xx)
    max_err = float(jnp.max(jnp.abs(f_vals - exact_vals)))
    print(f"\nMax error of exp(x) chebfun at 1000 equispaced pts: {max_err:.2e}")
    assert max_err < 1e-13, f"Max error too large: {max_err}"
    assert abs(val_at_0 - 1.0) < 1e-14
    assert abs(integral_sin - 2.0) < 1e-12
    assert abs(integral_h - ref) < 1e-8, f"sin(cos(x)) integral error: {abs(integral_h - ref)}"
    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f, title="exp(x) on [-1, 1]", label="exp(x)")
    plot(g, ax=ax, color="#E04040", label="cos(πx)")
    plot(h, ax=ax, color="#228B22", label="sin(cos(x))")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "polynomial_approximation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plotcoeffs(f, title="Chebyshev coefficients of exp(x)")
    fig2.savefig(os.path.join(_here, "polynomial_approximation_coeffs.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
