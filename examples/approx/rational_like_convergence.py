"""Convergence rates for different function classes.

Demonstrates that Chebfun length depends on the smoothness of the
function: entire functions need fewer terms than analytic functions,
which need fewer than smooth functions with singularities.

Credit: Inspired by Chebfun approx/SpectralDecay.m.
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
    print("Convergence rates for different function classes")
    print("=" * 60)

    # --- Entire functions (exponential convergence) -------------------
    entire_fns = [
        ("exp(x)",   lambda x: jnp.exp(x)),
        ("sin(3x)",  lambda x: jnp.sin(3.0*x)),
        ("cos(5x)",  lambda x: jnp.cos(5.0*x)),
    ]
    print("\nEntire functions (should need few coefficients):")
    for name, f_func in entire_fns:
        f = cj.chebfun(f_func)
        print(f"  {name}: n = {len(f)}")

    # --- Functions with poles off [-1,1]: 1/(1+c*x^2) ----------------
    # Convergence rate depends on distance of pole to [-1,1]
    print("\n1/(1+c*x^2): fewer terms for larger c (poles farther from axis):")
    for c in [1.0, 4.0, 25.0]:
        f = cj.chebfun(lambda x, c=c: 1.0 / (1.0 + c * x**2))
        print(f"  c = {c:5.0f}: n = {len(f)}")

    # Verify 1/(1+x^2) integrates correctly: integral = pi/2
    f_rational = cj.chebfun(lambda x: 1.0 / (1.0 + x**2))
    integral = float(f_rational.sum())
    exact_int = float(jnp.pi / 2.0)
    print(f"\n  Integral of 1/(1+x^2) over [-1,1] = {integral:.15f}")
    print(f"  Exact = pi/2 = {exact_int:.15f}")
    print(f"  Error = {abs(integral - exact_int):.2e}")
    assert abs(integral - exact_int) < 1e-12

    # --- Runge function ----------------------------------------------
    # 1/(1+25*x^2) is the classic Runge function
    f_runge = cj.chebfun(lambda x: 1.0 / (1.0 + 25.0 * x**2))
    print(f"\nRunge function 1/(1+25x^2): n = {len(f_runge)}")
    # Verify it evaluates correctly at x = 0
    assert abs(float(f_runge(jnp.array(0.0))) - 1.0) < 1e-14
    # Verify at x = 1: 1/(1+25) = 1/26
    assert abs(float(f_runge(jnp.array(1.0))) - 1.0/26.0) < 1e-13

    # --- Effect of highly oscillatory functions ----------------------
    print("\nOscillatory functions (more coefficients needed):")
    for freq in [10, 50, 100]:
        f = cj.chebfun(lambda x, k=freq: jnp.cos(k * x))
        print(f"  cos({freq}*x): n = {len(f)}")

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    _f1 = cj.chebfun(lambda x: 1.0 / (1.0 + x**2))
    _f25 = cj.chebfun(lambda x: 1.0 / (1.0 + 25.0 * x**2))
    fig, ax = plot(_f1, title="Convergence: functions near poles", label="1/(1+x²)")
    plot(_f25, ax=ax, color="#E04040", label="1/(1+25x²)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "rational_like_convergence.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plotcoeffs(_f25, title="|Chebyshev coeffs| of 1/(1+25x²)")
    fig2.savefig(os.path.join(_here, "rational_like_convergence_coeffs.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
