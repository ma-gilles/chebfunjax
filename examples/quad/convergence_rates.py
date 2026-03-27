"""Quadrature convergence for smooth vs non-smooth integrands.

For smooth integrands, Clenshaw-Curtis (Chebfun sum) is spectrally
accurate. For functions with interior discontinuities, convergence
is algebraic unless the domain is split at the singularity.

Credit: Inspired by Chebfun example quad/QuadratureConvergence.m.
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


def run():
    print("=" * 60)
    print("Quadrature convergence with Chebfun")
    print("=" * 60)

    # --- 1. Smooth function: exp(cos(x)) ----------------------------
    f = cj.chebfun(lambda x: jnp.exp(jnp.cos(x)))
    integral = float(f.sum())
    print(f"\nIntegral of exp(cos(x)) over [-1,1]:")
    print(f"  Computed: {integral:.15f}")
    # Reference from scipy
    from scipy.integrate import quad as scipy_quad
    exact, _ = scipy_quad(lambda x: float(np.exp(np.cos(x))), -1.0, 1.0)
    err = abs(integral - exact)
    print(f"  Scipy:    {exact:.15f}")
    print(f"  Error:    {err:.2e}")
    assert err < 1e-12

    # --- 2. Oscillatory: cos(100*pi*x) on [-1, 1] -------------------
    dom_osc = (-1.0, 1.0)
    f_osc = cj.chebfun(lambda x: jnp.cos(100.0 * jnp.pi * x), domain=dom_osc)
    int_osc = float(f_osc.sum())
    # Exact: integral of cos(100*pi*x) from -1 to 1 = 0 (symmetric odd period)
    print(f"\nIntegral of cos(100*pi*x) over [-1,1]:")
    print(f"  Computed: {int_osc:.2e}  (exact: 0)")
    print(f"  Length:   {len(f_osc)}")
    print(f"  Error:    {abs(int_osc):.2e}")
    assert abs(int_osc) < 1e-11

    # --- 3. Piecewise smooth: |x| on [-1, 1] with breakpoint --------
    f_abs_smooth = cj.chebfun(lambda x: jnp.abs(x))
    # With explicit breakpoint at x=0
    f_abs_piece = cj.chebfun(lambda x: jnp.abs(x), domain=[-1.0, 0.0, 1.0])
    int_smooth = float(f_abs_smooth.sum())
    int_piece = float(f_abs_piece.sum())
    exact_abs = 1.0  # integral of |x| over [-1,1]
    print(f"\nIntegral of |x| over [-1,1]:")
    print(f"  Without breakpoint: {int_smooth:.15f}, length={len(f_abs_smooth)}")
    print(f"  With breakpoint:    {int_piece:.15f}, length={len(f_abs_piece)}")
    print(f"  Exact: 1.0")
    # With breakpoint uses fewer terms and is more accurate
    # Without breakpoint: may not fully converge (|x| has a kink at 0)
    assert abs(int_smooth - exact_abs) < 1e-6   # |x| without breakpoint is approximate
    assert abs(int_piece - exact_abs) < 1e-14

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f, title="exp(cos(x)) — smooth integrand")
    fig.savefig(os.path.join(_here, "convergence_rates.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
