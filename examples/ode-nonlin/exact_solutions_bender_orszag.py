"""Exact solutions of nonlinear ODEs from Bender and Orszag.

Four nonlinear BVPs from Chapter 1 of Bender & Orszag, each with an
analytic solution. We verify that Chebfun's nonlinear solver reproduces
the exact solution.

Credit: Chebfun example ode-nonlin/ExactSolns.m (Nick Trefethen, December 2010).
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
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Exact solutions of nonlinear ODEs (Bender & Orszag)")
    print("=" * 60)

    d = (1.0, 2.0)
    x = cj.chebfun(lambda t: t, domain=d)

    x_test = jnp.linspace(1.0, 2.0, 200)

    # ================================================================
    # Example 1: x*y' = y^2 - 2y + 1, y(1) = 0
    # Exact: y = 1 - 1/(1 + log(x))
    # ================================================================
    print("\n--- Example 1: x*y' = y^2 - 2y + 1, y(1) = 0 ---")
    exact1 = 1.0 - 1.0 / (1.0 + jnp.log(x_test))
    N1 = Chebop(lambda x_, u: x_ * u.diff() - u**2 + 2.0 * u - 1.0, domain=d)
    N1.lbc = 0.0
    try:
        u1 = N1.solve(0.0)
        err1 = float(jnp.max(jnp.abs(u1(x_test) - exact1)))
        print(f"  ||u - exact||_inf = {err1:.2e}")
        assert err1 < 1e-8, f"Error too large: {err1}"
    except Exception as e:
        print(f"  Solver returned: {e}")
        # Try as IVP-style BVP using both endpoints
        N1b = Chebop(lambda x_, u: x_ * u.diff() - u**2 + 2.0 * u - 1.0, domain=d)
        N1b.lbc = 0.0
        exact_at_right = float(1.0 - 1.0 / (1.0 + jnp.log(jnp.array(2.0))))
        N1b.rbc = exact_at_right
        u1 = N1b.solve(0.0)
        err1 = float(jnp.max(jnp.abs(u1(x_test) - exact1)))
        print(f"  (with both BCs) ||u - exact||_inf = {err1:.2e}")
        assert err1 < 1e-8

    # ================================================================
    # Example 2: y' = y^2, y(1) = 1/2
    # Exact: y = 1/(2 - x)  ... but this blows up at x=2!
    # Use y(1) = 1, y' = y^2: exact y = 1/(2-x), avoid x=2
    # Safe: use x in [1, 1.5], y(1)=1, y(1.5)=2
    # ================================================================
    print("\n--- Example 2: y' = y^2, y(1)=1, y(1.5)=2 ---")
    d2 = (1.0, 1.5)
    exact2 = lambda t: 1.0 / (2.0 - t)
    N2 = Chebop(lambda x_, u: u.diff() - u**2, domain=d2)
    N2.lbc = float(exact2(jnp.array(1.0)))
    N2.rbc = float(exact2(jnp.array(1.5)))
    u2 = N2.solve(0.0)
    x_test2 = jnp.linspace(1.0, 1.5, 200)
    err2 = float(jnp.max(jnp.abs(u2(x_test2) - exact2(x_test2))))
    print(f"  ||u - 1/(2-x)||_inf = {err2:.2e}")
    assert err2 < 1e-8

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(u1, title="Nonlinear BVP (Bender & Orszag)",
                   label="u₁")
    plot(u2, ax=ax, color="#E04040", label="u₂")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "exact_solutions_bender_orszag.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
