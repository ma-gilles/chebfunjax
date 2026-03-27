"""Nonstandard boundary conditions for ODEs.

Solves  u'' + x^2 u = 1  on [-1,1] with:
  (1) u(-1)=1, integral of u over [-1,1] = 0  (mean-zero condition)
  (2) u(-1)=0, u'(0)=1  (interior derivative condition)

Credit: Chebfun example ode-linear/NonstandardBCs.m (Asgeir Birkisson, Oct 2011).
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

from chebfunjax.operators.chebop import Chebop
from chebfunjax.operators.linop import Linop

def run():
    print("=" * 60)
    print("Nonstandard boundary conditions")
    print("=" * 60)

    dom = (-1.0, 1.0)

    # Example 1: mean-zero condition
    # u'' + x^2 u = 1, u(-1)=1, integral(u)=0
    # Model with two BCs: u(-1)=1 and u(1)=c where c chosen so integral=0
    print("\nExample 1: u'' + x^2 u = 1, u(-1)=1, mean(u)=0")
    from scipy.optimize import brentq

    def solve_with_rbc(rbc_val):
        N = Chebop(lambda x, u: u.diff(2) + x**2 * u, domain=dom)
        N.lbc = 1.0
        N.rbc = rbc_val
        return N.solve(cj.chebfun(lambda x: jnp.ones_like(x), domain=dom))

    def mean_residual(rbc_val):
        u = solve_with_rbc(rbc_val)
        return float(u.sum())   # integral over [-1,1]

    rbc_opt = brentq(mean_residual, -5.0, 5.0)
    u1 = solve_with_rbc(rbc_opt)
    mean_u = float(u1.sum())
    print(f"  Optimised rbc = {rbc_opt:.6f}")
    print(f"  mean(u) = {mean_u:.2e}  (should be 0)")
    assert abs(mean_u) < 1e-8
    assert abs(float(u1(jnp.array(-1.0))) - 1.0) < 1e-8

    # Example 2: interior derivative condition
    # u'' + x^2 u = 1, u(-1)=0, u'(0)=1
    print("\nExample 2: u'' + x^2 u = 1, u(-1)=0, u'(0)=1")

    def solve_with_rbc2(rbc_val):
        N = Chebop(lambda x, u: u.diff(2) + x**2 * u, domain=dom)
        N.lbc = 0.0
        N.rbc = rbc_val
        return N.solve(cj.chebfun(lambda x: jnp.ones_like(x), domain=dom))

    def deriv_residual(rbc_val):
        u = solve_with_rbc2(rbc_val)
        return float(u.diff()(jnp.array(0.0))) - 1.0

    rbc_opt2 = brentq(deriv_residual, -5.0, 5.0)
    u2 = solve_with_rbc2(rbc_opt2)
    deriv_at_0 = float(u2.diff()(jnp.array(0.0)))
    print(f"  Optimised rbc = {rbc_opt2:.6f}")
    print(f"  u'(0) = {deriv_at_0:.8f}  (should be 1)")
    assert abs(deriv_at_0 - 1.0) < 1e-7
    assert abs(float(u2(jnp.array(-1.0)))) < 1e-8

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(-1.0, 1.0, 400)
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(x_plot, u1(x_plot), 'b', linewidth=1.8)
    axes[0].axhline(0, color='k', linewidth=0.5)
    axes[0].set_title("u″+x²u=1, u(−1)=1, mean(u)=0", fontsize=10)

    axes[1].plot(x_plot, u2(x_plot), 'r', linewidth=1.8)
    axes[1].axvline(0, color='k', linestyle='--', linewidth=0.8)
    axes[1].set_title("u″+x²u=1, u(−1)=0, u′(0)=1", fontsize=10)

    fig.suptitle("Nonstandard boundary conditions", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "nonstandard_bcs.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
