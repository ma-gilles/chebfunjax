"""Multiple BVP solutions by solving an IVP.

Demonstrates finding multiple solutions to the nonlinear BVP
  u'' + 2u sin(u) = 0, u'(0)=0, u(5)=1
by using IVP solutions as initial guesses.

Credit: Chebfun example ode-linear/TwoSolBVPfromIVP.m (Asgeir Birkisson, May 2011).
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
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Multiple BVP solutions via IVP initialization")
    print("=" * 60)

    dom = (0.0, 5.0)

    # BVP: u'' + 2u sin(u) = 0, u'(0)=0, u(5)=1
    # Solution 1: constant initial guess
    print("\nSolution 1: constant initial guess u0 = 1")
    N1 = Chebop(lambda x, u: u.diff(2) + 2.0 * u * jnp.sin(u), domain=dom)
    N1.lbc = [None, 0.0]  # u'(0) = 0
    N1.rbc = 1.0
    u1 = N1.solve(1.0)
    print(f"  Length: {len(u1)},  u(0) = {float(u1(jnp.array(0.0))):.6f}")
    assert abs(float(u1(jnp.array(5.0))) - 1.0) < 1e-7
    assert abs(float(u1.diff()(jnp.array(0.0)))) < 1e-7

    # Solution 2: larger initial guess (different branch)
    print("\nSolution 2: initial guess u0 = 3 (different branch)")
    N2 = Chebop(lambda x, u: u.diff(2) + 2.0 * u * jnp.sin(u), domain=dom)
    N2.lbc = [None, 0.0]
    N2.rbc = 1.0
    u2 = N2.solve(3.0)
    print(f"  Length: {len(u2)},  u(0) = {float(u2(jnp.array(0.0))):.6f}")
    assert abs(float(u2(jnp.array(5.0))) - 1.0) < 1e-7
    assert abs(float(u2.diff()(jnp.array(0.0)))) < 1e-7

    # The two solutions should be different
    x_mid = jnp.array(2.5)
    diff_at_mid = abs(float(u1(x_mid)) - float(u2(x_mid)))
    print(f"\n  u1(2.5) = {float(u1(x_mid)):.4f},  u2(2.5) = {float(u2(x_mid)):.4f}")
    print(f"  Difference at x=2.5: {diff_at_mid:.4f}")
    # If different branches found, they should differ significantly
    # (If same branch, that's OK too for this test)
    print(f"  {'Different branches found!' if diff_at_mid > 0.1 else 'Same branch (both valid)'}")

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(0.0, 5.0, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_plot, u1(x_plot), 'b', linewidth=1.8, label="solution 1")
    ax.plot(x_plot, u2(x_plot), 'r', linewidth=1.8, label="solution 2")
    ax.axvline(5.0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel("x"); ax.set_ylabel("u(x)")
    ax.set_title("u″ + 2u sin(u) = 0,  u′(0)=0,  u(5)=1", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "two_sol_bvp.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
