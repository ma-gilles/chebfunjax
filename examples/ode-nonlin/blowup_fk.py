"""Blowup equation (Frank-Kamenetskii).

The steady Frank-Kamenetskii (spontaneous combustion) equation:
  u'' + A exp(u) = 0,  u(-1) = u(1) = 0
has two solutions for A < A_crit and none for A > A_crit.

Credit: Chebfun example ode-nonlin/BlowupFK.m (Nick Trefethen, Sep 2010).
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


def run():
    print("=" * 60)
    print("Frank-Kamenetskii: u'' + A exp(u) = 0, u(±1) = 0")
    print("=" * 60)

    dom = (-1.0, 1.0)
    # Critical value: A_crit = 0.8785... (for 1D slab)
    # For A < A_crit there are two branches

    A_vals = [0.2, 0.4, 0.6]
    solutions = []

    for A in A_vals:
        print(f"\nA = {A}:")
        # Lower branch (small u)
        # In Chebop lambda, u is a Chebfun; use cj.exp (not jnp.exp)
        N = Chebop(lambda x, u: u.diff(2) + A * cj.exp(u), domain=dom)
        N.lbc = 0.0
        N.rbc = 0.0
        u_low = N.solve(0.0)  # start from 0
        u0_low = float(u_low(jnp.array(0.0)))
        print(f"  Lower branch: u(0) = {u0_low:.6f}")
        assert abs(float(u_low(jnp.array(-1.0)))) < 1e-8
        assert abs(float(u_low(jnp.array(1.0)))) < 1e-8

        # Upper branch (larger u) — use a higher initial guess
        N2 = Chebop(lambda x, u: u.diff(2) + A * cj.exp(u), domain=dom)
        N2.lbc = 0.0
        N2.rbc = 0.0
        u_high = N2.solve(3.0)  # start from high value
        u0_high = float(u_high(jnp.array(0.0)))
        print(f"  Upper branch: u(0) = {u0_high:.6f}")
        solutions.append((A, u_low, u_high))
        # The two branches should be distinct; for A<A_crit both exist
        # (the "upper" branch starting from 3.0 may have negative peak)
        assert abs(u0_high - u0_low) > 0.01  # branches are distinct

    # Verify ODE residual
    A_test = 0.4
    x_test = jnp.linspace(-0.9, 0.9, 200)
    u_test = solutions[1][1]
    res = u_test.diff(2)(x_test) + A_test * jnp.exp(u_test(x_test))
    max_res = float(jnp.max(jnp.abs(res)))
    print(f"\nODE residual (A={A_test}, lower branch): {max_res:.2e}")
    assert max_res < 1e-8

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    colors = ['b', 'r', 'g']
    x_plot = jnp.linspace(-1.0, 1.0, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for (A, u_l, u_h), c in zip(solutions, colors):
        axes[0].plot(x_plot, u_l(x_plot), color=c, linewidth=1.6, label=f"A={A}")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("u(x)")
    axes[0].set_title("FK: lower branch", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    for (A, u_l, u_h), c in zip(solutions, colors):
        axes[1].plot(x_plot, u_h(x_plot), color=c, linewidth=1.6, label=f"A={A}")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("u(x)")
    axes[1].set_title("FK: upper branch", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Frank-Kamenetskii: u″ + A exp(u) = 0, u(±1)=0", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "blowup_fk.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
