"""Matched asymptotics and boundary layers.

Solves  -eps y'' + (2 - x^2) y = 1,  y(±1) = 0  for small eps,
and compares with the leading-order outer solution y_outer = 1/(2-x^2).

Credit: Chebfun example ode-linear/MatchedAsymp.m (Nick Trefethen, Dec 2010).
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
    print("Matched asymptotics: -eps y'' + (2-x^2)y = 1, y(±1)=0")
    print("=" * 60)

    dom = (-1.0, 1.0)
    eps_vals = [0.1, 0.01, 0.001]

    # Outer solution (eps=0 limit): y_outer = 1/(2-x^2)
    # But y_outer(±1) = 1/(2-1) = 1 ≠ 0, so there are boundary layers at x=±1
    y_outer = lambda x: 1.0 / (2.0 - x**2)

    solutions = []
    for eps in eps_vals:
        N = Chebop(
            lambda x, u: -eps * u.diff(2) + (2.0 - x**2) * u,
            domain=dom
        )
        N.lbc = 0.0
        N.rbc = 0.0
        u = N.solve(cj.chebfun(lambda x: jnp.ones_like(x), domain=dom))
        solutions.append((eps, u))

        x_mid = jnp.linspace(-0.9, 0.9, 200)
        max_diff = float(jnp.max(jnp.abs(u(x_mid) - y_outer(x_mid))))
        print(f"\neps={eps}:")
        print(f"  Solution length: {len(u)}")
        print(f"  max|y - y_outer| on [-0.9,0.9]: {max_diff:.4f}")
        assert abs(float(u(jnp.array(-1.0)))) < 1e-8
        assert abs(float(u(jnp.array(1.0)))) < 1e-8

    # Verify: as eps->0, interior solution approaches outer solution
    u_small = solutions[-1][1]
    x_mid = jnp.linspace(-0.5, 0.5, 100)
    diff_inner = float(jnp.max(jnp.abs(u_small(x_mid) - y_outer(x_mid))))
    print(f"\nSmallest eps={eps_vals[-1]}: max diff from outer on [-0.5,0.5] = {diff_inner:.4f}")
    assert diff_inner < 0.01  # should be very close in interior

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(-1.0, 1.0, 600)
    colors = ['b', 'r', 'g']

    fig, ax = plt.subplots()
    for (eps, u), c in zip(solutions, colors):
        ax.plot(x_plot, u(x_plot), color=c, linewidth=1.6, label=f"ε={eps}")
    ax.plot(x_plot, y_outer(x_plot), 'k--', linewidth=1.2, label="outer 1/(2-x²)")
    ax.set_xlabel("x"); ax.set_ylabel("y(x)")
    ax.set_title("Matched asymptotics: −ε y″ + (2−x²)y = 1, y(±1)=0", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "matched_asymp.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
