"""Near-nonuniqueness and near-nonexistence for linear ODE BVPs.

Solves  eps u'' - x u' + u = 1,  u(±1) = 0  for eps << 1.
The solution should be even by symmetry, but numerical near-singular
behavior illustrates the concept of near-nonuniqueness.

Credit: Chebfun example ode-linear/NearNonuniqueness.m (Nick Trefethen, Oct 2016).
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
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Near-nonuniqueness: eps u'' - x u' + u = 1, u(±1)=0")
    print("=" * 60)

    dom = (-1.0, 1.0)
    eps_vals = [0.1, 0.05, 0.02]

    solutions = []
    for eps in eps_vals:
        L = Chebop(lambda x, u: eps * u.diff(2) - x * u.diff() + u, domain=dom)
        L.lbc = 0.0
        L.rbc = 0.0
        u = L.solve(cj.chebfun(lambda x: jnp.ones_like(x), domain=dom))
        solutions.append((eps, u))

        u_vals = u(jnp.linspace(-1.0, 1.0, 400))
        print(f"\neps={eps}:")
        print(f"  Solution length: {len(u)}")
        print(f"  u(0) = {float(u(jnp.array(0.0))):.6f}")
        print(f"  u(-0.5) = {float(u(jnp.array(-0.5))):.6f}")
        print(f"  u(0.5) = {float(u(jnp.array(0.5))):.6f}")

        # BCs should be satisfied
        assert abs(float(u(jnp.array(-1.0)))) < 1e-8
        assert abs(float(u(jnp.array(1.0)))) < 1e-8

    # The solution should be approximately even (symmetric)
    # Check evenness: u(-x) ≈ u(x)
    u_test = solutions[0][1]
    x_check = jnp.linspace(0.0, 0.9, 50)
    asymm = float(jnp.max(jnp.abs(u_test(x_check) - u_test(-x_check))))
    print(f"\nAsymmetry of solution (eps=0.1): {asymm:.2e}")
    # For not-too-small eps, solution should be nearly even
    assert asymm < 0.1

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(-1.0, 1.0, 500)
    colors = ['b', 'r', 'g']

    fig, ax = plt.subplots(figsize=(7, 4))
    for (eps, u), c in zip(solutions, colors):
        ax.plot(x_plot, u(x_plot), color=c, linewidth=1.6, label=f"ε={eps}")
    ax.set_xlabel("x"); ax.set_ylabel("u(x)")
    ax.set_title("Near-nonuniqueness: ε u″ − x u′ + u = 1, u(±1)=0", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "near_nonuniqueness.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
