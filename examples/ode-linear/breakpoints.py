"""Boundary layers and convergence of spectral methods.

Solves the advection-diffusion BVP
  -eps * u'' - u' = 1,  u(0) = u(1) = 0
for decreasing eps = 1e-1, ..., 1e-4, demonstrating how the Chebfun
length grows as the boundary layer thins.

Credit: Chebfun example ode-linear/Breakpoints.m (Nick Trefethen, Jan 2016).
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
    print("Boundary layers: convergence of spectral methods")
    print("=" * 60)

    dom = (0.0, 1.0)
    eps_values = [1e-1, 1e-2, 1e-3, 1e-4]
    solutions = []

    print(f"\n{'eps':>10}  {'length':>8}  {'max(u)':>12}")
    print("-" * 34)
    for eps in eps_values:
        N = Chebop(lambda x, u: -eps * u.diff(2) - u.diff(), domain=dom)
        N.lbc = 0.0
        N.rbc = 0.0
        u = N.solve(1.0)
        x_test = jnp.linspace(0.0, 1.0, 500)
        max_u = float(jnp.max(u(x_test)))
        print(f"  {eps:10.1e}  {len(u):8d}  {max_u:12.6f}")
        solutions.append((eps, u))
        assert abs(float(u(jnp.array(0.0)))) < 1e-8
        assert abs(float(u(jnp.array(1.0)))) < 1e-8

    # Larger eps should need fewer coefficients
    lengths = [len(u) for _, u in solutions]
    assert lengths[0] <= lengths[-1], "Larger eps should yield shorter chebfun"

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    colors = ['b', 'r', 'g', 'm']
    x_plot = jnp.linspace(0.0, 1.0, 600)

    fig, ax = plt.subplots(figsize=(7, 4))
    for (eps, u), c in zip(solutions, colors):
        ax.plot(x_plot, u(x_plot), color=c, linewidth=1.6,
                label=f"ε = {eps:.0e}")
    ax.legend(fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("u(x)")
    ax.set_title("Boundary layers: −ε u″ − u′ = 1, u(0)=u(1)=0", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "breakpoints.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
