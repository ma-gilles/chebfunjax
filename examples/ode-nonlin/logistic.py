"""Logistic map and chaos.

Illustrates the logistic map  x_{n+1} = r x_n (1-x_n)  and its
chaotic behavior using Chebfun to analyze the orbit structure.

Credit: Chebfun example ode-nonlin/Logistic.m (Nick Trefethen, Jul 2013).
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
    print("Logistic map and chaos")
    print("=" * 60)

    # x_{n+1} = r * x_n * (1 - x_n)
    # Bifurcation diagram: vary r, find attractor from x0=0.5

    r_vals = np.linspace(2.8, 4.0, 800)
    x0 = 0.5
    n_transient = 500
    n_display = 100
    bifurcation_x = []
    bifurcation_r = []

    for r in r_vals:
        x = x0
        for _ in range(n_transient):
            x = r * x * (1 - x)
        for _ in range(n_display):
            x = r * x * (1 - x)
            bifurcation_r.append(r)
            bifurcation_x.append(x)

    print(f"\nBifurcation diagram: {len(bifurcation_r)} points")

    # Verify: for r < 3, single fixed point x* = 1 - 1/r
    r_stable = 2.5
    x = 0.5
    for _ in range(1000):
        x = r_stable * x * (1 - x)
    x_fixed_numerical = x
    x_fixed_exact = 1.0 - 1.0 / r_stable
    print(f"\nr={r_stable}: fixed point = {x_fixed_numerical:.8f}  (exact: {x_fixed_exact:.8f})")
    assert abs(x_fixed_numerical - x_fixed_exact) < 1e-8

    # Period-2 orbit for r > 3
    r_2cycle = 3.2
    x = 0.5
    for _ in range(1000):
        x = r_2cycle * x * (1 - x)
    orbit_2 = set()
    for _ in range(100):
        x = r_2cycle * x * (1 - x)
        orbit_2.add(round(x, 6))
    print(f"r={r_2cycle}: orbit = {sorted(orbit_2)[:5]}  (should be 2-cycle)")
    assert len(orbit_2) == 2

    # Use Chebop to analyze: find fixed points of f(x) = rx(1-x)
    print("\nFixed points of f(x) = r*x*(1-x) via Chebop (r=3.5):")
    r_fp = 3.5
    # Fixed points: r*x*(1-x) = x => r*(1-x) = 1 => x = 1 - 1/r
    dom = (0.0, 1.0)
    # Use chebfun to find roots of f(x) - x
    h = cj.chebfun(lambda x: r_fp * x * (1 - x) - x, domain=dom)
    roots = np.sort(np.array(h.roots()))
    print(f"  Roots of rx(1-x)-x: {roots}")
    assert len(roots) >= 1  # should find x=0 and x=1-1/r

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(bifurcation_r, bifurcation_x, ',k', markersize=0.5, alpha=0.3)
    axes[0].set_title("Logistic map bifurcation diagram", fontsize=10)

    # Single trajectory
    r_traj = 3.7
    x = 0.5
    n_traj = 80
    traj = [x]
    for _ in range(n_traj):
        x = r_traj * x * (1 - x)
        traj.append(x)
    axes[1].plot(range(n_traj + 1), traj, 'b.-', markersize=4, linewidth=0.8)
    axes[1].set_title(f"Chaotic trajectory (r={r_traj})", fontsize=10)

    fig.suptitle("Logistic map: x_{n+1} = r x_n (1−x_n)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "logistic.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
