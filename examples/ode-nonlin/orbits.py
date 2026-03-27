"""Orbital mechanics: planet orbiting a fixed star.

Solves Newton's equations for a planet of unit mass orbiting a unit-mass
star at the origin with gravitational constant G=1.

Credit: Chebfun example ode-nonlin/Orbits.m (Nick Trefethen, Nov 2011).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

def run():
    print("=" * 60)
    print("Orbital mechanics: planet around a fixed star")
    print("=" * 60)

    # Planet at (-1, 1), moving east with speed v
    # G = 1, M = 1 (star at origin)
    # x'' = -x / r^3, y'' = -y / r^3
    # State: [x, y, vx, vy]

    def orbit_rhs(t, state):
        x, y, vx, vy = state
        r3 = (x**2 + y**2)**1.5
        return [vx, vy, -x/r3, -y/r3]

    T = 20.0
    t_eval = np.linspace(0, T, 5000)

    # Three different initial speeds
    speeds = [0.5, 1.0, 1.3]
    solutions = []

    print(f"\n{'speed':>8}  {'orbit type':>20}  {'min r':>10}  {'max r':>10}")
    print("-" * 52)

    for v in speeds:
        ic = [-1.0, 1.0, v, 0.0]  # planet at (-1,1), heading east
        sol = solve_ivp(orbit_rhs, [0, T], ic, t_eval=t_eval, rtol=1e-10)
        x, y = sol.y[0], sol.y[1]
        r = np.sqrt(x**2 + y**2)
        min_r = np.min(r)
        max_r = np.max(r)

        # Determine orbit type from energy
        vx0, vy0 = ic[2], ic[3]
        r0 = np.sqrt(ic[0]**2 + ic[1]**2)
        energy = 0.5*(vx0**2 + vy0**2) - 1.0/r0
        orbit_type = "ellipse" if energy < 0 else "hyperbola" if energy > 0 else "parabola"

        print(f"  {v:8.2f}  {orbit_type:>20}  {min_r:10.4f}  {max_r:10.4f}")
        solutions.append((v, sol, orbit_type))
        assert min_r > 0.01  # should not hit the star

    # Verify conservation of energy
    for v, sol, oname in solutions:
        x, y, vx, vy = sol.y
        r = np.sqrt(x**2 + y**2)
        E = 0.5*(vx**2 + vy**2) - 1.0/r
        E_variation = np.max(E) - np.min(E)
        print(f"  v={v}: energy variation = {E_variation:.2e}")
        assert E_variation < 0.01, f"Energy not conserved: {E_variation}"

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    colors = ['b', 'r', 'g']
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(0, 0, 'y*', markersize=12, label="star")
    for (v, sol, otype), c in zip(solutions, colors):
        x, y = sol.y[0], sol.y[1]
        axes[0].plot(x, y, color=c, linewidth=1.2, label=f"v={v} ({otype})")
        axes[0].plot(x[0], y[0], 'o', color=c, markersize=5)
    axes[0].set_title("Orbital trajectories", fontsize=10)
    axes[0].legend(fontsize=7); axes[0].set_aspect('equal')

    for (v, sol, otype), c in zip(solutions, colors):
        r = np.sqrt(sol.y[0]**2 + sol.y[1]**2)
        axes[1].plot(sol.t, r, color=c, linewidth=1.2, label=f"v={v}")
    axes[1].set_title("Distance from star vs time", fontsize=10)
    axes[1].legend(fontsize=7)

    fig.suptitle("Orbital mechanics (G=M=1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "orbits.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
