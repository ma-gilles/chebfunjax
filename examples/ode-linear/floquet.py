"""Floquet theory for periodic linear ODE systems.

Considers the Mathieu equation  u'' + (a - 2q cos(2t)) u = 0
and computes the Floquet multipliers numerically to study stability.

Credit: Chebfun example ode-linear/Floquet.m (R. M. Slevinsky, Oct 2014).
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

from chebfunjax.operators.chebop import Chebop

def monodromy(a, q):
    """Compute the 2x2 monodromy matrix for the Mathieu equation
    u'' + (a - 2q cos(2t)) u = 0 over one period T=pi."""
    T = np.pi

    def rhs(t, y):
        # y = [u, u']
        c = a - 2 * q * np.cos(2 * t)
        return [y[1], -c * y[0]]

    # Column 1: IC = [1, 0]
    sol1 = solve_ivp(rhs, [0, T], [1.0, 0.0], dense_output=True,
                     rtol=1e-12, atol=1e-14)
    # Column 2: IC = [0, 1]
    sol2 = solve_ivp(rhs, [0, T], [0.0, 1.0], dense_output=True,
                     rtol=1e-12, atol=1e-14)

    M = np.array([
        [sol1.y[0, -1], sol2.y[0, -1]],
        [sol1.y[1, -1], sol2.y[1, -1]],
    ])
    return M

def run():
    print("=" * 60)
    print("Floquet theory: Mathieu equation stability")
    print("=" * 60)

    # Mathieu equation: u'' + (a - 2q cos(2t)) u = 0
    # Stable if |trace(M)| < 2  (Floquet multipliers on unit circle)

    # Stability diagram over (a, q) grid
    a_vals = np.linspace(-1.0, 6.0, 40)
    q_vals = np.linspace(0.0, 3.0, 30)
    stable = np.zeros((len(q_vals), len(a_vals)), dtype=bool)

    print("\nComputing Mathieu stability diagram...")
    for i, q in enumerate(q_vals):
        for j, a in enumerate(a_vals):
            M = monodromy(a, q)
            tr = np.trace(M)
            stable[i, j] = abs(tr) < 2.0

    # Check a few known stable/unstable points
    # a=1, q=0: standard harmonic osc => stable (trace M = 2 cos(pi) = -2... borderline)
    M0 = monodromy(1.0, 0.0)
    tr0 = np.trace(M0)
    print(f"\na=1, q=0: trace(M) = {tr0:.6f}  (exact: 2 cos(pi) = -2)")
    assert abs(abs(tr0) - 2.0) < 0.01   # borderline

    M1 = monodromy(4.0, 0.0)
    tr1 = np.trace(M1)
    print(f"a=4, q=0: trace(M) = {tr1:.6f}  (exact: 2 cos(2pi) = 2)")
    assert abs(tr1 - 2.0) < 0.01

    n_stable = int(np.sum(stable))
    print(f"\nStable cells: {n_stable}/{stable.size}")
    assert n_stable > 0

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    axes[0].contourf(a_vals, q_vals, stable.astype(float), levels=[0.5, 1.5],
                     colors=['lightblue'], alpha=0.8)
    axes[0].contour(a_vals, q_vals, stable.astype(float), levels=[0.5], colors='k')
    axes[0].set_title("Mathieu stability diagram (blue = stable)", fontsize=10)

    # Trajectory at a stable and unstable point
    t_arr = np.linspace(0, 5 * np.pi, 500)
    T = np.pi

    def mathieu_traj(a, q, t_end, n=500):
        def rhs(t, y):
            c = a - 2 * q * np.cos(2 * t)
            return [y[1], -c * y[0]]
        sol = solve_ivp(rhs, [0, t_end], [1.0, 0.0], t_eval=np.linspace(0, t_end, n),
                        rtol=1e-10)
        return sol.t, sol.y[0]

    t1, u1 = mathieu_traj(1.0, 0.5, 5 * np.pi)
    t2, u2 = mathieu_traj(0.5, 1.5, 5 * np.pi)

    axes[1].plot(t1 / np.pi, u1, 'b', linewidth=1.4, label="stable (a=1,q=0.5)")
    axes[1].plot(t2 / np.pi, u2, 'r', linewidth=1.4, label="unstable (a=0.5,q=1.5)")
    axes[1].set_title("Mathieu equation trajectories", fontsize=10)
    axes[1].legend(fontsize=8)

    fig.suptitle("Floquet theory: Mathieu equation", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "floquet.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
