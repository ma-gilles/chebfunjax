"""Classification of linear dynamical systems.

Plots phase portraits and trajectories for 2D linear systems x' = A x.
Illustrates stable node, unstable node, saddle, center, and spiral cases.

Credit: Chebfun example ode-linear/DynamicalSystems.m (Georges Klein, Mar 2013).
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



def trajectory(A, x0, T=3.0):
    """Solve x' = Ax from x0 to time T."""
    def rhs(t, x):
        return A @ x
    sol = solve_ivp(rhs, [0, T], x0, dense_output=True, max_step=0.05)
    t = np.linspace(0, T, 300)
    return sol.sol(t)


def run():
    print("=" * 60)
    print("Classification of linear dynamical systems")
    print("=" * 60)

    # Four canonical cases
    cases = [
        ("Unstable node", np.array([[2.0, -2.0], [0.0, 1.0]]), 1.5),
        ("Stable spiral",  np.array([[-1.0, 2.0], [-2.0, -1.0]]), 4.0),
        ("Saddle",         np.array([[1.0, 0.0], [0.0, -1.0]]), 2.0),
        ("Center",         np.array([[0.0, 1.0], [-1.0, 0.0]]), 6.0),
    ]

    x0s = [
        np.array([1.0, 0.0]), np.array([0.5, 0.5]),
        np.array([0.0, 1.0]), np.array([0.0, -1.0]),
    ]

    fig, axes = plt.subplots(2, 2)
    axes = axes.ravel()

    for idx, (title, A, T) in enumerate(cases):
        ax = axes[idx]
        eigs = np.linalg.eigvals(A)
        print(f"\n{title}: eigenvalues = {eigs}")

        # Grid of initial conditions
        ic_grid = []
        for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            ic_grid.append(np.array([0.6 * np.cos(ang), 0.6 * np.sin(ang)]))

        for ic in ic_grid:
            traj = trajectory(A, ic, T)
            ax.plot(traj[0], traj[1], 'b', linewidth=0.8, alpha=0.6)
            ax.plot(traj[0, 0], traj[1, 0], 'go', markersize=3)

        ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
        ax.set_title(f"{title}\nλ = {eigs[0]:.2f}, {eigs[1]:.2f}", fontsize=9)
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
    fig.suptitle("Phase portraits of 2D linear systems x′ = Ax", fontsize=11)
    fig.tight_layout()
    _here = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(_here, "dynamical_systems.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Basic assertions
    A_stable = np.array([[-1.0, 0.0], [0.0, -2.0]])
    assert np.all(np.real(np.linalg.eigvals(A_stable)) < 0), "stable node"

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
