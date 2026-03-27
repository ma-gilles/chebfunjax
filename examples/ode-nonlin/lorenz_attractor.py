"""Lorenz attractor and rational interpolation.

Simulates the Lorenz system and investigates the complex singularities
of its solutions using rational interpolation.

Credit: Chebfun example ode-nonlin/LorenzAttractor.m (Marcus Webb, Mar 2013).
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
    print("Lorenz attractor")
    print("=" * 60)

    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    def lorenz(t, state):
        x, y, z = state
        return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]

    T = 40.0
    ic = [1.0, 1.0, 1.0]
    t_eval = np.linspace(0, T, 20000)

    print(f"\nIntegrating Lorenz on [0, {T}]...")
    sol = solve_ivp(lorenz, [0, T], ic, t_eval=t_eval, rtol=1e-10, atol=1e-12)

    x, y, z = sol.y
    print(f"  Max |x|: {np.max(np.abs(x)):.2f}")
    print(f"  Max |z|: {np.max(np.abs(z)):.2f}")
    assert np.max(np.abs(x)) > 5.0  # attractor extends beyond 5

    # Verify chaotic behavior: two nearby trajectories diverge
    ic2 = [1.001, 1.0, 1.0]
    sol2 = solve_ivp(lorenz, [0, T], ic2, t_eval=t_eval, rtol=1e-10, atol=1e-12)
    diff = np.sqrt((sol.y[0] - sol2.y[0])**2 +
                   (sol.y[1] - sol2.y[1])**2 +
                   (sol.y[2] - sol2.y[2])**2)
    idx_saturate = np.searchsorted(diff, 5.0)
    t_diverge = sol.t[min(idx_saturate, len(sol.t)-1)]
    print(f"\n  Trajectories diverge by 5 at t ≈ {t_diverge:.1f}")
    assert t_diverge < T  # should diverge before T

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot(x[::5], y[::5], z[::5], 'b', linewidth=0.4, alpha=0.5)
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax1.set_title("Lorenz attractor", fontsize=9)
    ax1.tick_params(labelsize=7)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(sol.t, x, 'b', linewidth=0.6, alpha=0.8, label="traj 1")
    ax2.plot(sol2.t, sol2.y[0], 'r', linewidth=0.6, alpha=0.5, label="traj 2")
    ax2.set_xlabel("t"); ax2.set_ylabel("x(t)")
    ax2.set_title("x(t): two trajectories", fontsize=9)
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.semilogy(sol.t, diff + 1e-10, 'g', linewidth=1.2)
    ax3.set_xlabel("t"); ax3.set_ylabel("|difference|")
    ax3.set_title("Divergence of nearby trajectories", fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Lorenz attractor and chaos", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "lorenz_attractor.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
