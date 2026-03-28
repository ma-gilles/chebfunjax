"""Collective dynamics and consensus.

Demonstrates how two particles undergoing random walks can be attracted
together by a nonlinear coupling force, exhibiting consensus dynamics.

Two particles u, v each with random walk forcing f, g:
  du/dt = -f(t) + F*(u-v)*exp(-(u-v)^2)
  dv/dt = -g(t) + F*(v-u)*exp(-(v-u)^2)

When F is large enough, the particles synchronize.

Following ode-random/Consensus.m by Nick Trefethen (May 2017).

Original MATLAB: https://www.chebfun.org/examples/ode-random/Consensus.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

def run():
    print("=" * 60)
    print("Collective dynamics and consensus")
    print("=" * 60)

    print("\nTwo particles with random walk + mutual attraction force F")
    print("  du/dt = -f + F*(u-v)*exp(-(u-v)^2)")
    print("  dv/dt = -g + F*(v-u)*exp(-(v-u)^2)")

    domain = [0.0, 40.0]
    lam = 0.2
    d = 1.0  # initial separation
    t_eval = np.linspace(0, 40, 2000)

    # Generate random forcings (same for all experiments)
    f_fn = cj.randnfun(lam, domain=domain, seed=3, big=False)
    g_fn = cj.randnfun(lam, domain=domain, seed=4, big=False)
    t_grid = t_eval
    f_vals = np.array([float(f_fn(np.array(ti))) for ti in t_grid])
    g_vals = np.array([float(g_fn(np.array(ti))) for ti in t_grid])

    def solve_consensus(F):
        """Solve consensus system with attraction strength F."""
        def rhs(t, uv):
            fi = np.interp(t, t_grid, f_vals)
            gi = np.interp(t, t_grid, g_vals)
            u, v = uv[0], uv[1]
            diff = u - v
            attract = diff * np.exp(-diff**2)
            dudt = -fi + F * attract
            dvdt = -gi - F * attract  # symmetric: F*(v-u)*exp(-(v-u)^2) = -F*attract
            return [dudt, dvdt]

        sol = solve_ivp(rhs, [0, 40], [d, -d], t_eval=t_eval,
                        method='RK45', rtol=1e-7, atol=1e-9)
        return sol.y[0], sol.y[1]

    # Independent random walks (F=0)
    print("\n1. Independent random walks (F=0)")
    u0, v0 = solve_consensus(F=0.0)
    print(f"   u(40) = {u0[-1]:.3f}, v(40) = {v0[-1]:.3f}")
    print(f"   |u(40)-v(40)| = {abs(u0[-1]-v0[-1]):.3f}")

    # Strong attraction (F=3)
    print("\n2. Strong attraction (F=3)")
    u3, v3 = solve_consensus(F=3.0)
    gap3 = np.mean(np.abs(u3 - v3))
    print(f"   u(40) = {u3[-1]:.3f}, v(40) = {v3[-1]:.3f}")
    print(f"   Mean |u-v| = {gap3:.3f} (small → consensus)")

    # Weak attraction (F=1)
    print("\n3. Weak attraction (F=1)")
    u1, v1 = solve_consensus(F=1.0)
    gap1 = np.mean(np.abs(u1 - v1))
    print(f"   u(40) = {u1[-1]:.3f}, v(40) = {v1[-1]:.3f}")
    print(f"   Mean |u-v| = {gap1:.3f}")

    # Check: strong coupling leads to smaller gap than independent
    gap0 = np.mean(np.abs(u0 - v0))
    print(f"\n  Mean gap: F=0: {gap0:.3f}, F=1: {gap1:.3f}, F=3: {gap3:.3f}")
    assert gap3 < gap0, "Strong coupling should reduce gap vs independent walks"
    print("  PASS: strong coupling reduces particle separation")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3)

    def plot_pair(ax, t, u, v, title):
        ax.plot(t, u, color='#0072BD', linestyle='-', linewidth=2, alpha=0.8, label='u')
        ax.plot(t, v, color='#D95319', linestyle='-', linewidth=2, alpha=0.8, label='v')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    plot_pair(axes[0], t_eval, u0, v0, "Independent random walks (F=0)")
    plot_pair(axes[1], t_eval, u3, v3, "Strong attraction (F=3)")
    plot_pair(axes[2], t_eval, u1, v1, "Weak attraction (F=1)")

    fig.suptitle("Collective dynamics: consensus via attraction force", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "consensus.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True

if __name__ == "__main__":
    run()
