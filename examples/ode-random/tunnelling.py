"""Tunnelling between metastable states.

Solves the bistable ODE
  y' = y - y^3 + f
where f is a random forcing. Stable fixed points at y=±1, unstable at y=0.
Demonstrates noise-driven tunnelling between the two stable states.

Following ode-random/Tunnelling.m by Nick Trefethen (May 2017).

Original MATLAB: https://www.chebfun.org/examples/ode-random/Tunnelling.html
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
    print("Tunnelling between metastable states")
    print("=" * 60)

    print("\nODE: y' = y - y^3 + f  (bistable)")
    print("Stable fixed points at y = ±1, unstable at y = 0")
    print("Noise drives tunnelling between the two wells")

    lam = 0.5
    eps = 0.45  # noise amplitude from MATLAB code

    # ----------------------------------------------------------------
    # Part 1: 6 short trajectories on [0, 30]
    # ----------------------------------------------------------------
    domain_short = [0.0, 30.0]
    t_eval_short = np.linspace(0, 30, 600)

    print("\nPart 1: 6 paths on [0, 30]")
    paths_short = []
    for k in range(6):
        f_fn = cj.randnfun(lam, domain=domain_short, seed=4 + k, big=False)
        t_grid = np.linspace(0, 30, 3000)
        f_vals = np.array([float(f_fn(np.array(ti))) for ti in t_grid])

        def rhs(t, y, f_v=f_vals, t_g=t_grid):
            f_t = np.interp(t, t_g, f_v)
            return [y[0] - y[0]**3 + eps * f_t]

        sol = solve_ivp(rhs, [0, 30], [0.0], t_eval=t_eval_short,
                        method='RK45', rtol=1e-6, atol=1e-8)
        paths_short.append(sol.y[0])
        print(f"  Path {k+1}: y(30) = {sol.y[0,-1]:.3f}")

    # ----------------------------------------------------------------
    # Part 2: Long trajectory on [0, 800] showing tunnelling
    # ----------------------------------------------------------------
    print("\nPart 2: Long trajectory on [0, 800] showing tunnelling")
    domain_long = [0.0, 800.0]
    t_eval_long = np.linspace(0, 800, 4000)

    f_long = cj.randnfun(lam, domain=domain_long, seed=4, big=False)
    t_grid_long = np.linspace(0, 800, 8000)
    f_vals_long = np.array([float(f_long(np.array(ti))) for ti in t_grid_long])

    def rhs_long(t, y):
        f_t = np.interp(t, t_grid_long, f_vals_long)
        return [y[0] - y[0]**3 + eps * f_t]

    sol_long = solve_ivp(rhs_long, [0, 800], [0.0], t_eval=t_eval_long,
                         method='RK45', rtol=1e-6, atol=1e-8)
    y_long = sol_long.y[0]

    # Count tunnelling events
    y_sign = np.sign(y_long[np.abs(y_long) > 0.3])
    if len(y_sign) > 1:
        n_tunnels = np.sum(np.diff(y_sign) != 0)
    else:
        n_tunnels = 0
    print(f"  Tunnelling events (approx): {n_tunnels}")
    print(f"  y(800) = {y_long[-1]:.3f}")

    # ----------------------------------------------------------------
    # Part 3: Larger noise → faster tunnelling
    # ----------------------------------------------------------------
    print("\nPart 3: Larger noise amplitude (eps=0.60)")
    eps2 = 0.60
    f_vals_long2 = f_vals_long * (eps2 / eps)  # just rescale

    def rhs_long2(t, y):
        f_t = np.interp(t, t_grid_long, f_vals_long2)
        return [y[0] - y[0]**3 + f_t]

    sol_long2 = solve_ivp(rhs_long2, [0, 800], [0.0], t_eval=t_eval_long,
                          method='RK45', rtol=1e-6, atol=1e-8)
    y_long2 = sol_long2.y[0]
    y_sign2 = np.sign(y_long2[np.abs(y_long2) > 0.3])
    n_tunnels2 = np.sum(np.diff(y_sign2) != 0) if len(y_sign2) > 1 else 0
    print(f"  Tunnelling events (approx): {n_tunnels2}")

    # Check: trajectories spend time near ±1
    y_residual = np.abs(np.abs(y_long) - 1.0)
    frac_near_stable = np.mean(y_residual < 0.3)
    print(f"\n  Fraction of time near stable points: {frac_near_stable:.1%}")
    assert frac_near_stable > 0.3, \
        f"Should spend >30% time near ±1, got {frac_near_stable:.1%}"
    print("  PASS: trajectory spends time near metastable fixed points")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = plt.cm.Set2(np.linspace(0, 1, 6))
    for i, p in enumerate(paths_short):
        axes[0].plot(t_eval_short, p, color=colors[i], linewidth=1.5, alpha=0.8)
    axes[0].axhline(1, color='k', linestyle='--', linewidth=1, alpha=0.4)
    axes[0].axhline(-1, color='k', linestyle='--', linewidth=1, alpha=0.4)
    axes[0].axhline(0, color='k', linestyle=':', linewidth=0.8, alpha=0.3)
    axes[0].set_title("Bistability (6 paths, t=[0,30])", fontsize=10)
    axes[0].set_xlabel("t"); axes[0].set_ylabel("y")
    axes[0].set_ylim([-1.7, 1.7]); axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_eval_long, y_long, 'b-', linewidth=0.6, alpha=0.9)
    axes[1].axhline(1, color='k', linestyle='--', linewidth=1, alpha=0.4)
    axes[1].axhline(-1, color='k', linestyle='--', linewidth=1, alpha=0.4)
    axes[1].set_title(f"Tunnelling (eps=0.45, {n_tunnels} events)", fontsize=10)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("y")
    axes[1].set_ylim([-1.7, 1.7]); axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_eval_long, y_long2, 'r-', linewidth=0.6, alpha=0.9)
    axes[2].axhline(1, color='k', linestyle='--', linewidth=1, alpha=0.4)
    axes[2].axhline(-1, color='k', linestyle='--', linewidth=1, alpha=0.4)
    axes[2].set_title(f"Faster tunnelling (eps=0.60, {n_tunnels2} events)", fontsize=10)
    axes[2].set_xlabel("t"); axes[2].set_ylabel("y")
    axes[2].set_ylim([-1.7, 1.7]); axes[2].grid(True, alpha=0.3)

    fig.suptitle("Tunnelling: y' = y - y³ + f", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "tunnelling.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
