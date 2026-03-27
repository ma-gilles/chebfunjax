"""Random level hopping.

Solves the bistable ODE y' = -2*sin(2*pi*y) + f where f is a random
function, demonstrating hopping between integer fixed points.

Following ode-random/LevelHopping.m by Nick Trefethen (May 2017).

Original MATLAB: https://www.chebfun.org/examples/ode-random/LevelHopping.html
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
    print("Random level hopping")
    print("=" * 60)

    print("\nODE: y' = -2*sin(2*pi*y) + f")
    print("Fixed points: integers n (stable), half-integers (unstable)")

    domain = [0.0, 100.0]
    t_eval = np.linspace(0, 100, 2000)
    t_grid = np.linspace(0, 100, 10000)

    def solve_level_hop(lam, seed=0):
        """Solve ODE with random forcing of given wavelength."""
        f_fn = cj.randnfun(lam, domain=domain, seed=seed, big=False)
        f_vals = f_fn(t_grid)

        def rhs(t, y):
            f_t = np.interp(t, t_grid, f_vals)
            return [-2 * np.sin(2 * np.pi * y[0]) + f_t]

        sol = solve_ivp(rhs, [0, 100], [0.0], t_eval=t_eval,
                        method='RK45', rtol=1e-6, atol=1e-8)
        return sol.t, sol.y[0]

    # lambda = 0.4 (coarser noise)
    print("\nSolving with lambda = 0.4...")
    t1, y1 = solve_level_hop(0.4, seed=0)
    n_hops_1 = np.sum(np.abs(np.diff(np.round(y1))) > 0.5)
    print(f"  Level hops (approx): {n_hops_1}")
    print(f"  y(100) = {y1[-1]:.3f}")
    levels_visited = np.unique(np.round(y1)).astype(int)
    print(f"  Levels visited: {levels_visited}")

    # lambda = 0.2 (finer noise → more hops)
    print("\nSolving with lambda = 0.2...")
    t2, y2 = solve_level_hop(0.2, seed=0)
    n_hops_2 = np.sum(np.abs(np.diff(np.round(y2))) > 0.5)
    print(f"  Level hops (approx): {n_hops_2}")

    # Check y stays near integer levels for most of the time
    y1_residual = y1 - np.round(y1)
    frac_near_integer = np.mean(np.abs(y1_residual) < 0.3)
    print(f"\n  Fraction of time near integer: {frac_near_integer:.2%}")
    assert frac_near_integer > 0.5, "Should spend >50% time near integers"
    print("  PASS: trajectory spends most time near integer fixed points")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(t1, y1, 'b-', linewidth=1.5)
    for n in range(int(y1.min()) - 1, int(y1.max()) + 2):
        axes[0].axhline(n, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    axes[0].set_title("Level hopping (λ=0.4)", fontsize=11)
    axes[0].set_xlabel("t"); axes[0].set_ylabel("y")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t2, y2, 'r-', linewidth=1.2)
    for n in range(int(y2.min()) - 1, int(y2.max()) + 2):
        axes[1].axhline(n, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    axes[1].set_title("Level hopping (λ=0.2, finer noise)", fontsize=11)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("y")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Random level hopping: y' = -2sin(2πy) + f", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "level_hopping.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
