"""Phase-locking in a Duffing-type equation.

Solves the bistable ODE y' = t*y - y^3 + f where f is a random function,
demonstrating phase-locking to one of two stable branches.

Following ode-random/PhaseLocking.m by Burrage & Trefethen (May 2017).

Original MATLAB: https://www.chebfun.org/examples/ode-random/PhaseLocking.html
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
    print("Phase-locking in a Duffing-type equation")
    print("=" * 60)

    print("\nODE: y' = t*y - y^3 + f")
    print("Fixed points of det. part: y=0, y=±sqrt(t)")
    print("For large t, noise is small compared to the gap → phase locking")

    domain = [0.0, 6.0]
    t_eval = np.linspace(0, 6, 600)
    t_coarse = np.linspace(0, 6, 1200)

    def solve_phase(lam, n_paths=6, seed_base=0):
        """Solve bistable ODE with random forcing."""
        paths = []
        for k in range(n_paths):
            f_fn = cj.randnfun(lam, domain=domain, seed=seed_base + k, big=True)
            f_coarse = f_fn(t_coarse)

            def rhs(t, y, fc=f_coarse):
                f_t = np.interp(t, t_coarse, fc)
                return [t * y[0] - y[0]**3 + f_t]

            sol = solve_ivp(rhs, [0, 6], [0.0], t_eval=t_eval,
                            method='RK45', rtol=1e-6, atol=1e-8,
                            max_step=0.02)
            paths.append(sol.y[0])
        return paths

    # lambda = 0.2
    print("\nSolving with lambda=0.2 (6 paths)...")
    paths_02 = solve_phase(0.2, n_paths=6, seed_base=0)

    # Count paths that locked to positive/negative branch
    y_final = np.array([p[-1] for p in paths_02])
    expected_pos = np.sqrt(6.0)  # stable point at t=6 is ±sqrt(6)
    n_pos = np.sum(y_final > 0.5 * expected_pos)
    n_neg = np.sum(y_final < -0.5 * expected_pos)
    print(f"  Positive branch: {n_pos} paths, negative branch: {n_neg} paths")
    print(f"  (Random due to random forcing, roughly 50/50)")

    # lambda = 0.05 (finer noise → faster locking)
    print("\nSolving with lambda=0.05 (6 paths)...")
    paths_005 = solve_phase(0.05, n_paths=6, seed_base=10)
    y_final_005 = np.array([p[-1] for p in paths_005])
    n_pos_005 = np.sum(y_final_005 > 0.5 * expected_pos)
    n_neg_005 = np.sum(y_final_005 < -0.5 * expected_pos)
    print(f"  Positive branch: {n_pos_005} paths, negative branch: {n_neg_005} paths")

    # Most paths should lock to one branch by t=6
    n_locked = n_pos + n_neg
    assert n_locked >= 4, f"Only {n_locked}/6 paths locked to a branch"
    print(f"\n  PASS: {n_locked}/6 paths locked to ±sqrt(t) branch")

    # Stable branches
    t_stab = t_eval[t_eval > 0]
    branch_pos = np.sqrt(t_stab)
    branch_neg = -np.sqrt(t_stab)

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    colors = plt.cm.Set2(np.linspace(0, 1, 6))
    for i, p in enumerate(paths_02):
        axes[0].plot(t_eval, p, color=colors[i], linewidth=1.5, alpha=0.8)
    axes[0].plot(t_stab, branch_pos, 'k--', linewidth=2, label='±√t')
    axes[0].plot(t_stab, branch_neg, 'k--', linewidth=2)
    axes[0].set_title("Phase-locking (λ=0.2, 6 paths)", fontsize=11)
    axes[0].set_xlabel("t"); axes[0].set_ylabel("y")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-3, 3])

    for i, p in enumerate(paths_005):
        axes[1].plot(t_eval, p, color=colors[i], linewidth=1.5, alpha=0.8)
    axes[1].plot(t_stab, branch_pos, 'k--', linewidth=2, label='±√t')
    axes[1].plot(t_stab, branch_neg, 'k--', linewidth=2)
    axes[1].set_title("Phase-locking (λ=0.05, finer noise)", fontsize=11)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("y")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-3, 3])

    fig.suptitle("Phase-locking: y' = ty - y³ + f", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "phase_locking.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
