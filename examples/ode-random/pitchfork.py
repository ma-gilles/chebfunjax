"""Pitchfork bifurcation triggered by noise.

Solves the second-order ODE
  y'' = 2*c(t)*y - 4*y^3 + 0.003*f(t)
with c(t) = -1 + t/300 slowly crossing zero, demonstrating pitchfork
bifurcation triggered by a small random forcing.

Following ode-random/Pitchfork.m (actually Bifurcation.m) by Nick Trefethen (May 2017).

Original MATLAB: https://www.chebfun.org/examples/ode-random/Pitchfork.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj


def run():
    print("=" * 60)
    print("Pitchfork bifurcation triggered by noise")
    print("=" * 60)

    print("\nODE: y'' = 2*c(t)*y - 4*y^3 + eps*f(t)")
    print("c(t) = -1 + t/300, t in [0, 600]")
    print("Fixed points: y=0 (unstable for t>300), y=±sqrt(c/2) (stable for t>300)")

    domain = [0.0, 600.0]
    lam = 2.0
    eps = 0.003
    t_eval = np.linspace(0, 600, 1200)

    def solve_pitchfork(f_fn=None, damping=0.0):
        """Solve ODE y'' = 2*c(t)*y - 4*y^3 [+ damping*y'] + eps*f."""
        if f_fn is not None:
            # Sample the random function
            t_coarse = np.linspace(0, 600, 6000)
            f_coarse = np.array([float(f_fn(np.array(ti))) for ti in t_coarse])
        else:
            t_coarse = np.array([0.0, 600.0])
            f_coarse = np.array([0.0, 0.0])

        def rhs(t, y):
            ct = -1.0 + t / 300.0
            f_t = np.interp(t, t_coarse, f_coarse) if f_fn is not None else 0.0
            # y = [y0, y1=y']
            dydt = y[1]
            dy1dt = 2.0 * ct * y[0] - 4.0 * y[0]**3 + eps * f_t - damping * y[1]
            return [dydt, dy1dt]

        sol = solve_ivp(rhs, [0, 600], [0.0, 0.0], t_eval=t_eval,
                        method='RK45', rtol=1e-6, atol=1e-8,
                        max_step=0.5)
        return sol.y[0]

    # Without noise (dashed baseline)
    print("\nSolving undisturbed ODE (no noise)...")
    y0 = solve_pitchfork(f_fn=None, damping=0.0)
    print(f"  y(600) = {y0[-1]:.6f}  (should be ~0)")

    # With noise, no damping
    print("\nSolving with noise (lambda=2, no damping)...")
    f1 = cj.randnfun(lam, domain=domain, seed=1, big=True)
    f2 = cj.randnfun(lam, domain=domain, seed=2, big=True)
    y1 = solve_pitchfork(f_fn=f1, damping=0.0)
    y2 = solve_pitchfork(f_fn=f2, damping=0.0)
    print(f"  y1(600) = {y1[-1]:.4f}")
    print(f"  y2(600) = {y2[-1]:.4f}")

    # Check: solutions deviated from 0
    assert abs(y1[-1]) > 0.1 or abs(y2[-1]) > 0.1, \
        "Expected at least one path to deviate from y=0"
    print("  PASS: at least one path left the unstable y=0 branch")

    # With noise + damping
    print("\nSolving with noise and damping (0.2*y')...")
    y1d = solve_pitchfork(f_fn=f1, damping=0.2)
    y2d = solve_pitchfork(f_fn=f2, damping=0.2)
    print(f"  y1_damp(600) = {y1d[-1]:.4f}")
    print(f"  y2_damp(600) = {y2d[-1]:.4f}")

    # Stable branches (for t > 300, c(t) = (t-300)/300 > 0)
    t_stab = t_eval[t_eval > 300]
    c_stab = -1 + t_stab / 300.0
    branch_pos = np.sqrt(c_stab / 2.0)
    branch_neg = -np.sqrt(c_stab / 2.0)

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: no damping
    axes[0].plot(t_eval, y0, 'k--', linewidth=2, alpha=0.7, label='no noise')
    axes[0].plot(t_eval, y1, 'b-', linewidth=1.5, alpha=0.8, label='path 1')
    axes[0].plot(t_eval, y2, 'r-', linewidth=1.5, alpha=0.8, label='path 2')
    axes[0].plot(t_stab, branch_pos, 'k-', linewidth=1, alpha=0.4)
    axes[0].plot(t_stab, branch_neg, 'k-', linewidth=1, alpha=0.4)
    axes[0].axvline(300, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    axes[0].set_title("Pitchfork (undamped)", fontsize=11)
    axes[0].set_xlabel("t"); axes[0].set_ylabel("y")
    axes[0].set_ylim([-0.8, 0.8])
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # Panel 2: with damping
    axes[1].plot(t_eval, y0, 'k--', linewidth=2, alpha=0.7, label='no noise')
    axes[1].plot(t_eval, y1d, 'b-', linewidth=1.5, alpha=0.8, label='path 1')
    axes[1].plot(t_eval, y2d, 'r-', linewidth=1.5, alpha=0.8, label='path 2')
    axes[1].plot(t_stab, branch_pos, 'k-', linewidth=1, alpha=0.4)
    axes[1].plot(t_stab, branch_neg, 'k-', linewidth=1, alpha=0.4)
    axes[1].axvline(300, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    axes[1].set_title("Pitchfork with damping (0.2*y')", fontsize=11)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("y")
    axes[1].set_ylim([-0.8, 0.8])
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Pitchfork bifurcation: y'' = 2c(t)y - 4y³ + 0.003f", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "pitchfork.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
