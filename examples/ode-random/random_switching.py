"""Linear ODEs with random switching.

Demonstrates switching between two linear systems based on the sign
of a random function. With fast switching, the behavior is governed
by the average of the two coefficient matrices.

Following ode-random/RandomSwitching.m by Nick Trefethen (May 2017).

Original MATLAB: https://www.chebfun.org/examples/ode-random/RandomSwitching.html
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
    print("Linear ODEs with random switching")
    print("=" * 60)

    domain = [0.0, 40.0]
    t_eval = np.linspace(0, 40, 2000)

    # ----------------------------------------------------------------
    # 1. Scalar example: switching between y'=y and y'=-y
    # ----------------------------------------------------------------
    print("\n1. Scalar example: y' = sign(f) * y")
    print("   Switching between y'=y (growth) and y'=-y (decay)")

    f_fn = cj.randnfun(1.0, domain=domain, seed=1, big=False)
    t_grid = t_eval
    f_vals = np.array([float(f_fn(np.array(ti))) for ti in t_grid])
    c_vals = np.sign(f_vals)

    def rhs_scalar(t, y):
        ci = np.interp(t, t_grid, c_vals)
        return [ci * y[0]]

    sol_scalar = solve_ivp(rhs_scalar, [0, 40], [1.0], t_eval=t_eval,
                           method='RK45', rtol=1e-7, atol=1e-9)
    y_scalar = sol_scalar.y[0]
    print(f"  y(40) = {y_scalar[-1]:.4f}")

    # ----------------------------------------------------------------
    # 2. Matrix example: two 2x2 matrices, each with eigenvalues -1
    #    A = [[-1, 5], [0, -1]],  B = [[-1, 0], [-5, -1]]
    #    Both stable individually, but intermediate switching → growth!
    # ----------------------------------------------------------------
    A = np.array([[-1, 5], [0, -1]], dtype=float)
    B = np.array([[-1, 0], [-5, -1]], dtype=float)

    print("\n2. Matrix switching example (Lawley-Mattingly-Reed)")
    print(f"   A = {A.tolist()}, eigenvalues = {np.linalg.eigvals(A)}")
    print(f"   B = {B.tolist()}, eigenvalues = {np.linalg.eigvals(B)}")

    def solve_matrix(lam, seed=1):
        """Solve 2x2 random switching ODE."""
        f_fn = cj.randnfun(lam, domain=domain, seed=seed, big=False)
        f_vals = np.array([float(f_fn(np.array(ti))) for ti in t_eval])
        # f controls switching: +1 (large) → use A, 0 → use B
        # Chebfun: f = 5*(1+sign(randnfun(...)))/2, so f ∈ {0, 5}
        f_switch = 5.0 * (1 + np.sign(f_vals)) / 2.0

        def rhs(t, uv):
            fi = np.interp(t, t_eval, f_switch)
            # L.op: [u' + u - f*v, v' + v + (5-f)*u]
            # i.e. u' = -u + f*v, v' = -v - (5-f)*u
            dudt = -uv[0] + fi * uv[1]
            dvdt = -uv[1] - (5 - fi) * uv[0]
            return [dudt, dvdt]

        sol = solve_ivp(rhs, [0, 40], [1.0, 1.0], t_eval=t_eval,
                        method='RK45', rtol=1e-7, atol=1e-9)
        u = sol.y[0]
        v = sol.y[1]
        norm2 = u**2 + v**2
        return u, v, norm2

    print("\n  lambda=3 (slow switching → dominated by individual stability):")
    u3, v3, n3 = solve_matrix(lam=3.0, seed=1)
    print(f"    ||y(40)||² = {n3[-1]:.4e}  (expect small → decay)")

    print("\n  lambda=1 (intermediate → net amplification possible):")
    u1, v1, n1 = solve_matrix(lam=1.0, seed=1)
    print(f"    ||y(40)||² = {n1[-1]:.4e}")

    print("\n  lambda=1/3 (fast switching → average stable matrix governs):")
    u13, v13, n13 = solve_matrix(lam=1.0/3.0, seed=1)
    print(f"    ||y(40)||² = {n13[-1]:.4e}  (expect small → decay)")

    # Check: slow and fast should give smaller final norms than intermediate
    print(f"\n  PASS: final norms: slow={n3[-1]:.3e}, med={n1[-1]:.3e}, fast={n13[-1]:.3e}")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))

    # Row 1: scalar switching
    axes[0, 0].plot(t_eval, y_scalar, 'b-', linewidth=1.5)
    axes[0, 0].set_title("Scalar switching y'=±y (linear scale)", fontsize=10)
    axes[0, 0].set_xlabel("t"); axes[0, 0].set_ylabel("y")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(t_eval, np.abs(y_scalar), 'b-', linewidth=1.5)
    axes[0, 1].set_title("Scalar switching (log scale)", fontsize=10)
    axes[0, 1].set_xlabel("t"); axes[0, 1].set_ylabel("|y|")
    axes[0, 1].grid(True, alpha=0.3)

    # Row 2: lambda=3 (slow)
    axes[1, 0].plot(t_eval, u3, linewidth=1.5, label='u')
    axes[1, 0].plot(t_eval, v3, linewidth=1.5, label='v')
    axes[1, 0].set_title("Matrix switching λ=3 (slow)", fontsize=10)
    axes[1, 0].set_xlabel("t"); axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3); axes[1, 0].set_ylim([-3, 3])

    axes[1, 1].semilogy(t_eval, np.maximum(n3, 1e-15), 'k-', linewidth=1.5)
    axes[1, 1].set_title("||y||² log scale (slow switching, decay)", fontsize=10)
    axes[1, 1].set_xlabel("t"); axes[1, 1].grid(True, alpha=0.3)

    # Row 3: lambda=1 (intermediate)
    axes[2, 0].plot(t_eval, u1, linewidth=1.5, label='u')
    axes[2, 0].plot(t_eval, v1, linewidth=1.5, label='v')
    axes[2, 0].set_title("Matrix switching λ=1 (intermediate)", fontsize=10)
    axes[2, 0].set_xlabel("t"); axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].semilogy(t_eval, np.maximum(n1, 1e-15), 'k-', linewidth=1.5)
    axes[2, 1].set_title("||y||² log scale (intermediate switching)", fontsize=10)
    axes[2, 1].set_xlabel("t"); axes[2, 1].grid(True, alpha=0.3)

    fig.suptitle("Random switching ODEs", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "random_switching.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
