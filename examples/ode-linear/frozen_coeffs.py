"""Frozen coefficients do not determine stability.

Demonstrates with a 2D linear ODE x' = A(t) x that even if
all frozen-coefficient matrices A(t0) are stable (eigenvalues with
negative real parts), the time-varying system may be unstable.

Credit: Chebfun example ode-linear/FrozenCoeffs.m (Nick Trefethen, Mar 2017).
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
    print("Frozen coefficients do not determine stability")
    print("=" * 60)

    # Example 1: Discrete case
    # A_even = [[0,2],[0,0]], A_odd = [[0,0],[2,0]]
    # Both have eigenvalues 0 (stable), but product A_odd @ A_even = [[4,0],[0,0]]
    A_even = np.array([[0.0, 2.0], [0.0, 0.0]])
    A_odd = np.array([[0.0, 0.0], [2.0, 0.0]])

    print("\nDiscrete case: both matrices nilpotent (eigenvalues = 0)")
    print(f"  eig(A_even) = {np.linalg.eigvals(A_even)}")
    print(f"  eig(A_odd)  = {np.linalg.eigvals(A_odd)}")
    product = A_odd @ A_even
    print(f"  A_odd @ A_even = {product}")
    print(f"  eig(product)   = {np.linalg.eigvals(product)}")
    assert np.all(np.abs(np.linalg.eigvals(A_even)) < 1e-10)
    assert np.all(np.abs(np.linalg.eigvals(A_odd)) < 1e-10)
    assert np.max(np.abs(np.linalg.eigvals(product))) > 1.0

    # Example 2: continuous ODE x' = A(t) x where A(t) switches periodically
    # A(t) = A1 if sin(omega*t) > 0 else A2
    # A1 has eigenvalues with Re < 0, but system is unstable
    A1 = np.array([[-1.0, 10.0], [0.0, -1.0]])
    A2 = np.array([[-1.0, 0.0], [10.0, -1.0]])

    omega = 4.0

    def rhs_switched(t, x):
        A = A1 if np.sin(omega * t) > 0 else A2
        return A @ x

    T_end = 20.0
    x0 = np.array([1.0, 0.0])
    sol = solve_ivp(rhs_switched, [0, T_end], x0,
                    t_eval=np.linspace(0, T_end, 2000),
                    rtol=1e-10, atol=1e-12)

    norm_x = np.sqrt(sol.y[0]**2 + sol.y[1]**2)
    print(f"\nContinuous case with switching A(t):")
    print(f"  eig(A1) = {np.linalg.eigvals(A1).real} (all negative)")
    print(f"  eig(A2) = {np.linalg.eigvals(A2).real} (all negative)")
    print(f"  ||x(0)||  = {norm_x[0]:.4f}")
    print(f"  ||x(T)||  = {norm_x[-1]:.4f}")
    assert np.all(np.real(np.linalg.eigvals(A1)) < 0)
    assert np.all(np.real(np.linalg.eigvals(A2)) < 0)
    # The norm may grow (the system can be unstable despite frozen-coeff stability)
    print(f"  System is {'unstable' if norm_x[-1] > norm_x[0] else 'stable'}")

    # Also show a stable switched system for contrast
    A3 = np.array([[-2.0, 0.5], [-0.5, -2.0]])
    A4 = np.array([[-2.0, -0.5], [0.5, -2.0]])

    def rhs_stable(t, x):
        A = A3 if np.sin(omega * t) > 0 else A4
        return A @ x

    sol2 = solve_ivp(rhs_stable, [0, T_end], x0,
                     t_eval=np.linspace(0, T_end, 2000),
                     rtol=1e-10, atol=1e-12)
    norm_x2 = np.sqrt(sol2.y[0]**2 + sol2.y[1]**2)
    print(f"\nStable switched system: ||x(T)|| = {norm_x2[-1]:.4f}")

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    axes[0].semilogy(sol.t, norm_x, 'b', linewidth=1.4, label="switched (A1/A2)")
    axes[0].semilogy(sol2.t, norm_x2, 'r', linewidth=1.4, label="stable (A3/A4)")
    axes[0].set_title("Switched systems: norm of solution", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].plot(sol.y[0], sol.y[1], 'b', linewidth=0.8, alpha=0.8)
    axes[1].plot(sol.y[0, 0], sol.y[1, 0], color='#77AC30', marker='o', linestyle='none', markersize=6, label="start")
    axes[1].set_title("Phase portrait (unstable switched)", fontsize=10)
    axes[1].legend(fontsize=8)

    fig.suptitle("Frozen coefficients ≠ stability", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "frozen_coeffs.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
