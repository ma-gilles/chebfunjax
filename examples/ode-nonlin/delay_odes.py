"""Delay differential equations.

Demonstrates solving simple delay differential equations (DDEs) numerically.
Includes the pantograph equation y'(t) = -y(t/2), y(0)=1 whose solution
is related to the q-exponential.

Credit: Chebfun example ode-nonlin/DelayDifferentialEquations.m (Nick Hale, Jun 2022).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj


def run():
    print("=" * 60)
    print("Delay differential equations (DDEs)")
    print("=" * 60)

    # Pantograph equation: y'(t) = -y(t/2), y(0) = 1
    # Exact solution: y(t) = prod_{k=0}^{inf} exp(-t/2^k) = exp(-2t + ...) ... complicated
    # For comparison: y(t) ~ exp(-t) is NOT the solution but a reference

    # Solve numerically using dense output from solve_ivp
    # y(t) for t in [0, T], y'(t) = -y(t/2)
    # Use step-by-step integration
    T = 4.0
    dt = 1e-4
    t_arr = np.arange(0.0, T + dt, dt)
    y = np.ones(len(t_arr))

    def y_interp(t_val, t_arr, y_arr):
        """Linear interpolation for y at time t_val."""
        if t_val <= t_arr[0]:
            return 1.0  # initial condition
        idx = np.searchsorted(t_arr, t_val) - 1
        idx = min(idx, len(t_arr) - 2)
        alpha = (t_val - t_arr[idx]) / (t_arr[idx+1] - t_arr[idx])
        return y_arr[idx] + alpha * (y_arr[idx+1] - y_arr[idx])

    print("\nSolving pantograph equation y'(t) = -y(t/2), y(0)=1...")
    for i in range(1, len(t_arr)):
        t_half = t_arr[i] / 2.0
        y_half = y_interp(t_half, t_arr[:i], y[:i])
        y[i] = y[i-1] - dt * y_half

    print(f"  y(T={T}) ≈ {y[-1]:.6f}")
    assert 0 < y[-1] < 1.0  # should decrease from 1

    # Compare with y'(t) = -y(t) which is solvable: y = exp(-t)
    print("\nReference: simple DDE y'(t) = -y(t), y(0)=1 => y=exp(-t)")
    t_ref = np.linspace(0, T, 500)
    y_ref = np.exp(-t_ref)

    # Use Chebop to solve the reference (no delay)
    from chebfunjax.operators.chebop import Chebop
    dom_ref = (0.0, T)
    N = Chebop(lambda t, y_: y_.diff() + y_, domain=dom_ref)
    N.lbc = 1.0
    N.rbc = float(np.exp(-T))
    y_cheb = N.solve(0.0)
    err = float(jnp.max(jnp.abs(y_cheb(jnp.array(t_ref, dtype=jnp.float64)) -
                                  jnp.array(y_ref))))
    print(f"  Chebop max error vs exp(-t): {err:.2e}")
    assert err < 1e-10

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(t_arr, y, 'b', linewidth=1.6, label="pantograph y'=−y(t/2)")
    axes[0].plot(t_ref, y_ref, 'r--', linewidth=1.2, label="reference exp(−t)")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("y(t)")
    axes[0].set_title("Pantograph vs reference DDE", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(t_arr, y, 'b', linewidth=1.6, label="pantograph")
    axes[1].semilogy(t_ref, y_ref, 'r--', linewidth=1.2, label="exp(−t)")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("y(t) [log scale]")
    axes[1].set_title("Log-scale comparison", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Delay differential equations", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "delay_odes.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
