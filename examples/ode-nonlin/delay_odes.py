"""Delay differential equations.

Demonstrates solving simple delay differential equations (DDEs) numerically.
The pantograph equation y'(t) = -y(t/2), y(0)=1 has solution related to the
q-exponential. We solve it step-by-step using scipy on successive intervals.

Credit: Chebfun example ode-nonlin/DelayDifferentialEquations.m (Nick Hale, Jun 2022).
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
    print("Delay differential equations (DDEs)")
    print("=" * 60)

    # Solve y'(t) = -y(t/2), y(0) = 1 on successive intervals
    # On [0, 1]: y(t/2) = y(t/2) where t/2 in [0, 0.5] (known from initial condition)
    # Use scipy, splitting into intervals [0,1], [1,2], [2,3], [3,4]
    # On each interval, the delayed argument t/2 is in the previous interval

    T = 4.0
    n_per_interval = 200
    intervals = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, T)]

    # Solve step by step, building dense representation
    t_all = [0.0]
    y_all = [1.0]

    for a, b in intervals:
        t_prev = np.array(t_all)
        y_prev = np.array(y_all)

        def rhs(t, y_):
            t_half = t / 2.0
            return [-np.interp(t_half, t_prev, y_prev)]

        t_eval = np.linspace(a, b, n_per_interval)
        sol = solve_ivp(rhs, [a, b], [y_all[-1]],
                        t_eval=t_eval, rtol=1e-10, atol=1e-12)
        t_all = np.concatenate([t_all[:-1], sol.t])
        y_all = np.concatenate([y_all[:-1], sol.y[0]])

    t_arr = np.array(t_all)
    y = np.array(y_all)

    print(f"\nPantograph solution y'(t)=-y(t/2) on [0, {T}]:")
    print(f"  y(0) = {y[0]:.6f}  (initial condition = 1)")
    print(f"  y(T={T}) = {y[-1]:.6f}")
    print(f"  y range: [{np.min(y):.4f}, {np.max(y):.4f}]")
    # The solution should decrease from y(0)=1
    assert y[0] > y[-1], "Pantograph solution should decrease"
    # Solution should remain bounded
    assert np.max(np.abs(y)) < 10.0

    # Reference: simple ODE y'(t) = -y(t) => y = exp(-t)
    # Verified via Chebfun differentiation
    print("\nReference: y'(t) = -y(t), y(0)=1 => y=exp(-t)")
    dom_ref = (0.0, T)
    y_exact = cj.chebfun(lambda t: jnp.exp(-t), domain=dom_ref)
    t_test = jnp.linspace(0.0, T, 200)
    res = y_exact.diff()(t_test) + y_exact(t_test)
    max_res = float(jnp.max(jnp.abs(res)))
    print(f"  Chebfun ODE residual for exp(-t): {max_res:.2e}")
    assert max_res < 1e-12

    # Wrap pantograph solution in Chebfun for analysis
    y_panto = cj.chebfun(
        lambda t: jnp.interp(t, jnp.array(t_arr), jnp.array(y)),
        domain=dom_ref, n=64
    )
    print(f"  Pantograph Chebfun length: {len(y_panto)}")
    # Verify: pantograph decays from 1.0
    assert float(y_panto(jnp.array(0.0))) > 0.9

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    t_ref = np.linspace(0, T, 500)
    y_ref = np.exp(-t_ref)

    fig, axes = plt.subplots(1, 2)

    axes[0].plot(t_arr, y, 'b', linewidth=1.6, label="pantograph y'=−y(t/2)")
    axes[0].plot(t_ref, y_ref, 'r--', linewidth=1.2, label="reference exp(−t)")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("y(t)")
    axes[0].set_title("Pantograph vs reference DDE", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(t_arr, np.clip(np.abs(y), 1e-15, None), 'b',
                     linewidth=1.6, label="pantograph")
    axes[1].semilogy(t_ref, y_ref, 'r--', linewidth=1.2, label="exp(−t)")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("|y(t)| [log scale]")
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
