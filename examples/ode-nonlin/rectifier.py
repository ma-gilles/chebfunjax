"""Half-wave rectifier ODE.

Solves the stiff IVP
  v' + ep*v = ep*(exp(alpha*(sin(t) - v)) - 1), v(0) = 0
which models a half-wave rectifier converting AC to DC current.

Credit: Chebfun example ode-nonlin/Rectifier.m (Toby Driscoll, May 2011).
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
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Half-wave rectifier ODE (stiff)")
    print("=" * 60)

    ep = 0.1
    alpha = 10.0
    T = 10.0 * 2 * np.pi

    print(f"\nParameters: ep={ep}, alpha={alpha}")

    def rhs(t, v):
        return [ep * (np.exp(np.clip(alpha * (np.sin(t) - v[0]), -500, 500)) - 1) - ep * v[0]]

    sol = solve_ivp(rhs, [0, T], [0.0], t_eval=np.linspace(0, T, 5000),
                    rtol=1e-8, atol=1e-10, method='Radau')

    v = sol.y[0]
    t = sol.t
    print(f"\nSolution: max(v) = {np.max(v):.4f},  mean(v) at end = {np.mean(v[-500:]):.4f}")
    assert np.max(v) > 0.1  # rectifier should produce positive output
    assert np.mean(v[-500:]) > 0.0  # DC component is positive

    # Use Chebop on a single period as BVP
    dom_period = (0.0, 2 * np.pi)
    print("\nChebop on single period [0, 2pi] as BVP:")
    N = Chebop(
        lambda t, v: v.diff() + ep * v - ep * (jnp.exp(
            jnp.clip(alpha * (jnp.sin(t) - v), -50.0, 50.0)) - 1.0),
        domain=dom_period
    )
    v_period_end = float(np.interp(2 * np.pi, t, v))
    N.lbc = float(np.interp(0.0, t, v))
    N.rbc = v_period_end
    v_cheb = N.solve(0.5)

    v_cheb_vals = np.array(v_cheb(jnp.linspace(0, 2*np.pi, 200, dtype=jnp.float64)))
    v_scipy = np.interp(np.linspace(0, 2*np.pi, 200), t, v)
    err = np.max(np.abs(v_cheb_vals - v_scipy))
    print(f"  Max error vs scipy: {err:.2e}")
    assert err < 0.05

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    n_periods = 5
    t_5 = np.linspace(0, n_periods * 2 * np.pi, 2500)
    v_5 = np.interp(t_5, t, v)
    axes[0].plot(t_5 / (2*np.pi), np.sin(t_5), 'b', linewidth=1.0, alpha=0.5, label="AC (sin t)")
    axes[0].plot(t_5 / (2*np.pi), v_5, 'r', linewidth=1.4, label="DC output v(t)")
    axes[0].set_xlabel("t / (2π)"); axes[0].set_ylabel("voltage")
    axes[0].set_title(f"Half-wave rectifier (ep={ep}, α={alpha})", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    t_single = np.linspace(0, 2*np.pi, 200)
    axes[1].plot(t_single, v_scipy, 'b', linewidth=1.6, label="scipy")
    axes[1].plot(t_single, v_cheb_vals, 'r--', linewidth=1.2, label="chebfunjax")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("v(t)")
    axes[1].set_title("Single period comparison", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Half-wave rectifier (stiff ODE)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "rectifier.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
