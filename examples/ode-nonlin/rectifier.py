"""Half-wave rectifier ODE.

Solves the stiff IVP
  v' + ep*v = ep*(exp(alpha*(sin(t) - v)) - 1), v(0) = 0
which models a half-wave rectifier converting AC to DC current.

Credit: Chebfun example ode-nonlin/Rectifier.m (Toby Driscoll, May 2011).
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

    # Wrap the scipy solution in a Chebfun for smooth evaluation
    # Use fixed n to avoid slow adaptive convergence on interpolated data
    dom_last = (float(T - 2*np.pi), float(T))
    mask = (t >= T - 2*np.pi)
    t_last, v_last = t[mask], v[mask]
    v_cheb = cj.chebfun(
        lambda tt: jnp.interp(tt, jnp.array(t_last), jnp.array(v_last)),
        domain=dom_last, n=64
    )
    print(f"\nChebfun of last period (n={len(v_cheb)}):")

    # Verify ODE residual via Chebfun differentiation on last period
    t_inner = jnp.linspace(float(T - 2*np.pi) + 0.2, float(T) - 0.2, 100)
    v_vals = v_cheb(t_inner)
    exp_arg = jnp.clip(alpha * (jnp.sin(t_inner) - v_vals), -500, 500)
    res = v_cheb.diff()(t_inner) + ep * v_vals - ep * (jnp.exp(exp_arg) - 1.0)
    max_res = float(jnp.max(jnp.abs(res)))
    print(f"  ODE residual (last period interior): {max_res:.2e}")
    assert max_res < 0.1  # jnp.interp accuracy limits residual

    # DC component: mean of v in steady state
    dc = float(v_cheb.sum() / (2 * np.pi))
    print(f"  DC component (mean of last period): {dc:.4f}")
    assert dc > 0.0

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    n_periods = 5
    t_5 = np.linspace(0, n_periods * 2 * np.pi, 2500)
    v_5 = np.interp(t_5, t, v)
    axes[0].plot(t_5 / (2*np.pi), np.sin(t_5), 'b', linewidth=1.0, alpha=0.5, label="AC (sin t)")
    axes[0].plot(t_5 / (2*np.pi), v_5, 'r', linewidth=1.4, label="DC output v(t)")
    axes[0].set_title(f"Half-wave rectifier (ep={ep}, α={alpha})", fontsize=10)
    axes[0].legend(fontsize=8)

    t_single = jnp.linspace(float(T - 2*np.pi), float(T), 200)
    axes[1].plot(t_single - float(T - 2*np.pi), v_cheb(t_single), 'b', linewidth=1.6, label="Chebfun")
    axes[1].plot(t_last - t_last[0], v_last, 'r--', linewidth=1.0, label="scipy", alpha=0.7)
    axes[1].set_title("Last period: scipy vs Chebfun", fontsize=10)
    axes[1].legend(fontsize=8)

    fig.suptitle("Half-wave rectifier (stiff ODE)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "rectifier.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
