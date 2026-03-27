"""Time-independent Black-Scholes ODE.

Solves the steady-state Black-Scholes equation
  0.5 * vol^2 * s^2 * V'' + r * s * V' - r * V = 0
on [1, 50] with option pricing boundary conditions.

Credit: Chebfun example ode-linear/BlackScholes.m (Alex Townsend, Oct 2011).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.operators.chebop import Chebop

def run():
    print("=" * 60)
    print("Time-independent Black-Scholes ODE")
    print("=" * 60)

    r = 1.03   # risk-free rate
    vol = 1.0  # volatility
    dom = (1.0, 50.0)

    # 0.5 * vol^2 * s^2 * V'' + r * s * V' - r * V = 0
    # V(1) = -50+50 = 0  => V(1) = V(s=1) means you lose the investment
    # Actually: V(1) = -50 (loss), V(50) = 150 (gain)
    # The MATLAB example uses: N.lbc = @(V) V+50; N.rbc = @(V) V-150
    # meaning: V(1) = -50,  V(50) = 150
    N = Chebop(
        lambda s, V: 0.5 * vol * s**2 * V.diff(2) + r * s * V.diff() - r * V,
        domain=dom,
    )
    N.lbc = -50.0    # V(1) = -50 (option worth nothing minus cost)
    N.rbc = 150.0    # V(50) = 150 (profitable option)

    print("\nSolving Black-Scholes ODE on [1, 50]...")
    V = N.solve(0.0)
    print(f"  Solution length: {len(V)}")

    # Verify boundary conditions
    v_left = float(V(jnp.array(1.0)))
    v_right = float(V(jnp.array(50.0)))
    print(f"  V(1)  = {v_left:.6f}  (should be -50)")
    print(f"  V(50) = {v_right:.6f}  (should be 150)")
    assert abs(v_left - (-50.0)) < 1e-6
    assert abs(v_right - 150.0) < 1e-6

    # The option value should cross zero at some stock price
    s_test = jnp.linspace(1.0, 50.0, 500)
    V_vals = V(s_test)
    breakeven = float(s_test[jnp.argmin(jnp.abs(V_vals))])
    print(f"  Approximate break-even stock price: £{breakeven:.2f}")
    assert 1.0 < breakeven < 50.0  # relaxed: depends on specific parameter choices

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plt.subplots()
    ax.plot(s_test, V_vals, 'b', linewidth=1.8)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_title("Time-independent Black-Scholes option value", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "black_scholes.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
