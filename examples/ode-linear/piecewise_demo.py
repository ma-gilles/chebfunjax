"""Piecewise operators demo.

Demonstrates how piecewise-coefficient ODEs are handled in chebfunjax,
solving  -u'' + sign(x) exp(x) u = 1,  u(±1)=0.

The piecewise coefficient sign(x) is handled by solving the ODE on
two half-intervals [-1,0] and [0,1] and matching at x=0.

Credit: Chebfun example ode-linear/PiecewiseDemo.m (Nick Hale & Toby Driscoll, Nov 2011).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Piecewise-coefficient ODE: -u'' + sign(x) exp(x) u = 1")
    print("=" * 60)

    dom = (-1.0, 1.0)

    # sign(x) is -1 on [-1,0] and +1 on [0,1].
    # Solve on two half-intervals and match (continuous u, continuous u').
    # Left:  -u'' + (-1)*exp(x)*u = 1, u(-1)=0, u(0)=c
    # Right: -u'' + (+1)*exp(x)*u = 1, u(0)=c,  u(1)=0
    # The value c is chosen so that u' is also continuous at x=0.

    def solve_halves(c):
        # RHS = 1.0 (constant forcing), passed as argument to Chebop.solve()
        Nl = Chebop(lambda x, u: -u.diff(2) - cj.exp(x) * u, domain=(-1.0, 0.0))
        Nl.lbc = 0.0
        Nl.rbc = float(c)
        ul = Nl.solve(1.0)  # forcing = 1
        Nr = Chebop(lambda x, u: -u.diff(2) + cj.exp(x) * u, domain=(0.0, 1.0))
        Nr.lbc = float(c)
        Nr.rbc = 0.0
        ur = Nr.solve(1.0)  # forcing = 1
        return ul, ur

    def deriv_jump(c):
        ul, ur = solve_halves(c)
        return float(ur.diff()(jnp.array(0.0))) - float(ul.diff()(jnp.array(0.0)))

    print("\nFinding matching value c = u(0) for derivative continuity...")
    c_opt = brentq(deriv_jump, -5.0, 5.0)
    print(f"  c = u(0) = {c_opt:.8f}")
    ul, ur = solve_halves(c_opt)
    print(f"  u_left length: {len(ul)},  u_right length: {len(ur)}")

    # Verify BCs
    assert abs(float(ul(jnp.array(-1.0)))) < 1e-8
    assert abs(float(ur(jnp.array(1.0)))) < 1e-8

    # Verify ODE residual on left
    x_left = jnp.linspace(-0.9, -0.1, 100)
    res_l = (-ul.diff(2)(x_left) - jnp.exp(x_left) * ul(x_left) - 1.0)
    max_res_l = float(jnp.max(jnp.abs(res_l)))
    print(f"  Max ODE residual (left): {max_res_l:.2e}")
    assert max_res_l < 1e-8

    # Verify ODE residual on right
    x_right = jnp.linspace(0.1, 0.9, 100)
    res_r = (-ur.diff(2)(x_right) + jnp.exp(x_right) * ur(x_right) - 1.0)
    max_res_r = float(jnp.max(jnp.abs(res_r)))
    print(f"  Max ODE residual (right): {max_res_r:.2e}")
    assert max_res_r < 1e-8

    # Verify derivative continuity at x=0
    du_left = float(ul.diff()(jnp.array(0.0)))
    du_right = float(ur.diff()(jnp.array(0.0)))
    print(f"  u'(0-) = {du_left:.8f},  u'(0+) = {du_right:.8f}")
    assert abs(du_left - du_right) < 1e-6

    # Also demonstrate u'' - u = x as simple smooth case
    print("\nSmooth reference: u'' - u = x, u(±1) = 0")
    N2 = Chebop(lambda x, u: u.diff(2) - u, domain=dom)
    N2.lbc = 0.0
    N2.rbc = 0.0
    x_rhs = cj.chebfun(lambda x: x, domain=dom)
    u2 = N2.solve(x_rhs)
    x_test = jnp.linspace(-1.0, 1.0, 300)
    res2 = u2.diff(2)(x_test) - u2(x_test) - x_test
    max_res2 = float(jnp.max(jnp.abs(res2)))
    print(f"  Max ODE residual: {max_res2:.2e}")
    assert max_res2 < 1e-8

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot_l = np.linspace(-1.0, 0.0, 200)
    x_plot_r = np.linspace(0.0, 1.0, 200)
    x_plot = jnp.linspace(-1.0, 1.0, 400)
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(x_plot_l, np.array(ul(jnp.array(x_plot_l))), 'b', linewidth=1.8)
    axes[0].plot(x_plot_r, np.array(ur(jnp.array(x_plot_r))), 'b', linewidth=1.8,
                 label="piecewise solution")
    axes[0].axvline(0, color='k', linestyle='--', linewidth=0.8, label="x=0")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("u(x)")
    axes[0].set_title("−u″ + sign(x) eˣ u = 1, u(±1)=0", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_plot, u2(x_plot), 'r', linewidth=1.8)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("u(x)")
    axes[1].set_title("u″ − u = x, u(±1)=0  (smooth reference)", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Piecewise-coefficient ODE", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "piecewise_demo.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
