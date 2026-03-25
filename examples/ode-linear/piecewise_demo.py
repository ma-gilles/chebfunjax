"""Piecewise operators demo.

Demonstrates how piecewise-coefficient ODEs are handled in chebfunjax,
solving  -u'' + sign(x) exp(x) u = 0,  u(±1)=0.

Credit: Chebfun example ode-linear/PiecewiseDemo.m (Nick Hale & Toby Driscoll, Nov 2011).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Piecewise-coefficient ODE: -u'' + sign(x) exp(x) u = 0")
    print("=" * 60)

    dom = (-1.0, 1.0)

    # -u'' + sign(x) * exp(x) * u = 0, u(±1) = 0 + some non-trivial forcing
    # Add forcing: -u'' + sign(x)*exp(x)*u = 1, u(±1) = 0
    N = Chebop(
        lambda x, u: -u.diff(2) + jnp.sign(x) * jnp.exp(x) * u,
        domain=dom
    )
    N.lbc = 0.0
    N.rbc = 0.0
    u = N.solve(cj.chebfun(lambda x: jnp.ones_like(x), domain=dom))

    print(f"\nSolution length: {len(u)}")
    # Verify BCs
    assert abs(float(u(jnp.array(-1.0)))) < 1e-8
    assert abs(float(u(jnp.array(1.0)))) < 1e-8

    # Verify ODE residual away from x=0
    x_left = jnp.linspace(-0.9, -0.1, 100)
    x_right = jnp.linspace(0.1, 0.9, 100)
    for x_seg, sign_val in [(x_left, -1.0), (x_right, 1.0)]:
        res = (-u.diff(2)(x_seg) + sign_val * jnp.exp(x_seg) * u(x_seg) - 1.0)
        max_res = float(jnp.max(jnp.abs(res)))
        print(f"  Max ODE residual on {'left' if sign_val<0 else 'right'}: {max_res:.2e}")
        assert max_res < 1e-8

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
    assert max_res2 < 1e-10

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(-1.0, 1.0, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x_plot, u(x_plot), 'b', linewidth=1.8)
    axes[0].axvline(0, color='k', linestyle='--', linewidth=0.8)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("u(x)")
    axes[0].set_title("−u″ + sign(x) eˣ u = 1, u(±1)=0", fontsize=10)
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
