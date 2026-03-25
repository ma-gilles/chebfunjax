"""Lane-Emden equation (linear case n=0 and n=1).

The Lane-Emden equation is  x u'' + 2 u' + x u^n = 0,  u'(0)=0, u(0)=1.
For n=0 (linear): u = 1 - x^2/6 (exact).
For n=1 (linear in u): u = sin(x)/x.

Credit: Chebfun example ode-linear/LaneEmden.m (Alex Townsend, May 2011).
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
    print("Lane-Emden equation: x u'' + 2 u' + x u^n = 0")
    print("=" * 60)

    # Solve on [0, R] as BVP with u(0)=1, u'(0)=0
    # Rephrase: at x=0 we use u(0)=1 and symmetry u'(0)=0
    # Solve as BVP on [epsilon, R] with u(R)=exact(R)

    # n=0: u'' + (2/x) u' + 1 = 0 => exact u = 1 - x^2/6
    R = float(jnp.sqrt(6.0))  # first zero for n=0: u(sqrt(6)) = 0
    eps_x = 1e-4  # avoid x=0

    print("\nCase n=0 (exact: u = 1 - x^2/6)")
    exact_n0 = lambda x: 1.0 - x**2 / 6.0
    # BVP on [eps_x, R]: u(eps_x) = exact(eps_x), u(R) = 0
    N0 = Chebop(
        lambda x, u: x * u.diff(2) + 2.0 * u.diff() + x,
        domain=(eps_x, R)
    )
    N0.lbc = float(exact_n0(jnp.array(eps_x)))
    N0.rbc = 0.0
    u0 = N0.solve(0.0)
    x_test = jnp.linspace(eps_x, R - 0.01, 300)
    err0 = float(jnp.max(jnp.abs(u0(x_test) - exact_n0(x_test))))
    print(f"  Max error vs exact: {err0:.2e}")
    assert err0 < 1e-8

    # n=1: u'' + (2/x) u' + u = 0 => exact u = sin(x)/x
    R1 = float(jnp.pi)  # first zero for n=1: u(pi) = 0
    print("\nCase n=1 (exact: u = sin(x)/x)")
    exact_n1 = lambda x: jnp.sin(x) / x
    N1 = Chebop(
        lambda x, u: x * u.diff(2) + 2.0 * u.diff() + x * u,
        domain=(eps_x, R1)
    )
    N1.lbc = float(exact_n1(jnp.array(eps_x)))
    N1.rbc = 0.0
    u1 = N1.solve(0.0)
    x_test1 = jnp.linspace(eps_x, R1 - 0.01, 300)
    err1 = float(jnp.max(jnp.abs(u1(x_test1) - exact_n1(x_test1))))
    print(f"  Max error vs exact: {err1:.2e}")
    assert err1 < 1e-8

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x_plot0 = jnp.linspace(eps_x, R, 300)
    axes[0].plot(x_plot0, u0(x_plot0), 'b', linewidth=1.8, label="chebfunjax")
    axes[0].plot(x_plot0, exact_n0(x_plot0), 'r--', linewidth=1.2, label="exact 1−x²/6")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("u(x)")
    axes[0].set_title("Lane-Emden n=0: x u″+2u′+x = 0", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    x_plot1 = jnp.linspace(eps_x, R1, 300)
    axes[1].plot(x_plot1, u1(x_plot1), 'b', linewidth=1.8, label="chebfunjax")
    axes[1].plot(x_plot1, exact_n1(x_plot1), 'r--', linewidth=1.2, label="exact sin(x)/x")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("u(x)")
    axes[1].set_title("Lane-Emden n=1: x u″+2u′+xu = 0", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Lane-Emden equation (linear cases n=0,1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "lane_emden_linear.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
