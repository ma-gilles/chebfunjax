"""Blasius boundary-layer function.

Solves the Blasius ODE  2u''' + u u'' = 0  on a truncated domain [0, L]
with u(0)=u'(0)=0 and u'(L)=1. The solution is a smooth function
related to laminar boundary-layer flow.

Credit: Chebfun example ode-nonlin/Blasius.m (Hrothgar, Jun 2014).
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
    print("Blasius function: 2u''' + u u'' = 0, u(0)=u'(0)=0, u'(L)=1")
    print("=" * 60)

    L = 8.0  # truncated domain
    dom = (0.0, L)

    N = Chebop(lambda x, u: 2.0 * u.diff(3) + u * u.diff(2), domain=dom)
    N.lbc = [0.0, 0.0]    # u(0)=0, u'(0)=0
    N.rbc = 1.0            # u'(L)=1

    print(f"\nSolving on [0, {L}]...")
    u = N.solve(0.0)
    print(f"  Solution length: {len(u)}")

    # Blasius constant: u''(0) ≈ 0.33206
    blasius_const = float(u.diff(2)(jnp.array(0.0)))
    print(f"  u''(0) ≈ {blasius_const:.5f}  (exact: 0.33206)")
    assert abs(blasius_const - 0.33206) < 0.002

    # Verify BCs
    assert abs(float(u(jnp.array(0.0)))) < 1e-8
    assert abs(float(u.diff()(jnp.array(0.0)))) < 1e-8
    assert abs(float(u.diff()(jnp.array(L))) - 1.0) < 1e-6

    # Check ODE residual in interior
    x_test = jnp.linspace(0.5, L - 0.5, 200)
    res = 2.0 * u.diff(3)(x_test) + u(x_test) * u.diff(2)(x_test)
    max_res = float(jnp.max(jnp.abs(res)))
    print(f"  Max ODE residual: {max_res:.2e}")
    assert max_res < 1e-8

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(0.0, L, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x_plot, u(x_plot), 'b', linewidth=1.8, label="u(x)")
    axes[0].plot(x_plot, u.diff()(x_plot), 'r', linewidth=1.4, label="u'(x)")
    axes[0].set_xlabel("x"); axes[0].legend(fontsize=9)
    axes[0].set_title("Blasius function and its derivative", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_plot, u.diff(2)(x_plot), 'g', linewidth=1.8, label="u''(x)")
    axes[1].axhline(0, color='k', linewidth=0.5)
    axes[1].set_xlabel("x"); axes[1].legend(fontsize=9)
    axes[1].set_title(f"Second derivative (u''(0)≈{blasius_const:.5f})", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Blasius equation: 2u‴ + u u″ = 0", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "blasius.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
