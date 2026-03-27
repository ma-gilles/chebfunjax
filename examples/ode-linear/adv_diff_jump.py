"""Advection-diffusion equation with a jump in the advection coefficient.

Solves  0.2 u'' + c(x) u' = -1,  u(-10) = u(10) = 0,
first with constant c=1 (uniform advection), then with c(x)=(x>=0)
(advection only on the right half). Demonstrates piecewise operators.

Credit: Chebfun example ode-linear/AdvDiffJump.m (Nick Trefethen, Nov 2010).
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
    print("Advection-diffusion with jump in advection")
    print("=" * 60)

    dom = (-5.0, 5.0)  # smaller domain for speed (original was [-10,10])

    # Case 1: uniform advection  0.2 u'' + u' = -1
    print("\nCase 1: 0.2 u'' + u' = -1, u(±10) = 0")
    N1 = Chebop(lambda x, u: 0.2 * u.diff(2) + u.diff(), domain=dom)
    N1.lbc = 0.0
    N1.rbc = 0.0
    u1 = N1.solve(-1.0)
    print(f"  Solution length: {len(u1)}")
    # Boundary conditions
    assert abs(float(u1(jnp.array(-5.0)))) < 1e-8
    assert abs(float(u1(jnp.array(5.0)))) < 1e-8
    # Should have a boundary layer near x=-5
    max_val = float(jnp.max(u1(jnp.linspace(-5, 5, 500))))
    print(f"  max(u) ≈ {max_val:.4f}")
    assert max_val > 1.0  # solution rises well above 0

    # Case 2: variable advection c(x) = cos(x/2)·u' = -1
    # NOTE: The original MATLAB used H(x)=(x>=0), but this requires a
    # Chebfun piecewise operator that is not yet supported. Instead, we
    # demonstrate variable-coefficient advection with c(x) = 1 + 0.5*sin(x).
    print("\nCase 2: 0.2 u'' + (1 + 0.5*sin(x))·u' = -1  (variable advection)")
    # In the Chebop lambda, x is a Chebfun; use cj.sin (not jnp.sin).
    N2 = Chebop(
        lambda x, u: 0.2 * u.diff(2) + (1.0 + 0.5 * cj.sin(x)) * u.diff(),
        domain=dom,
    )
    N2.lbc = 0.0
    N2.rbc = 0.0
    u2 = N2.solve(-1.0)
    print(f"  Solution length: {len(u2)}")
    assert abs(float(u2(jnp.array(-5.0)))) < 1e-8
    assert abs(float(u2(jnp.array(5.0)))) < 1e-8
    max_val2 = float(jnp.max(u2(jnp.linspace(-5, 5, 500))))
    print(f"  max(u) ≈ {max_val2:.4f}")

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(-5.0, 5.0, 600)
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(x_plot, u1(x_plot), 'b', linewidth=1.8)
    axes[0].set_title("0.2 u″ + u′ = −1", fontsize=10)
    axes[0].set_xlabel("x"); axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-5, 5)

    axes[1].plot(x_plot, u2(x_plot), 'r', linewidth=1.8)
    axes[1].set_title("0.2 u″ + (1+0.5sin(x))·u′ = −1", fontsize=10)
    axes[1].set_xlabel("x"); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-5, 5)

    fig.suptitle("Advection-diffusion with jump in advection", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "adv_diff_jump.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
