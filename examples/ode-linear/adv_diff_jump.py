"""Advection-diffusion equation with a jump in the advection coefficient.

Solves  0.2 u'' + c(x) u' = -1,  u(-10) = u(10) = 0,
first with constant c=1 (uniform advection), then with c(x)=(x>=0)
(advection only on the right half). Demonstrates piecewise operators.

Credit: Chebfun example ode-linear/AdvDiffJump.m (Nick Trefethen, Nov 2010).
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
    print("Advection-diffusion with jump in advection")
    print("=" * 60)

    dom = (-10.0, 10.0)

    # Case 1: uniform advection  0.2 u'' + u' = -1
    print("\nCase 1: 0.2 u'' + u' = -1, u(±10) = 0")
    N1 = Chebop(lambda x, u: 0.2 * u.diff(2) + u.diff(), domain=dom)
    N1.lbc = 0.0
    N1.rbc = 0.0
    u1 = N1.solve(-1.0)
    print(f"  Solution length: {len(u1)}")
    # Boundary conditions
    assert abs(float(u1(jnp.array(-10.0)))) < 1e-8
    assert abs(float(u1(jnp.array(10.0)))) < 1e-8
    # Should have a boundary layer near x=-10
    max_val = float(jnp.max(u1(jnp.linspace(-10, 10, 500))))
    print(f"  max(u) ≈ {max_val:.4f}")
    assert max_val > 1.0  # solution rises well above 0

    # Case 2: advection only on right half  0.2 u'' + H(x) u' = -1
    print("\nCase 2: 0.2 u'' + H(x)·u' = -1  (advection only for x≥0)")
    N2 = Chebop(
        lambda x, u: 0.2 * u.diff(2) + jnp.where(x >= 0, 1.0, 0.0) * u.diff(),
        domain=dom,
    )
    N2.lbc = 0.0
    N2.rbc = 0.0
    u2 = N2.solve(-1.0)
    print(f"  Solution length: {len(u2)}")
    assert abs(float(u2(jnp.array(-10.0)))) < 1e-8
    assert abs(float(u2(jnp.array(10.0)))) < 1e-8
    max_val2 = float(jnp.max(u2(jnp.linspace(-10, 10, 500))))
    print(f"  max(u) ≈ {max_val2:.4f}")

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(-10.0, 10.0, 600)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axes[0].plot(x_plot, u1(x_plot), 'b', linewidth=1.8)
    axes[0].set_title("0.2 u″ + u′ = −1", fontsize=10)
    axes[0].set_xlabel("x"); axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-10, 10)

    axes[1].plot(x_plot, u2(x_plot), 'r', linewidth=1.8)
    axes[1].set_title("0.2 u″ + H(x)·u′ = −1", fontsize=10)
    axes[1].set_xlabel("x"); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-10, 10)

    fig.suptitle("Advection-diffusion with jump in advection", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "adv_diff_jump.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
