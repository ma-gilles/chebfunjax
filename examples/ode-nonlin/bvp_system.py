"""System of two nonlinear BVPs.

Solves the coupled system:
  u'' - sin(v) = 0,   v'' + cos(u) = 0,   u(±1) = v(±1) = 0.

Credit: Chebfun example ode-nonlin/BVPSystem.m (Birkisson & Driscoll, Sep 2010).
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
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Coupled nonlinear BVP system")
    print("=" * 60)

    dom = (-1.0, 1.0)

    # Solve each equation iteratively until convergence
    # u'' = sin(v),  v'' = -cos(u),  u(±1) = v(±1) = 0
    u = cj.chebfun(lambda x: jnp.zeros_like(x), domain=dom)
    v = cj.chebfun(lambda x: jnp.zeros_like(x), domain=dom)

    print("\nIterating system...")
    for iteration in range(10):
        # Solve for u with fixed v: u'' = sin(v), u(±1)=0
        rhs_u = cj.chebfun(lambda x: jnp.sin(v(x)), domain=dom)
        Nu = Chebop(lambda x, u_: u_.diff(2), domain=dom)
        Nu.lbc = 0.0; Nu.rbc = 0.0
        u_new = Nu.solve(rhs_u)

        # Solve for v with fixed u: v'' = -cos(u), v(±1)=0
        rhs_v = cj.chebfun(lambda x: -jnp.cos(u(x)), domain=dom)
        Nv = Chebop(lambda x, v_: v_.diff(2), domain=dom)
        Nv.lbc = 0.0; Nv.rbc = 0.0
        v_new = Nv.solve(rhs_v)

        x_test = jnp.linspace(-1.0, 1.0, 100)
        change = max(float(jnp.max(jnp.abs(u_new(x_test) - u(x_test)))),
                     float(jnp.max(jnp.abs(v_new(x_test) - v(x_test)))))
        u, v = u_new, v_new
        if change < 1e-12:
            print(f"  Converged at iteration {iteration+1}")
            break
        if iteration % 3 == 0:
            print(f"  iter {iteration+1}: change = {change:.2e}")

    # Verify BCs
    for fn, name in [(u, 'u'), (v, 'v')]:
        assert abs(float(fn(jnp.array(-1.0)))) < 1e-8, f"{name}(-1) ≠ 0"
        assert abs(float(fn(jnp.array(1.0)))) < 1e-8, f"{name}(1) ≠ 0"

    # Verify ODE residuals
    x_test = jnp.linspace(-0.99, 0.99, 200)
    res_u = u.diff(2)(x_test) - jnp.sin(v(x_test))
    res_v = v.diff(2)(x_test) + jnp.cos(u(x_test))
    max_u = float(jnp.max(jnp.abs(res_u)))
    max_v = float(jnp.max(jnp.abs(res_v)))
    print(f"\n  Max residual |u'' - sin(v)|: {max_u:.2e}")
    print(f"  Max residual |v'' + cos(u)|: {max_v:.2e}")
    assert max_u < 1e-6, f"u residual too large: {max_u}"
    assert max_v < 1e-6, f"v residual too large: {max_v}"

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(-1.0, 1.0, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_plot, u(x_plot), 'b', linewidth=1.8, label="u(x)")
    ax.plot(x_plot, v(x_plot), 'r', linewidth=1.8, label="v(x)")
    ax.set_xlabel("x"); ax.set_ylabel("solution")
    ax.set_title("u″ = sin(v), v″ = −cos(u), u(±1)=v(±1)=0", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "bvp_system.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
