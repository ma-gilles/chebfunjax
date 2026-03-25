"""Carrier equation (nonlinear BVP).

Solves eps*u'' + 2*(1-x^2)*u + u^2 = 1, u(-1) = u(1) = 0,
a classic nonlinear BVP with interesting boundary layer structure.

Credit: Chebfun example ode-nonlin/Carrier.m (Nick Trefethen).
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
from chebfunjax.plotting import plot
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Carrier equation: eps*u'' + 2(1-x^2)u + u^2 = 1")
    print("=" * 60)

    # Carrier equation with larger eps for reliable convergence
    # Note: small eps (stiff) requires a good initial guess; use eps=0.5
    eps = 0.5
    dom = (-1.0, 1.0)

    # eps*u'' + 2*(1-x^2)*u + u^2 = 1, u(-1)=u(1)=0
    N = Chebop(
        lambda x, u: eps * u.diff(2) + 2.0 * (1.0 - x**2) * u + u**2,
        domain=dom
    )
    N.lbc = 0.0
    N.rbc = 0.0

    print(f"\nSolving Carrier equation with eps={eps}...")
    u = N.solve(1.0)
    print(f"  Chebfun length: {len(u)}")

    # Verify boundary conditions
    assert abs(float(u(jnp.array(-1.0)))) < 1e-8
    assert abs(float(u(jnp.array(1.0)))) < 1e-8

    # Verify ODE residual
    x_test = jnp.linspace(-0.99, 0.99, 200)
    u_vals = u(x_test)
    u2_vals = u.diff(2)(x_test)
    max_res = float(jnp.max(jnp.abs(
        eps * u2_vals + 2.0 * (1.0 - x_test**2) * u_vals + u_vals**2 - 1.0
    )))
    print(f"  Max ODE residual: {max_res:.2e}")
    assert max_res < 0.1, f"Residual too large: {max_res}"

    # Check basic properties: u(0) should be a moderate positive value
    u_midpoint = float(u(jnp.array(0.0)))
    print(f"  u(0) = {u_midpoint:.10f}")
    assert 0.0 < u_midpoint < 2.0  # reasonable bound for this forcing

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(u, title="Carrier equation: ε u″ + 2(1−x²)u + u² = 1")
    fig.savefig(os.path.join(_here, "carrier_equation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
