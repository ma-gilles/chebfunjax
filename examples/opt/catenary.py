"""Catenary: shape of a hanging chain.

Verifies properties of the catenary y = a*cosh(x/a):
- ODE: y'' = y/a (the catenary equation)
- Arc length formula: s = 2*a*sinh(L/a)
- Minimum height at x = 0 equals a

Credit: Inspired by Chebfun opt/Catenary.m.
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

from chebfunjax.plotting import plot

def run():
    print("=" * 60)
    print("Catenary: shape of a hanging chain")
    print("=" * 60)

    a = 2.0
    dom = (-3.0, 3.0)
    y = cj.chebfun(lambda x: a * jnp.cosh(x / a), domain=dom)
    yp = y.diff()     # y'
    ypp = y.diff(2)   # y''

    # Check ODE: y'' = (1/a)*cosh(x/a) = y/a^2
    # (y = a*cosh(x/a), y'' = (1/a)*cosh(x/a) = y/a^2)
    x_test = jnp.linspace(-3.0, 3.0, 200)
    ode_residual = ypp(x_test) - (1.0/a**2) * y(x_test)
    max_ode_err = float(jnp.max(jnp.abs(ode_residual)))
    print(f"\ny = {a}*cosh(x/{a}):")
    print(f"  ||y'' - y/{a}^2||_inf = {max_ode_err:.2e}")
    assert max_ode_err < 1e-12

    # Arc length: s = 2*a*sinh(L/a)
    L = 3.0
    arc_integrand = cj.chebfun(lambda x: jnp.sqrt(1.0 + yp(x)**2), domain=dom)
    arc_length = float(arc_integrand.sum())
    arc_exact = 2.0 * a * float(jnp.sinh(jnp.array(L / a)))
    print(f"  Arc length from -{L} to {L}: {arc_length:.12f}")
    print(f"  Exact: 2a*sinh({L}/{a}) = {arc_exact:.12f}")
    print(f"  Error: {abs(arc_length - arc_exact):.2e}")
    assert abs(arc_length - arc_exact) < 1e-10

    # Minimum height at x = 0 is a
    y0 = float(y(jnp.array(0.0)))
    print(f"\n  y(0) = {y0:.12f}  (exact: a = {a})")
    assert abs(y0 - a) < 1e-12

    # Center of mass height
    cm_numerator = cj.chebfun(
        lambda x: a * jnp.cosh(x / a) * jnp.sqrt(1.0 + jnp.sinh(x / a)**2),
        domain=dom
    )
    cm_height = float(cm_numerator.sum()) / arc_length
    print(f"\n  Center of mass height (catenary): {cm_height:.12f}")
    # Center of mass should be above the minimum (y0 = a)
    assert cm_height > a

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(y, title="Catenary y = a·cosh(x/a)", ylabel="y")
    fig.savefig(os.path.join(_here, "catenary.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
