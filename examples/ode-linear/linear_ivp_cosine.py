"""Linear IVP: u'' + u = 0 giving cos(x).

Demonstrates solving the harmonic oscillator IVP:
    u'' + u = 0, u(0) = 1, u'(0) = 0
whose solution is cos(x). Tests on a large interval [0, 100].

Credit: Inspired by Chebfun ode-linear examples.
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
from chebfunjax.plotting import plot
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("IVP: u'' + u = 0, solution cos(x)")
    print("=" * 60)

    # --- Solve on [0, 10*pi] -----------------------------------------
    d = (0.0, float(10.0 * jnp.pi))
    N = Chebop(lambda x, u: u.diff(2) + u, domain=d)
    # IVP: u(0) = 1, u'(0) = 0
    N.lbc = [1.0, 0.0]  # [value, derivative] at left endpoint

    u = N.solve(0.0)
    print(f"\nu'' + u = 0 on [0, 10*pi], u(0)=1, u'(0)=0:")
    print(f"  Chebfun length: {len(u)}")

    # Compare with exact solution cos(x)
    dom = d
    cos_cheb = cj.chebfun(lambda x: jnp.cos(x), domain=dom)
    x_test = jnp.linspace(0.0, float(10.0 * jnp.pi), 500)
    err = float(jnp.max(jnp.abs(u(x_test) - jnp.cos(x_test))))
    print(f"  Max error vs cos(x): {err:.2e}")
    assert err < 1e-8, f"Error too large: {err}"

    # Verify initial conditions
    u0 = float(u(jnp.array(0.0)))
    du = u.diff()
    du0 = float(du(jnp.array(0.0)))
    print(f"  u(0)  = {u0:.15f}  (exact: 1.0)")
    print(f"  u'(0) = {du0:.15f}  (exact: 0.0)")
    assert abs(u0 - 1.0) < 1e-12
    assert abs(du0) < 1e-11

    # Verify ODE residual: u'' + u should be ~0
    d2u = u.diff(2)
    residual = d2u + u
    x_check = jnp.linspace(0.1, float(10.0 * jnp.pi) - 0.1, 100)
    res_err = float(jnp.max(jnp.abs(residual(x_check))))
    print(f"  ||u'' + u||_inf = {res_err:.2e}")
    assert res_err < 1e-8

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(u, title="u′ − u = 0, u(0)=1 (solution = exp(x))",
                   label="Chebfun")
    plot(cos_cheb, ax=ax, color="#E04040", linestyle="--", label="cos(x)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "linear_ivp_cosine.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
