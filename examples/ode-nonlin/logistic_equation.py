"""Logistic equation and variants.

Solves the logistic ODE: u' = r*u*(1 - u/K) with exact solutions,
and a slight modification to demonstrate Chebfun's nonlinear solver.

Credit: Inspired by Chebfun ode-nonlin/Logistic.m.
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Logistic equation")
    print("=" * 60)

    # --- Standard logistic: u' = u*(1-u), u(0) = u0 ------------------
    # Exact: u(t) = u0 / (u0 + (1-u0)*exp(-t))
    r = 1.0  # growth rate
    K = 1.0  # carrying capacity
    u0 = 0.1  # initial population fraction

    dom = (0.0, 5.0)
    exact = lambda t: u0 / (u0 + (1.0 - u0) * jnp.exp(-t))

    # Solve as BVP: u(0) = u0, u(5) = exact(5)
    N = Chebop(lambda t, u: u.diff() - r * u * (1.0 - u / K), domain=dom)
    N.lbc = u0
    N.rbc = float(exact(jnp.array(5.0)))

    u = N.solve(0.0)
    print(f"\nLogistic ODE u' = u*(1-u) on [0, 5]:")
    print(f"  u(0) = {float(u(jnp.array(0.0))):.12f}  (exact: {u0})")
    print(f"  Chebfun length: {len(u)}")

    t_test = jnp.linspace(0.0, 5.0, 300)
    err = float(jnp.max(jnp.abs(u(t_test) - exact(t_test))))
    print(f"  Max error vs exact: {err:.2e}")
    assert err < 1e-6, f"Logistic error too large: {err}"

    # Verify that u approaches carrying capacity K
    u_final = float(u(jnp.array(5.0)))
    print(f"  u(5) = {u_final:.12f}  (K = {K})")

    # --- Second logistic: u' = 2*u*(1 - u/2), u(0) = 0.5 ------------
    # Exact: u(t) = 2 / (1 + 3*exp(-2*t)) * 0.5 ... actually
    # u' = r*u*(1 - u/K) with r=2, K=2: exact u(t) = K*u0 / (u0 + (K-u0)*exp(-r*t))
    r2, K2, u0_2 = 2.0, 2.0, 0.5
    dom2 = (0.0, 3.0)
    exact2 = lambda t: K2 * u0_2 / (u0_2 + (K2 - u0_2) * jnp.exp(-r2 * t))
    N2 = Chebop(lambda t, u: u.diff() - r2 * u * (1.0 - u / K2), domain=dom2)
    N2.lbc = u0_2
    N2.rbc = float(exact2(jnp.array(3.0)))
    u2 = N2.solve(0.0)
    t_test2 = jnp.linspace(0.0, 3.0, 300)
    err2 = float(jnp.max(jnp.abs(u2(t_test2) - exact2(t_test2))))
    print(f"\nLogistic (r=2, K=2) on [0, 3]:")
    print(f"  Max error vs exact: {err2:.2e}")
    assert err2 < 1e-5, f"Logistic (r=2,K=2) error too large: {err2}"

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
