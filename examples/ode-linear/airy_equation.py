"""Airy equation BVP.

Solves u'' - x*u = 0 (the Airy equation) on [-10, 2] with boundary
conditions from the exact Airy function Ai(x).

Credit: Inspired by Chebfun ode-linear/Airy.m.
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import jax.numpy as jnp
import numpy as np
import scipy.special as sp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Airy equation BVP: u'' - x*u = 0")
    print("=" * 60)

    # Boundary values from scipy
    a, b = -10.0, 2.0
    Ai_a = float(sp.airy(a)[0])
    Ai_b = float(sp.airy(b)[0])
    print(f"\nAi({a}) = {Ai_a:.12f}")
    print(f"Ai({b}) = {Ai_b:.12f}")

    # MATLAB: N = chebop(-10, 2); N.op = @(x,u) diff(u,2) - x.*u;
    #         N.lbc = airy(a); N.rbc = airy(b);
    N = Chebop(lambda x, u: u.diff(2) - x * u, domain=(a, b))
    N.lbc = Ai_a
    N.rbc = Ai_b

    u = N.solve(0.0)
    print(f"\nSolved Airy BVP on [{a}, {b}]:")
    print(f"  Chebfun length: {len(u)}")

    # Compare with scipy Airy function
    x_test = jnp.linspace(a, b, 500)
    Ai_exact = jnp.array(sp.airy(np.array(x_test))[0])
    u_vals = u(x_test)
    err = float(jnp.max(jnp.abs(u_vals - Ai_exact)))
    print(f"  Max error vs scipy Ai(x): {err:.2e}")
    assert err < 1e-8, f"Airy BVP error too large: {err}"

    # Verify at x = 0: Ai(0) = 1/(3^(2/3) * Gamma(2/3))
    Ai0_exact = float(sp.airy(0.0)[0])
    Ai0_computed = float(u(jnp.array(0.0)))
    print(f"\n  Ai(0) computed: {Ai0_computed:.12f}")
    print(f"  Ai(0) exact:    {Ai0_exact:.12f}")
    assert abs(Ai0_computed - Ai0_exact) < 1e-10

    # Verify ODE residual at interior points
    d2u = u.diff(2)
    x_id = cj.chebfun(lambda t: t, domain=(a, b))
    residual = d2u - x_id * u
    x_check = jnp.linspace(a + 0.1, b - 0.1, 100)
    res_err = float(jnp.max(jnp.abs(residual(x_check))))
    print(f"  ||u'' - x*u||_inf = {res_err:.2e}")
    assert res_err < 1e-9

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
