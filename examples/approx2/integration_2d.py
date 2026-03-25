"""2D integration with Chebfun2.

Demonstrates computing double integrals and partial integrals
(marginals) using Chebfun2.

Credit: Inspired by Chebfun2 approx2/Integration.m.
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj


def run():
    print("=" * 60)
    print("2D integration with Chebfun2")
    print("=" * 60)

    # --- Integral of exp(x+y) over [-1,1]^2 = (e - 1/e)^2 ----------
    f1 = cj.chebfun2(lambda x, y: jnp.exp(x + y))
    integral1 = float(f1.sum2())
    exact1 = float((jnp.exp(jnp.array(1.0)) - jnp.exp(jnp.array(-1.0)))**2)
    print(f"\nIntegral of exp(x+y) over [-1,1]^2:")
    print(f"  Computed: {integral1:.15f}")
    print(f"  Exact (e-1/e)^2: {exact1:.15f}")
    print(f"  Error: {abs(integral1 - exact1):.2e}")
    assert abs(integral1 - exact1) < 1e-10

    # --- Integral of x^2 + y^2 over [-1,1]^2 = 8/3 ------------------
    f2 = cj.chebfun2(lambda x, y: x**2 + y**2)
    integral2 = float(f2.sum2())
    exact2 = 8.0 / 3.0  # = 2*(int x^2 dx from -1 to 1)*2 = 2*2/3*2
    print(f"\nIntegral of x^2+y^2 over [-1,1]^2:")
    print(f"  Computed: {integral2:.15f}")
    print(f"  Exact: 8/3 = {exact2:.15f}")
    assert abs(integral2 - exact2) < 1e-13

    # --- Integral of sin(pi*x)*sin(pi*y) over [-1,1]^2 = 0 ----------
    pi = float(jnp.pi)
    f3 = cj.chebfun2(lambda x, y: jnp.sin(pi*x) * jnp.sin(pi*y))
    integral3 = float(f3.sum2())
    print(f"\nIntegral of sin(pi*x)*sin(pi*y) over [-1,1]^2:")
    print(f"  Computed: {integral3:.2e}  (exact: 0)")
    assert abs(integral3) < 1e-13

    # --- Custom domain: integral of 1 over [0,1]^2 = 1 --------------
    f4 = cj.chebfun2(lambda x, y: jnp.ones_like(x) + jnp.zeros_like(y),
                     domain=[0.0, 1.0, 0.0, 1.0])
    integral4 = float(f4.sum2())
    print(f"\nIntegral of 1 over [0,1]^2:")
    print(f"  Computed: {integral4:.15f}  (exact: 1.0)")
    assert abs(integral4 - 1.0) < 1e-13

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
