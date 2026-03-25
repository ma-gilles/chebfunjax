"""Boundary layer: advection-diffusion equation.

Solves eps*u'' + u' = 0, u(0)=0, u(1)=1 for small eps.
Exact: u = (1 - exp(-x/eps)) / (1 - exp(-1/eps)).

Credit: Inspired by Chebfun ode-linear/BoundaryLayer.m.
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
    print("Boundary layer: eps*u'' + u' = 0")
    print("=" * 60)

    for eps in [0.1, 0.01]:
        print(f"\n--- eps = {eps} ---")
        exact = lambda x, e=eps: (1.0 - jnp.exp(-x / e)) / (1.0 - jnp.exp(-1.0 / e))
        N = Chebop(lambda x, u, e=eps: e * u.diff(2) + u.diff(), domain=(0.0, 1.0))
        N.lbc = 0.0
        N.rbc = 1.0
        u = N.solve(0.0)
        print(f"  Chebfun length: {len(u)}")

        x_test = jnp.linspace(0.0, 1.0, 500)
        exact_vals = exact(x_test)
        err = float(jnp.max(jnp.abs(u(x_test) - exact_vals)))
        print(f"  Max error vs exact: {err:.2e}")
        assert err < 1e-7, f"eps={eps} error too large: {err}"

        # Verify boundary conditions
        assert abs(float(u(jnp.array(0.0)))) < 1e-10
        assert abs(float(u(jnp.array(1.0))) - 1.0) < 1e-10

        # Find the boundary layer width (where u goes from 0.1 to 0.9)
        # Approximately at x ~ eps * log(9), but let us use roots
        f_01 = u - 0.1
        f_09 = u - 0.9
        r01 = np.array(f_01.roots())
        r09 = np.array(f_09.roots())
        if len(r01) > 0 and len(r09) > 0:
            bl_width = float(r09[0]) - float(r01[0])
            print(f"  BL width (10%-90%): {bl_width:.6f}")

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
