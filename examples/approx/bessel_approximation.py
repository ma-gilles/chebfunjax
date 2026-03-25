"""Chebfun approximation of Bessel functions.

Demonstrates that Chebfun can accurately represent Bessel functions
and their derivatives, and verifies the recurrence relation
d/dx J_0(x) = -J_1(x).

Credit: Inspired by Chebfun approx/BesselApprox.m.
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import jax.numpy as jnp
import numpy as np
import scipy.special as sp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj


def run():
    print("=" * 60)
    print("Bessel function approximation")
    print("=" * 60)

    # Bessel function via scipy (no native JAX version for high accuracy)
    from scipy.special import j0 as besselj0, j1 as besselj1

    # --- J_0 on [0, 20] ----------------------------------------------
    # MATLAB: J0 = chebfun(@(x) besselj(0, x), [0, 20]);
    dom_20 = (0.0, 20.0)
    J0 = cj.chebfun(lambda x: jnp.array(besselj0(np.array(x))), domain=dom_20)
    print(f"\nJ_0(x) on [0, 20]:")
    print(f"  Chebfun length: {len(J0)}")

    # Evaluate at a few known values
    print(f"  J_0(0) = {float(J0(jnp.array(0.0))):.14f}  (exact: 1.0)")
    assert abs(float(J0(jnp.array(0.0))) - 1.0) < 1e-13

    # J_0(2.4048) ~ 0 (first zero)
    first_zero = 2.4048255576957727
    print(f"  J_0({first_zero:.6f}) ~ {float(J0(jnp.array(first_zero))):.2e}  (exact: 0)")
    assert abs(float(J0(jnp.array(first_zero)))) < 1e-12

    # --- J_1 on [0, 20] ----------------------------------------------
    J1 = cj.chebfun(lambda x: jnp.array(besselj1(np.array(x))), domain=dom_20)
    print(f"\nJ_1(x) on [0, 20]:")
    print(f"  Chebfun length: {len(J1)}")

    # --- Verify d/dx J_0 = -J_1 (recurrence relation) ---------------
    # MATLAB: norm(diff(J0) + J1) / norm(J1)
    dJ0 = J0.diff()
    x_test = jnp.linspace(0.1, 20.0, 500)
    dJ0_vals = dJ0(x_test)
    J1_vals = J1(x_test)
    err = float(jnp.max(jnp.abs(dJ0_vals + J1_vals)))
    print(f"\nVerifying d/dx J_0 = -J_1:")
    print(f"  ||J_0' + J_1||_inf = {err:.2e}  (should be ~0)")
    assert err < 1e-9, f"Recurrence relation error too large: {err}"

    # --- Integral of x*J_0(x) over [0, 1] = J_1(1) ------------------
    # By integration by parts: d/dx(x*J_1(x)) = x*J_0(x)
    # So int_0^1 x*J_0(x) dx = [x*J_1(x)]_0^1 = J_1(1)
    dom_01 = (0.0, 1.0)
    xJ0_short = cj.chebfun(lambda x: x * jnp.array(besselj0(np.array(x))), domain=dom_01)
    integral = float(xJ0_short.sum())
    exact_int = float(besselj1(1.0))
    print(f"\nIntegral of x*J_0(x) from 0 to 1:")
    print(f"  Computed: {integral:.14f}")
    print(f"  J_1(1) =  {exact_int:.14f}")
    print(f"  Error:    {abs(integral - exact_int):.2e}")
    assert abs(integral - exact_int) < 1e-12

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
