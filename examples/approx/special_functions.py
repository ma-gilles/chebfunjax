"""Chebfun approximation of special functions.

Demonstrates that Chebfun can adaptively approximate various special
functions (erf, Airy, Bessel) and supports operations on them.

Credit: Inspired by Chebfun approx examples.
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import jax.numpy as jnp
import numpy as np
from scipy.special import erf as scipy_erf, erfc as scipy_erfc, airy as scipy_airy
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj


def run():
    print("=" * 60)
    print("Chebfun approximation of special functions")
    print("=" * 60)

    # --- Error function erf(x) ---------------------------------------
    f_erf = cj.chebfun(lambda x: jnp.array(scipy_erf(np.array(x))), domain=(-3.0, 3.0))
    print(f"\nerf(x) on [-3, 3]:")
    print(f"  Length: {len(f_erf)}")
    print(f"  erf(0) = {float(f_erf(jnp.array(0.0))):.14f}  (exact: 0)")
    print(f"  erf(1) = {float(f_erf(jnp.array(1.0))):.14f}  "
          f"(exact: {float(scipy_erf(1.0)):.14f})")
    print(f"  erf(3) ~ {float(f_erf(jnp.array(3.0))):.14f}  "
          f"(exact: {float(scipy_erf(3.0)):.14f})")
    # erf(-x) = -erf(x)
    assert abs(float(f_erf(jnp.array(0.0)))) < 1e-14
    err_erf1 = abs(float(f_erf(jnp.array(1.0))) - float(scipy_erf(1.0)))
    assert err_erf1 < 1e-13

    # Integral of erf(x) from 0 to 1
    f_erf2 = cj.chebfun(lambda x: jnp.array(scipy_erf(np.array(x))), domain=(0.0, 1.0))
    integral_erf = float(f_erf2.sum())
    # Exact: integral_0^1 erf(x) dx = [x*erf(x) + exp(-x^2)/sqrt(pi)]_0^1
    #                                = erf(1) + exp(-1)/sqrt(pi) - 1/sqrt(pi)
    exact_erf_int = (float(scipy_erf(1.0)) +
                     (float(jnp.exp(jnp.array(-1.0))) - 1.0) /
                     float(jnp.sqrt(jnp.pi)))
    err_erf_int = abs(integral_erf - exact_erf_int)
    print(f"\n  Integral of erf(x) from 0 to 1: {integral_erf:.12f}")
    print(f"  Exact: {exact_erf_int:.12f}")
    print(f"  Error: {err_erf_int:.2e}")
    assert err_erf_int < 1e-12

    # --- erfc(x) = 1 - erf(x) ---------------------------------------
    f_erfc = cj.chebfun(lambda x: jnp.array(scipy_erfc(np.array(x))), domain=(-3.0, 3.0))
    print(f"\nerfc(x) on [-3, 3]:")
    print(f"  Length: {len(f_erfc)}")
    # Verify erfc(x) + erf(x) = 1 everywhere
    x_test = jnp.linspace(-3.0, 3.0, 100)
    sum_vals = f_erf(x_test) + f_erfc(x_test)
    err_sum = float(jnp.max(jnp.abs(sum_vals - 1.0)))
    print(f"  ||erf(x) + erfc(x) - 1||_inf = {err_sum:.2e}")
    assert err_sum < 1e-12

    # --- Airy function Ai(x) ----------------------------------------
    Ai_vals_func = lambda x: jnp.array(scipy_airy(np.array(x))[0])
    f_airy = cj.chebfun(Ai_vals_func, domain=(-5.0, 3.0))
    print(f"\nAi(x) (Airy function) on [-5, 3]:")
    print(f"  Length: {len(f_airy)}")
    print(f"  Ai(0) = {float(f_airy(jnp.array(0.0))):.12f}  "
          f"(exact: {scipy_airy(0.0)[0]:.12f})")
    err_airy0 = abs(float(f_airy(jnp.array(0.0))) - float(scipy_airy(0.0)[0]))
    assert err_airy0 < 1e-13

    # Verify Airy equation: Ai'' - x*Ai = 0
    f_airy_diff2 = f_airy.diff(2)
    x_id = cj.chebfun(lambda t: t, domain=(-5.0, 3.0))
    x_check = jnp.linspace(-5.0, 3.0, 50)
    residual_vals = f_airy_diff2(x_check) - (x_id * f_airy)(x_check)
    max_res = float(jnp.max(jnp.abs(residual_vals)))
    print(f"  ||Ai'' - x*Ai||_inf = {max_res:.2e}  (should be ~0)")
    assert max_res < 1e-8

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
