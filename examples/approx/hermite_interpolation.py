"""Polynomial interpolation and polyfit.

Demonstrates the polyfit method for fitting polynomials to data,
and shows the Runge phenomenon for equispaced interpolation.

Credit: Inspired by Chebfun approx/Interp.m and related examples.
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
    print("Polynomial interpolation and polyfit")
    print("=" * 60)

    # --- Runge function: 1/(1 + 25*x^2) ----------------------------
    # MATLAB: f = chebfun(@(x) 1./(1+25*x.^2));
    runge = lambda x: 1.0 / (1.0 + 25.0 * x**2)
    f = cj.chebfun(lambda x: runge(x))
    print(f"\nRunge function 1/(1+25x^2) on [-1,1]:")
    print(f"  Chebfun length: {len(f)}")

    # Evaluate at test points
    x_test = jnp.linspace(-1.0, 1.0, 200)
    f_vals = f(x_test)
    exact_vals = runge(x_test)
    err = float(jnp.max(jnp.abs(f_vals - exact_vals)))
    print(f"  Max error at 200 pts: {err:.2e}")
    assert err < 1e-13

    # --- polyfit: fit a polynomial to Chebyshev node data -----------
    # MATLAB: p = polyfit(chebpts(10), chebfun(@(x) exp(x)), 9);
    # chebfunjax's polyfit method
    f_smooth = cj.chebfun(lambda x: jnp.exp(jnp.sin(3.0 * x)))
    print(f"\nPolyfit: exp(sin(3x)) on [-1,1]:")
    print(f"  Full chebfun length: {len(f_smooth)}")

    # Low-degree approximation
    p5 = f_smooth.polyfit(5)   # degree-5 polynomial
    p10 = f_smooth.polyfit(10) # degree-10 polynomial
    # p_n should be a polynomial of degree n
    print(f"  deg-5 polyfit length: {len(p5)}")
    print(f"  deg-10 polyfit length: {len(p10)}")

    # Check that polyfit(n) gives a polynomial with n+1 terms
    assert len(p5) == 6, f"Expected 6, got {len(p5)}"
    assert len(p10) == 11, f"Expected 11, got {len(p10)}"

    # Error of degree-10 fit should be small
    err10 = float((f_smooth - p10).norm())
    print(f"  ||f - p_10||_2 = {err10:.2e}")

    # --- Exact polynomial: x^3 should be reproduced exactly ----------
    # MATLAB: p = chebfun('x.^3'); polyfit(p, 3)
    f_poly = cj.chebfun(lambda x: x**3 - 2.0*x + 1.0)
    p3 = f_poly.polyfit(3)
    err_poly = float((f_poly - p3).norm())
    print(f"\nx^3 - 2x + 1, polyfit(3) error: {err_poly:.2e}")
    assert err_poly < 1e-14, f"Polynomial not reproduced exactly: {err_poly}"

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
