"""Mean Value Theorem illustrated with Chebfun.

For a differentiable function f on [a, b], the MVT guarantees
a point c in (a, b) where f'(c) = (f(b) - f(a)) / (b - a).
We find such c using Chebfun's roots.

Credit: Inspired by Chebfun examples calc/MeanValueThm.m.
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
    print("Mean Value Theorem with Chebfun")
    print("=" * 60)

    # --- Example 1: f(x) = sin(x) on [0, 2*pi] ----------------------
    a, b = 0.0, float(2.0 * jnp.pi)
    dom = (a, b)
    f = cj.chebfun(lambda x: jnp.sin(x), domain=dom)
    fa = float(f(jnp.array(a)))
    fb = float(f(jnp.array(b)))
    mean_slope = (fb - fa) / (b - a)  # = 0 (sin(0) = sin(2*pi) = 0)

    print(f"\nf(x) = sin(x) on [0, 2*pi]:")
    print(f"  f(0)     = {fa:.10f}")
    print(f"  f(2*pi)  = {fb:.10f}")
    print(f"  Mean slope = {mean_slope:.2e}")

    df = f.diff()
    g = df - mean_slope
    c_pts = g.roots()
    c_pts_arr = np.array(c_pts)
    interior = c_pts_arr[(c_pts_arr > a + 1e-10) & (c_pts_arr < b - 1e-10)]
    print(f"  MVT points c (where f'(c)=0): {interior}")
    # For sin(x) on [0, 2*pi], f'(c) = cos(c) = 0 at c = pi/2, 3*pi/2
    assert len(interior) >= 2
    assert any(abs(c - float(jnp.pi / 2)) < 1e-6 for c in interior)
    assert any(abs(c - float(3 * jnp.pi / 2)) < 1e-6 for c in interior)

    # --- Example 2: f(x) = x^3 on [-2, 2] ---------------------------
    dom2 = (-2.0, 2.0)
    f2 = cj.chebfun(lambda x: x**3, domain=dom2)
    fa2 = float(f2(jnp.array(-2.0)))
    fb2 = float(f2(jnp.array(2.0)))
    mean_slope2 = (fb2 - fa2) / 4.0  # = (8 - (-8)) / 4 = 4
    print(f"\nf(x) = x^3 on [-2, 2]:")
    print(f"  Mean slope = {mean_slope2:.10f}  (exact: 4.0)")
    assert abs(mean_slope2 - 4.0) < 1e-12

    df2 = f2.diff()
    g2 = df2 - mean_slope2
    c2_pts = g2.roots()
    c2_arr = np.array(c2_pts)
    interior2 = c2_arr[(c2_arr > -2.0 + 1e-10) & (c2_arr < 2.0 - 1e-10)]
    print(f"  MVT points c: {interior2}")
    # f'(x) = 3x^2 = 4 => x = +-2/sqrt(3)
    c_exact = 2.0 / np.sqrt(3.0)
    assert any(abs(c - c_exact) < 1e-6 for c in interior2)
    assert any(abs(c + c_exact) < 1e-6 for c in interior2)

    # --- Example 3: exp(x) on [0, 1] --------------------------------
    dom3 = (0.0, 1.0)
    f3 = cj.chebfun(lambda x: jnp.exp(x), domain=dom3)
    fa3 = float(f3(jnp.array(0.0)))
    fb3 = float(f3(jnp.array(1.0)))
    mean_slope3 = fb3 - fa3  # = e - 1
    print(f"\nf(x) = exp(x) on [0, 1]:")
    print(f"  Mean slope = e - 1 = {mean_slope3:.10f}")

    df3 = f3.diff()  # = exp(x)
    g3 = df3 - mean_slope3
    c3_pts = g3.roots()
    c3_arr = np.array(c3_pts)
    interior3 = c3_arr[(c3_arr > 1e-10) & (c3_arr < 1.0 - 1e-10)]
    # exp(c) = e - 1 => c = log(e - 1)
    c3_exact = float(jnp.log(jnp.array(float(jnp.e) - 1.0)))
    print(f"  c* = {interior3[0]:.10f}  (exact: log(e-1) = {c3_exact:.10f})")
    assert len(interior3) >= 1
    assert abs(interior3[0] - c3_exact) < 1e-8

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
