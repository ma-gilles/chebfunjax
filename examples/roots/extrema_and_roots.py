"""Finding extrema and roots with Chebfun.

Demonstrates min(), max(), minandmax(), norm(), and roots() for
various functions.

Credit: Inspired by Chebfun examples roots/Extrema.m.
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
    print("Extrema and root-finding")
    print("=" * 60)

    # --- min and max of sin(x) on [0, 2*pi] -------------------------
    dom = (0.0, float(2.0 * jnp.pi))
    f = cj.chebfun(lambda x: jnp.sin(x), domain=dom)

    x_min, y_min = f.min()
    x_max, y_max = f.max()
    print(f"\nsin(x) on [0, 2*pi]:")
    print(f"  min: f({x_min:.10f}) = {y_min:.10f}  "
          f"(exact: f(3*pi/2) = -1)")
    print(f"  max: f({x_max:.10f}) = {y_max:.10f}  "
          f"(exact: f(pi/2) = 1)")

    assert abs(y_min + 1.0) < 1e-11
    assert abs(y_max - 1.0) < 1e-11
    assert abs(x_min - float(3.0 * jnp.pi / 2.0)) < 1e-10
    assert abs(x_max - float(jnp.pi / 2.0)) < 1e-10

    # --- minandmax of x^4 - 2x^2 on [-1.5, 1.5] -------------------
    # g(x) = x^4 - 2x^2 = (x^2-1)^2 - 1
    # g'(x) = 4x^3 - 4x = 4x(x^2-1) = 0 at x=0, ±1
    # g(0) = 0, g(1) = -1 (min), g(-1) = -1 (min), g(1.5) = (2.25-2.25+...) = 5.0625-4.5=-something
    # g(1.5) = (1.5)^4 - 2*(1.5)^2 = 5.0625 - 4.5 = 0.5625
    # g(-1.5) = 0.5625 (same by symmetry)
    # min = -1 at x = ±1; max = 0.5625 at x = ±1.5 (boundary)
    dom2 = (-1.5, 1.5)
    g = cj.chebfun(lambda x: x**4 - 2.0 * x**2, domain=dom2)
    (x_min2, y_min2), (x_max2, y_max2) = g.minandmax()
    print(f"\ng(x) = x^4 - 2x^2 on [-1.5, 1.5]:")
    print(f"  min: g({x_min2:.10f}) = {y_min2:.10f}  (exact: g(±1) = -1)")
    print(f"  max: g({x_max2:.10f}) = {y_max2:.10f}  (exact: g(±1.5) = 0.5625)")
    assert abs(y_min2 + 1.0) < 1e-10
    assert abs(y_max2 - 0.5625) < 1e-10

    # --- Roots = zeros of f ------------------------------------------
    dom3 = (-3.0, 3.0)
    p = cj.chebfun(lambda x: x**3 - x, domain=dom3)  # roots: -1, 0, 1
    roots_p = p.roots()
    roots_arr = np.sort(np.array(roots_p))
    print(f"\np(x) = x^3 - x on [-3, 3]:")
    print(f"  Roots: {roots_arr}")
    assert len(roots_arr) == 3
    assert abs(roots_arr[0] + 1.0) < 1e-11
    assert abs(roots_arr[1]) < 1e-11
    assert abs(roots_arr[2] - 1.0) < 1e-11

    # --- Roots of derivative = critical points ----------------------
    dp = p.diff()  # p'(x) = 3x^2 - 1, roots at ±1/sqrt(3)
    crit_pts = dp.roots()
    crit_arr = np.sort(np.array(crit_pts))
    exact_crit = 1.0 / np.sqrt(3.0)
    assert len(crit_arr) == 2
    assert abs(crit_arr[0] + exact_crit) < 1e-10
    assert abs(crit_arr[1] - exact_crit) < 1e-10

    # --- L2 norm of sin(x) on [0, pi] = sqrt(pi/2) ------------------
    dom4 = (0.0, float(jnp.pi))
    s = cj.chebfun(lambda x: jnp.sin(x), domain=dom4)
    norm2 = float(s.norm(2))
    exact_norm2 = float(jnp.sqrt(jnp.pi / 2.0))
    print(f"\n||sin(x)||_2 on [0,pi] = {norm2:.15f}")
    print(f"  exact = sqrt(pi/2) = {exact_norm2:.15f}")
    assert abs(norm2 - exact_norm2) < 1e-12

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
