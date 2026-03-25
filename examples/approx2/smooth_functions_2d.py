"""2D smooth function approximation (Chebfun2).

Demonstrates constructing Chebfun2 approximations of smooth bivariate
functions, checking ranks, and evaluating.

Credit: Inspired by Chebfun2 examples.
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
    print("2D smooth function approximation (Chebfun2)")
    print("=" * 60)

    # --- exp(x + y) is separable: rank 1 ----------------------------
    # MATLAB: f = chebfun2(@(x,y) exp(x+y));
    f1 = cj.chebfun2(lambda x, y: jnp.exp(x + y))
    print(f"\nexp(x+y) on [-1,1]^2:")
    print(f"  Rank: {f1.rank}")
    # exp(x+y) = exp(x)*exp(y) is rank-1
    assert f1.rank == 1, f"exp(x+y) should be rank 1, got {f1.rank}"

    # Evaluate at (0, 0): should be 1
    val00 = float(f1(jnp.array(0.0), jnp.array(0.0)))
    print(f"  f(0,0) = {val00:.15f}  (exact: 1.0)")
    assert abs(val00 - 1.0) < 1e-13

    # Evaluate at (0.5, -0.5): should be 1
    val_mid = float(f1(jnp.array(0.5), jnp.array(-0.5)))
    print(f"  f(0.5,-0.5) = {val_mid:.15f}  (exact: 1.0)")
    assert abs(val_mid - 1.0) < 1e-13

    # --- 2D integral: int int exp(x+y) dA = (e - 1/e)^2 -----------
    integral1 = float(f1.sum2())
    exact1 = float((jnp.exp(jnp.array(1.0)) - jnp.exp(jnp.array(-1.0)))**2)
    print(f"\n  Integral of exp(x+y) over [-1,1]^2:")
    print(f"  Computed: {integral1:.15f}")
    print(f"  Exact (e-1/e)^2: {exact1:.15f}")
    print(f"  Error: {abs(integral1 - exact1):.2e}")
    assert abs(integral1 - exact1) < 1e-10

    # --- cos(x + y^2) -----------------------------------------------
    f2 = cj.chebfun2(lambda x, y: jnp.cos(x + y**2))
    print(f"\ncos(x + y^2) on [-1,1]^2:")
    print(f"  Rank: {f2.rank}")
    # Not separable, should have rank > 1
    assert f2.rank > 1, "cos(x+y^2) should not be rank 1"

    # Verify evaluation
    x_test, y_test = 0.3, 0.7
    val_exact = float(jnp.cos(jnp.array(x_test + y_test**2)))
    val_computed = float(f2(jnp.array(x_test), jnp.array(y_test)))
    err = abs(val_computed - val_exact)
    print(f"  f({x_test}, {y_test}) error: {err:.2e}")
    assert err < 1e-12

    # --- Franke's function (classic test) ---------------------------
    def franke(x, y):
        t1 = 0.75 * jnp.exp(-((9*x-2)**2 + (9*y-2)**2)/4.0)
        t2 = 0.75 * jnp.exp(-((9*x+1)**2)/49.0 - (9*y+1)/10.0)
        t3 = 0.5  * jnp.exp(-((9*x-7)**2 + (9*y-3)**2)/4.0)
        t4 = -0.2 * jnp.exp(-((9*x-4)**2 + (9*y-7)**2))
        return t1 + t2 + t3 + t4

    f3 = cj.chebfun2(franke)
    print(f"\nFranke's function on [-1,1]^2:")
    print(f"  Rank: {f3.rank}")

    # Verify evaluation at a test point
    val3_computed = float(f3(jnp.array(0.0), jnp.array(0.0)))
    val3_exact = float(franke(jnp.array(0.0), jnp.array(0.0)))
    print(f"  f(0,0) error: {abs(val3_computed - val3_exact):.2e}")
    assert abs(val3_computed - val3_exact) < 1e-10

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
