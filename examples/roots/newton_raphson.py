"""Newton's method root-finding with Chebfun.

Demonstrates finding roots of a cubic polynomial and compares
with Newton's method applied iteratively.

Credit: Inspired by Chebfun examples roots/NewtonRaphson.m.
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import plot


def run():
    print("=" * 60)
    print("Newton's method and root-finding")
    print("=" * 60)

    # MATLAB: dom = [-3 3]; f = chebfun('x.^3 - 3*x.^2 + 2', dom);
    dom = (-3.0, 3.0)
    f = cj.chebfun(lambda x: x**3 - 3.0 * x**2 + 2.0, domain=dom)

    # MATLAB: roots(f)
    r = f.roots()
    r_arr = np.sort(np.array(r))
    print(f"\nf(x) = x^3 - 3x^2 + 2:")
    print(f"  All roots on [-3, 3]: {r_arr}")
    # Factor: x^3 - 3x^2 + 2 = (x-1)(x^2 - 2x - 2)?
    # Check: (x-1)(x^2-2x-2) = x^3 - 2x^2 - 2x - x^2 + 2x + 2 = x^3 - 3x^2 + 2 YES
    # So roots: x=1 and x = (2 ± sqrt(4+8))/2 = 1 ± sqrt(3)
    exact_roots = np.sort([1.0 - np.sqrt(3.0), 1.0, 1.0 + np.sqrt(3.0)])
    print(f"  Exact roots: {exact_roots}")
    assert len(r_arr) == 3, f"Expected 3 roots, got {len(r_arr)}"
    for k in range(3):
        err = abs(r_arr[k] - exact_roots[k])
        print(f"  Root {k+1}: computed = {r_arr[k]:.12f}, exact = {exact_roots[k]:.12f}, err = {err:.2e}")
        assert err < 1e-11, f"Root {k+1} error too large: {err}"

    # --- Manual Newton's method for comparison -----------------------
    print(f"\nManual Newton iteration on f(x) = x^3 - 3x^2 + 2:")
    # Starting from x0 = 2.5, converge to 1 + sqrt(3) ~ 2.732
    f_np = lambda x: x**3 - 3.0 * x**2 + 2.0
    df_np = lambda x: 3.0 * x**2 - 6.0 * x
    x_newton = 2.5
    for k in range(10):
        x_new = x_newton - f_np(x_newton) / df_np(x_newton)
        err = abs(x_new - (1.0 + np.sqrt(3.0)))
        print(f"  Step {k+1}: x = {x_new:.12f}, err = {err:.2e}")
        x_newton = x_new
        if err < 1e-14:
            break
    assert abs(x_newton - (1.0 + np.sqrt(3.0))) < 1e-12

    # --- Verify f(roots) = 0 -----------------------------------------
    for ri in r_arr:
        val = float(f(jnp.array(ri)))
        assert abs(val) < 1e-13, f"|f({ri})| = {abs(val)}"

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f, title="x³ − 3x² + 2: roots", label="p(x)")
    ax.axhline(0, color="k", linewidth=0.6)
    import numpy as _np
    ax.plot(r_arr, _np.zeros_like(r_arr), "o", color="#E04040",
            markersize=7, label="roots")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "newton_raphson.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
