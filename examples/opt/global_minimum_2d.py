"""Global minimum of a bivariate function using Chebfun2.

Demonstrates finding the global minimum of a smooth 2D function by
constructing a Chebfun2 and evaluating it on a grid, then verifying
against known analytical minima.

Credit: Inspired by Chebfun2 opt examples.
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import surf, contour


def run():
    print("=" * 60)
    print("Global minimum of bivariate functions")
    print("=" * 60)

    # --- f(x,y) = (x-0.3)^2 + (y+0.5)^2 has minimum at (0.3, -0.5) ---
    f1 = cj.chebfun2(lambda x, y: (x - 0.3)**2 + (y + 0.5)**2)
    # Evaluate on a fine grid to find the minimum
    xs = jnp.linspace(-1.0, 1.0, 200)
    ys = jnp.linspace(-1.0, 1.0, 200)
    XX, YY = jnp.meshgrid(xs, ys, indexing='xy')
    vals = f1(XX, YY)
    min_val = float(jnp.min(vals))
    min_idx = jnp.unravel_index(jnp.argmin(vals), vals.shape)
    x_min = float(xs[min_idx[1]])
    y_min = float(ys[min_idx[0]])
    print(f"\nf(x,y) = (x-0.3)^2 + (y+0.5)^2:")
    print(f"  Grid minimum at ({x_min:.3f}, {y_min:.3f}), value = {min_val:.2e}")
    print(f"  Exact minimum at (0.3, -0.5), value = 0")
    # With 200 grid points, we can only resolve to about 0.01
    assert abs(x_min - 0.3) < 0.02
    assert abs(y_min - (-0.5)) < 0.02
    assert min_val < 1e-3

    # --- Integral gives information about the function ----------------
    integral1 = float(f1.sum2())
    # int_{-1}^{1} int_{-1}^{1} (x-0.3)^2 + (y+0.5)^2 dx dy
    # = 2*(int (x-0.3)^2 dx from -1 to 1) + 2*(int (y+0.5)^2 dy from -1 to 1)
    # int (x-a)^2 dx from -1 to 1 = [x^3/3 - ax^2 + a^2 x]_{-1}^{1}
    # = (1/3 - a + a^2) - (-1/3 - a - a^2) = 2/3 + 2a^2
    # a=0.3: 2/3 + 2*(0.09) = 2/3 + 0.18 = 0.8467
    # a=-0.5: 2/3 + 2*(0.25) = 2/3 + 0.5 = 1.1667
    exact1 = 2.0 * (2.0/3.0 + 2.0*0.3**2) + 2.0 * (2.0/3.0 + 2.0*0.5**2)
    print(f"\n  Integral of f over [-1,1]^2: {integral1:.10f}")
    print(f"  Exact: {exact1:.10f}")
    assert abs(integral1 - exact1) < 1e-10

    # --- g(x,y) = cos(x)*cos(y): maximum at (0, 0) value 1 -----------
    g = cj.chebfun2(lambda x, y: jnp.cos(x) * jnp.cos(y))
    vals_g = g(XX, YY)
    max_val_g = float(jnp.max(vals_g))
    print(f"\ng(x,y) = cos(x)*cos(y):")
    print(f"  Grid maximum: {max_val_g:.10f}  (exact: 1.0)")
    assert abs(max_val_g - 1.0) < 1e-4  # Grid resolution limited

    # Verify integral: int cos(x)cos(y) over [-1,1]^2 = (2*sin(1))^2
    integral_g = float(g.sum2())
    exact_g = float((2.0 * jnp.sin(jnp.array(1.0)))**2)
    print(f"  Integral over [-1,1]^2: {integral_g:.12f}")
    print(f"  Exact (2*sin(1))^2:     {exact_g:.12f}")
    assert abs(integral_g - exact_g) < 1e-10

    # --- Rosenbrock-like: f(x,y) = (1-x)^2 + 10*(y-x^2)^2 -----------
    # on [-1,1]^2: minimum at (1,1), value=0 but x=1 is on boundary
    # Actually minimum in interior is near (1,1) but boundary constrained
    f_rb = cj.chebfun2(lambda x, y: (1.0 - x)**2 + 10.0 * (y - x**2)**2)
    integral_rb = float(f_rb.sum2())
    print(f"\nRosenbrock (1-x)^2 + 10*(y-x^2)^2:")
    print(f"  Integral over [-1,1]^2: {integral_rb:.6f}")
    # Just verify it's a positive function
    # min on boundary of [-1,1]^2; at (1,1): f=0
    val_at_11 = float(f_rb(jnp.array(1.0), jnp.array(1.0)))
    print(f"  f(1,1) = {val_at_11:.2e}  (exact: 0)")
    assert val_at_11 < 1e-12

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = contour(f1, title="(x−0.3)² + (y+0.5)² — global minimum")
    ax.plot(0.3, -0.5, "r*", markersize=12, label="min (0.3, −0.5)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "global_minimum_2d.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
