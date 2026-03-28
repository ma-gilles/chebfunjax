"""Smooth function approximation in 3D (Chebfun3).

Demonstrates Tucker-format 3D function approximation with chebfunjax,
following the Chebfun3 examples in approx3/ by Nick Trefethen.

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
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.chebfun3d import chebfun3

def run():
    print("=" * 60)
    print("Smooth function approximation in 3D (Chebfun3)")
    print("=" * 60)

    # exp(x+y+z) is rank-1 in Tucker format
    f = chebfun3(lambda x, y, z: jnp.exp(x + y + z))
    print(f"\nexp(x+y+z): Tucker rank {f.rank}")

    # Evaluate at (0, 0, 0): should be 1
    val = float(f(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)))
    print(f"f(0,0,0) = {val:.15f}  (exact: 1.0)")
    assert abs(val - 1.0) < 1e-10

    # Evaluate at (0.5, 0.3, -0.2): exp(0.6)
    exact = float(jnp.exp(jnp.array(0.6)))
    val2 = float(f(jnp.array(0.5), jnp.array(0.3), jnp.array(-0.2)))
    print(f"f(0.5,0.3,-0.2) = {val2:.15f}  (exact: {exact:.15f})")
    assert abs(val2 - exact) < 1e-10

    # Triple integral: int int int exp(x+y+z) dV over [-1,1]^3 = (e - 1/e)^3
    integral = float(f.sum3())
    exact_int = float((jnp.exp(jnp.array(1.0)) - jnp.exp(jnp.array(-1.0)))**3)
    print(f"\nIntegral of exp(x+y+z) over [-1,1]^3:")
    print(f"  Computed: {integral:.15f}")
    print(f"  Exact:    {exact_int:.15f}")
    print(f"  Error:    {abs(integral - exact_int):.2e}")
    assert abs(integral - exact_int) < 1e-8

    # A more complex function
    g = chebfun3(lambda x, y, z: jnp.sin(x) * jnp.cos(y) * jnp.exp(-z**2))
    print(f"\nsin(x)cos(y)exp(-z^2): Tucker rank {g.rank}")
    val3 = float(g(jnp.array(0.3), jnp.array(0.4), jnp.array(0.5)))
    exact3 = float(jnp.sin(jnp.array(0.3)) * jnp.cos(jnp.array(0.4)) * jnp.exp(jnp.array(-0.25)))
    print(f"g(0.3,0.4,0.5) error: {abs(val3 - exact3):.2e}")
    assert abs(val3 - exact3) < 1e-10

    # Plot cross-section slices using the library's plot_slices
    from chebfunjax.plotting import plot_slices

    _here = os.path.dirname(os.path.abspath(__file__))

    fig1, ax1 = plot_slices(f, title="exp(x+y+z)")
    fig1.savefig(os.path.join(_here, "smooth_functions_3d_exp.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plot_slices(g, title="sin(x)cos(y)exp(-z²)")
    fig2.savefig(os.path.join(_here, "smooth_functions_3d.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
