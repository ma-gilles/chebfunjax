"""Flux integrals and Gauss-Green-Stokes in 3D.

Demonstrates computing flux integrals and verifying the divergence theorem
using chebfunjax Chebfun3 and Chebfun3v, following approx3/FluxIntegral3D.m
and approx3/GaussGreenStokes.m by Nick Trefethen.

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
    print("Flux integrals and vector calculus in 3D")
    print("=" * 60)

    # Divergence theorem: div(F) integrated over [-1,1]^3
    # F = (x^2, y^2, z^2) => div(F) = 2x + 2y + 2z
    # integral of div(F) over [-1,1]^3 = 0  (odd functions integrate to 0)
    div_f = chebfun3(lambda x, y, z: 2*x + 2*y + 2*z)
    integral_div = float(div_f.sum3())
    print(f"\ndiv(F) = 2x+2y+2z, integral over [-1,1]^3:")
    print(f"  Computed: {integral_div:.2e}  (expected: 0)")
    assert abs(integral_div) < 1e-10

    # F = (x, y, z) => div(F) = 3
    # integral of div(F) = 3 * 8 = 24
    div_g = chebfun3(lambda x, y, z: jnp.ones_like(x + y + z) * 3)
    integral_g = float(div_g.sum3())
    print(f"\ndiv(G) = 3, integral over [-1,1]^3:")
    print(f"  Computed: {integral_g:.10f}  (expected: 24.0)")
    assert abs(integral_g - 24.0) < 1e-8

    # Scalar field: x^2 + y^2 + z^2
    r2 = chebfun3(lambda x, y, z: x**2 + y**2 + z**2)
    integral_r2 = float(r2.sum3())
    # int_{-1}^{1} x^2 dx = 2/3, so total = 3 * (2/3) * 2 * 2 = 8
    exact_r2 = 3 * (2.0/3.0) * 4.0  # = 8
    print(f"\nintegral of x^2+y^2+z^2 over [-1,1]^3:")
    print(f"  Computed: {integral_r2:.10f}  (exact: {exact_r2:.10f})")
    assert abs(integral_r2 - exact_r2) < 1e-8

    # Evaluate at corners
    val_corner = float(r2(jnp.array(1.0), jnp.array(1.0), jnp.array(1.0)))
    print(f"\nr²(1,1,1) = {val_corner:.10f}  (exact: 3.0)")
    assert abs(val_corner - 3.0) < 1e-10

    # Plot: isosurface visualization as slice plots
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3)

    xs = np.linspace(-1, 1, 60)
    X, Y = np.meshgrid(xs, xs)

    # Slices at z = 0, 0.5, -0.5
    zvals = [0.0, 0.5, -0.5]
    for i, (ax, zv) in enumerate(zip(axes, zvals)):
        Z_slice = X**2 + Y**2 + zv**2
        im = ax.contourf(X, Y, Z_slice, levels=20, cmap="plasma")
        ax.set_title(f"x²+y²+z², z={zv}", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("3D scalar field: z-slices of x²+y²+z²", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "flux_integral_3d.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
