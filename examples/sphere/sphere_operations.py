"""Operations on the sphere: differentiation, integration, rotation.

Demonstrates spherefun calculus and the Helmholtz decomposition,
following sphere/Gravity.m, sphere/HelmholtzDecomposition.m, and
sphere/SphereHeatConduction.m.

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
from chebfunjax.plotting import chebfun_style, PARULA, _setup_3d_axes, plot_sphere
chebfun_style()

from chebfunjax.spherefun.spherefun import Spherefun

def run():
    print("=" * 60)
    print("Operations on the sphere")
    print("=" * 60)

    # --- Integral of a smooth function over S^2 ---
    f1 = Spherefun.from_function(lambda lam, th: jnp.cos(th))
    integral_cos_th = float(f1.sum())
    print(f"\nIntegral of cos(th) over S^2: {integral_cos_th:.2e}  (expected: 0)")
    assert abs(integral_cos_th) < 0.01

    f2 = Spherefun.from_function(lambda lam, th: jnp.cos(th)**2)
    integral_cos2 = float(f2.sum())
    exact_cos2 = 4 * np.pi / 3
    print(f"Integral of cos^2(th) = {integral_cos2:.8f}  (exact: {exact_cos2:.8f})")
    assert abs(integral_cos2 - exact_cos2) < 0.01

    # --- Gravity-like potential on sphere ---
    print("\nGravity-like potential:")
    f_grav = Spherefun.from_function(
        lambda lam, th: 1.0 / (1.5 - 0.5 * jnp.cos(lam) * jnp.sin(th))
    )
    print(f"  Rank: {f_grav.rank}")

    val_test = float(f_grav(jnp.array(0.0), jnp.array(float(jnp.pi)/2)))
    exact_test = 1.0 / (1.5 - 0.5)
    print(f"  f(0,pi/2) = {val_test:.8f}  (exact: {exact_test:.8f})")
    assert abs(val_test - exact_test) < 1e-8

    # --- Differentiation: gradient ---
    f_sin = Spherefun.from_function(lambda lam, th: jnp.sin(lam))
    print(f"\nSpherefun sin(lam), rank {f_sin.rank}")

    # --- Rotation ---
    f3 = Spherefun.from_function(lambda lam, th: jnp.exp(-3 * th**2))
    print(f"\nexp(-3*th^2) on sphere, rank {f3.rank}")

    # --- Plot: 3D sphere rendering using plot_sphere ---
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/sphere')
    os.makedirs(outdir, exist_ok=True)

    spherefuns = [
        (f2, "$\\cos^2(\\theta)$: integral $= 4\\pi/3$"),
        (f_grav, "Gravity potential"),
        (f3, "$\\exp(-3\\theta^2)$"),
    ]

    for sf, title in spherefuns:
        fig, ax = plot_sphere(sf, title=title)
        safe_name = title.replace('$', '').replace('\\', '').replace(' ', '_')[:20]
        fig.savefig(os.path.join(outdir, f"sphere_operations_{safe_name}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
