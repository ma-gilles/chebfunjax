"""Spherical harmonics and Spherefun.

Demonstrates spherical harmonic approximation and spherefun operations,
following sphere/SphericalHarmonics.m and sphere/AtmosphericTemperature.m.

Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    print("Spherical harmonics and Spherefun")
    print("=" * 60)

    # --- Spherefun approximation ---
    # Constant function: f = 1
    f_const = Spherefun.from_function(lambda lam, th: jnp.ones_like(lam + th))
    print(f"\nSpherefun f=1:")
    print(f"  Rank: {f_const.rank}")

    # Integral of 1 over S^2 = 4*pi
    integral_const = float(f_const.sum())
    print(f"  Integral = {integral_const:.8f}  (exact: {4*np.pi:.8f})")
    assert abs(integral_const - 4*np.pi) < 0.01

    # cos(lambda) * sin(theta): first spherical harmonic Y_1^1
    f_Y11 = Spherefun.from_function(lambda lam, th: jnp.cos(lam) * jnp.sin(th))
    print(f"\nSpherefun cos(lam)sin(th) [~= Y_1^1]:")
    print(f"  Rank: {f_Y11.rank}")

    # Integral should be 0
    integral_Y11 = float(f_Y11.sum())
    print(f"  Integral = {integral_Y11:.2e}  (expected: ~0)")
    assert abs(integral_Y11) < 0.1

    # Evaluate on equator: f(lambda=0, theta=pi/2) = cos(0)*sin(pi/2) = 1
    val_eq = float(f_Y11(jnp.array(0.0), jnp.array(float(jnp.pi)/2)))
    exact_eq = float(jnp.cos(jnp.array(0.0)) * jnp.sin(jnp.array(float(jnp.pi)/2)))
    err_eq = abs(val_eq - exact_eq)
    print(f"  f at lam=0, th=pi/2 = {val_eq:.8f}  (exact: {exact_eq:.8f})")
    assert err_eq < 1e-8

    # --- Plot: three spherical harmonics as 3D coloured spheres ---
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/sphere')
    os.makedirs(outdir, exist_ok=True)

    # Create Spherefun objects for each harmonic and use plot_sphere
    Y11 = Spherefun.from_function(lambda lam, th: jnp.cos(lam) * jnp.sin(th))
    Y20 = Spherefun.from_function(lambda lam, th: (3 * jnp.cos(th)**2 - 1) / 2)
    Y22 = Spherefun.from_function(lambda lam, th: jnp.sin(th)**2 * jnp.cos(2 * lam))

    fig, ax = plot_sphere(Y11, title='$Y_1^1$')
    fig.savefig(os.path.join(outdir, "spherical_harmonics_Y11.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plot_sphere(Y20, title='$Y_2^0$')
    fig.savefig(os.path.join(outdir, "spherical_harmonics_Y20.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plot_sphere(Y22, title='$Y_2^2$')
    fig.savefig(os.path.join(outdir, "spherical_harmonics_Y22.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
