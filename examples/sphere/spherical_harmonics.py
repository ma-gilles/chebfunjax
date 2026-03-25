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
    # Spherefun takes (lambda, theta): lambda=longitude, theta=colatitude
    f_Y11 = Spherefun.from_function(lambda lam, th: jnp.cos(lam) * jnp.sin(th))
    print(f"\nSpherefun cos(λ)sin(θ) [≈ Y_1^1]:")
    print(f"  Rank: {f_Y11.rank}")

    # Integral should be 0
    integral_Y11 = float(f_Y11.sum())
    print(f"  Integral = {integral_Y11:.2e}  (expected: ~0)")
    assert abs(integral_Y11) < 0.1

    # Evaluate on equator: f(lambda=0, theta=pi/2) = cos(0)*sin(pi/2) = 1
    val_eq = float(f_Y11(jnp.array(0.0), jnp.array(float(jnp.pi)/2)))
    exact_eq = float(jnp.cos(jnp.array(0.0)) * jnp.sin(jnp.array(float(jnp.pi)/2)))
    err_eq = abs(val_eq - exact_eq)
    print(f"  f at λ=0, θ=π/2 = {val_eq:.8f}  (exact: {exact_eq:.8f})")
    assert err_eq < 1e-8

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig = plt.figure(figsize=(12, 4))

    # Mollweide-like projection of cos(lambda)*sin(theta)
    ax1 = fig.add_subplot(131)
    lam_p = np.linspace(0, 2*np.pi, 200)
    th_p = np.linspace(0, np.pi, 100)
    LAM, TH = np.meshgrid(lam_p, th_p)
    Z = np.cos(LAM) * np.sin(TH)
    im1 = ax1.contourf(np.degrees(LAM) - 180, np.degrees(TH) - 90, Z,
                        levels=20, cmap="RdBu_r")
    ax1.set_title("cos(λ)sin(θ)", fontsize=11)
    ax1.set_xlabel("Longitude (°)"); ax1.set_ylabel("Colatitude (°)")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    # sin(theta)^2 * cos(2*lambda)
    ax2 = fig.add_subplot(132)
    Z2 = np.sin(TH)**2 * np.cos(2 * LAM)
    im2 = ax2.contourf(np.degrees(LAM) - 180, np.degrees(TH) - 90, Z2,
                        levels=20, cmap="RdBu_r")
    ax2.set_title("sin²(θ)cos(2λ)", fontsize=11)
    ax2.set_xlabel("Longitude (°)")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    # 3D sphere plot
    ax3 = fig.add_subplot(133, projection='3d')
    U, V = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    X = np.cos(U) * np.sin(V)
    Y = np.sin(U) * np.sin(V)
    Zs = np.cos(V)
    colors = np.cos(U) * np.sin(V)
    ax3.plot_surface(X, Y, Zs, facecolors=plt.cm.RdBu_r((colors + 1) / 2),
                      alpha=0.9, linewidth=0)
    ax3.set_title("Sphere: cos(λ)sin(θ)", fontsize=11)
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")

    fig.suptitle("Spherical harmonics and Spherefun", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "spherical_harmonics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
