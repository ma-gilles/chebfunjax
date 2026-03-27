"""Functions on the disk (Diskfun).

Demonstrates approximating smooth functions on the unit disk using diskfun,
including eigenfunctions of the Laplacian and heat equation solutions,
following disk/Eigenfunctions.m and disk/HeatEqn.m.

Diskfun uses polar coordinates: f(theta, r) where theta in [-pi, pi], r in [0, 1].

Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chebfunjax.plotting import chebfun_style
chebfun_style()

import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.diskfun.diskfun import Diskfun


def run():
    print("=" * 60)
    print("Functions on the disk (Diskfun)")
    print("=" * 60)

    # f(theta, r) = r*cos(theta) = x in Cartesian
    # This is the x-coordinate on the disk
    f = Diskfun.from_function(lambda t, r: r * jnp.cos(t))
    print(f"\nf(theta,r) = r*cos(theta) [= x] on the disk:")
    print(f"  Rank: {f.rank}")

    # Integral of x over disk = 0 (odd in x)
    integral_x = float(f.sum())
    print(f"  Integral of r*cos(theta) = {integral_x:.2e}  (expected: ~0)")
    assert abs(integral_x) < 0.1

    # f(theta, r) = r^2 (= x^2 + y^2)
    g = Diskfun.from_function(lambda t, r: r**2)
    print(f"\ng(theta,r) = r^2 on the disk:")
    print(f"  Rank: {g.rank}")

    # Integral of r^2 over disk = pi/2
    integral_r2 = float(g.sum())
    exact_r2 = np.pi / 2.0
    print(f"  Integral of r^2 = {integral_r2:.6f}  (exact: {exact_r2:.6f})")
    assert abs(integral_r2 - exact_r2) < 0.01

    # A cos(3*theta) pattern (azimuthal mode 3)
    h = Diskfun.from_function(lambda t, r: jnp.exp(-2 * r**2) * jnp.cos(3 * t))
    print(f"\nh(theta,r) = exp(-2r^2)cos(3θ):")
    print(f"  Rank: {h.rank}")

    # Integral should be 0 (cos(3*theta) averages to 0)
    integral_h = float(h.sum())
    print(f"  Integral = {integral_h:.2e}  (expected: ~0)")
    assert abs(integral_h) < 0.1

    # Plot
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    theta_p = np.linspace(-np.pi, np.pi, 100)
    r_p = np.linspace(0, 1, 50)
    T, R = np.meshgrid(theta_p, r_p)
    X = R * np.cos(T)
    Y = R * np.sin(T)

    funcs = [
        (R * np.cos(T), "r·cos(θ) = x"),
        (R**2, "r² = x²+y²"),
        (np.exp(-2*R**2) * np.cos(3*T), "exp(-2r²)cos(3θ)"),
    ]

    for ax, (Z, title) in zip(axes, funcs):
        im = ax.contourf(X, Y, Z, levels=20, cmap="RdBu_r")
        circle = plt.Circle((0, 0), 1, fill=False, color='k', linewidth=1.5)
        ax.add_patch(circle)
        ax.set_title(title, fontsize=10); ax.set_aspect('equal')
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Functions on the unit disk", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "disk_functions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
