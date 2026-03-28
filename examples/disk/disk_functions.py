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

    # Plot each disk function using the library's plot_disk
    from chebfunjax.plotting import plot_disk

    _here = os.path.dirname(os.path.abspath(__file__))

    disk_funcs = [
        (f, "r cos(theta) = x", "disk_f1"),
        (g, "r^2 = x^2 + y^2", "disk_f2"),
        (h, "exp(-2r^2) cos(3 theta)", "disk_f3"),
    ]
    for df, title, tag in disk_funcs:
        fig_3d, ax_3d = plot_disk(df, title=title, mode="3d")
        fig_3d.savefig(os.path.join(_here, f"{tag}_3d.png"),
                       dpi=150, bbox_inches="tight")
        plt.close(fig_3d)

        fig_2d, ax_2d = plot_disk(df, title=title, mode="2d")
        fig_2d.savefig(os.path.join(_here, f"{tag}_2d.png"),
                       dpi=150, bbox_inches="tight")
        plt.close(fig_2d)

    # Combined overview figure (2D flat mode, matching MATLAB style)
    from chebfunjax.plotting import PARULA
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    titles = ["r cos(theta) = x", "r^2 = x^2 + y^2",
              "exp(-2r^2) cos(3 theta)"]
    for ax, df, title in zip(axes, [f, g, h], titles):
        plot_disk(df, ax=ax, title=title, mode="2d")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "disk_functions.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
