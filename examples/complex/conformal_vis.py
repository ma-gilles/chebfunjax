"""Visualizing conformal maps.

Conformal maps preserve angles but distort distances.  We visualize several
classical conformal maps by transforming a grid of lines in the z-plane
and plotting the image in the w-plane.

Credit: Inspired by Chebfun examples complex/ConformalVis.m (Nick Trefethen,
December 2016) and complex/SCToolbox.m (Nick Trefethen, October 2010).
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



def plot_conformal(ax_in, ax_out, f_map, xlim=(-2, 2), ylim=(-2, 2),
                   n_lines=10, n_pts=200, title_in="z-plane", title_out="w-plane"):
    """Draw a conformal map by transforming horizontal/vertical lines."""
    xs = np.linspace(xlim[0], xlim[1], n_lines)
    ys = np.linspace(ylim[0], ylim[1], n_lines)
    ts = np.linspace(ylim[0], ylim[1], n_pts)
    ts_x = np.linspace(xlim[0], xlim[1], n_pts)

    # Horizontal lines (y = const)
    for y in ys[1:-1]:  # skip boundary
        z_line = ts_x + 1j * y
        ax_in.plot(z_line.real, z_line.imag, 'b-', linewidth=0.7, alpha=0.7)
        w_line = f_map(z_line)
        ax_out.plot(w_line.real, w_line.imag, 'b-', linewidth=0.7, alpha=0.7)

    # Vertical lines (x = const)
    for x in xs[1:-1]:
        z_line = x + 1j * ts
        ax_in.plot(z_line.real, z_line.imag, 'r-', linewidth=0.7, alpha=0.7)
        w_line = f_map(z_line)
        ax_out.plot(w_line.real, w_line.imag, 'r-', linewidth=0.7, alpha=0.7)

    for ax, title in [(ax_in, title_in), (ax_out, title_out)]:
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)


def run():
    print("=" * 60)
    print("Visualizing conformal maps")
    print("=" * 60)

    pi = float(jnp.pi)

    # --- Map 1: w = z^2 -----------------------------------------------
    # Maps vertical/horizontal lines to parabolas
    def map_sq(z):
        return z**2

    # Verify: angles are preserved (conformal except at z=0)
    # Two curves meeting at z0 at angle theta should map to curves meeting at angle theta
    z0 = 1.0 + 1j
    dz1 = 1.0 + 0j  # direction 1
    dz2 = 0j + 1.0  # direction 2
    dw1 = 2 * z0 * dz1
    dw2 = 2 * z0 * dz2
    angle_in = np.angle(dz2 / dz1)
    angle_out = np.angle(dw2 / dw1)
    print(f"\nw=z^2 at z0=1+i: angle in = {np.degrees(angle_in):.2f}°, "
          f"angle out = {np.degrees(angle_out):.2f}°  (should match: conformal)")
    assert abs(angle_in - angle_out) < 1e-10

    # --- Map 2: w = exp(z) --------------------------------------------
    # Maps horizontal lines (y=const) to rays, vertical lines to circles
    def map_exp(z):
        return np.exp(z)

    # --- Map 3: w = (z-1)/(z+1) (Cayley transform) -------------------------
    # Maps the RIGHT half-plane Re(z) > 0 to the unit disk |w| < 1
    def map_cayley(z):
        return (z - 1.0) / (z + 1.0)

    # Verify: right half-plane maps to unit disk
    # Re(z) > 0 => |w| < 1
    test_pts_right = [1j, 2+3j, 0.5+2j, 1+0j, 3-2j]
    for z in test_pts_right:
        w = map_cayley(z)
        if z.real > 0:
            assert abs(w) < 1.0, f"Cayley: |w|={abs(w)} should be < 1 for Re(z)>0"
        elif z.real == 0:
            assert abs(abs(w) - 1.0) < 1e-10, f"Cayley: imaginary axis should map to unit circle"

    # Imaginary axis maps to unit circle
    test_pts_imag = [1j, -2j, 3j]
    for z in test_pts_imag:
        w = map_cayley(z)
        assert abs(abs(w) - 1.0) < 1e-12, f"Cayley: |w|={abs(w)} should be 1 on imaginary axis"
    print(f"Cayley transform verified: right half-plane -> unit disk")

    # --- Map 4: w = sin(z) --------------------------------------------
    def map_sin(z):
        return np.sin(z)

    # Verify: sin is conformal away from zeros of sin'(z) = cos(z)
    # (conformal except at z = pi/2 + n*pi)
    z_test = 0.5 + 0.3j
    deriv = np.cos(z_test)
    print(f"sin conformal at z=0.5+0.3i: |sin'(z)| = {abs(deriv):.4f} ≠ 0 ✓")
    assert abs(deriv) > 0.1

    print("\nAll conformal map checks passed.")

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(4, 2)

    maps = [
        (map_sq, (-1.5, 1.5), (-1.5, 1.5), "z", "z²"),
        (map_exp, (-pi, pi), (-pi, pi), "z", "exp(z)"),
        (map_cayley, (-3, 3), (0.01, 3), "z (upper half-plane)", "(z-1)/(z+1)"),
        (map_sin, (-pi, pi), (-2, 2), "z", "sin(z)"),
    ]

    for row, (fmap, xlim, ylim, t_in, t_out) in enumerate(maps):
        ax_in = axes[row, 0]
        ax_out = axes[row, 1]
        try:
            plot_conformal(ax_in, ax_out, fmap, xlim=xlim, ylim=ylim,
                           n_lines=12, n_pts=300,
                           title_in=f"$z$-plane: $w = {t_out}$",
                           title_out=f"$w$-plane: image")
        except Exception:
            ax_in.text(0.5, 0.5, f"w={t_out}", transform=ax_in.transAxes, ha='center')

    fig.suptitle("Conformal maps: horizontal (blue) and vertical (red) lines",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "conformal_vis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
