"""Overlap of two circles.

Computes the area of overlap between two circles using curve integrals.
Translated from geom/TwoCircles.m.

Original: https://www.chebfun.org/examples/geom/TwoCircles.html
Author: Nick Trefethen, May 2016
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    # Two unit circles: C1 centred at (0,0), C2 centred at (1,0).
    # They intersect at x = 0.5 (by symmetry).
    # Upper arcs:  y =  sqrt(1 - x^2)  and  y = sqrt(1 - (x-1)^2)
    # Lower arcs:  y = -sqrt(1 - x^2)  and  y = -sqrt(1 - (x-1)^2)

    def c1_upper(x):
        return np.sqrt(np.maximum(1.0 - x**2, 0.0))

    def c1_lower(x):
        return -c1_upper(x)

    def c2_upper(x):
        return np.sqrt(np.maximum(1.0 - (x - 1.0)**2, 0.0))

    def c2_lower(x):
        return -c2_upper(x)

    # Intersection x-coordinates: x^2 = (x-1)^2  =>  x = 0.5
    x_int = 0.5
    y_int = c1_upper(x_int)
    print(f"Intersection points: ({x_int:.4f}, ±{y_int:.4f})")

    # Overlap region bounded by x in [-0.5, 0.5] for C1 and [0.5, 1.5] for C2
    # but symmetrically we integrate from 0 to 0.5 with C1 above C2 lower, etc.
    # More cleanly: integrate width of overlap for each x in [0, 1]:
    #   left half  [0, 0.5]: min(c1_upper, c2_upper) - max(c1_lower, c2_lower)
    #   right half [0.5, 1]: same

    # Exact area of overlap for two unit circles with centre distance d=1:
    # A = 2 * arccos(d/2) - (d/2)*sqrt(4 - d^2)
    d = 1.0
    exact = 2.0 * np.arccos(d / 2.0) - (d / 2.0) * np.sqrt(4.0 - d**2)
    print(f"Exact overlap area: {exact:.10f}")

    # Numerical check via trapezoid
    xs_ov = np.linspace(0.0, 1.0, 50000)
    overlap_height = np.minimum(c1_upper(xs_ov), c2_upper(xs_ov)) \
                   - np.maximum(c1_lower(xs_ov), c2_lower(xs_ov))
    # Only valid where both circles are present
    in_c1 = xs_ov**2 <= 1.0
    in_c2 = (xs_ov - 1.0)**2 <= 1.0
    mask = in_c1 & in_c2
    area_numerical = np.trapezoid(np.where(mask, overlap_height, 0.0), xs_ov)
    print(f"Numerical overlap area: {area_numerical:.10f}")
    print(f"Error: {abs(area_numerical - exact):.2e}")
    assert abs(area_numerical - exact) < 1e-4

    # --- Plot ---
    fig, axes = plt.subplots(1, 2)

    theta = np.linspace(0, 2 * np.pi, 400)

    # Left panel: the two circles and their overlap region
    xs_fill = np.linspace(0.0, 1.0, 600)
    ys_top = np.minimum(c1_upper(xs_fill), c2_upper(xs_fill))
    ys_bot = np.maximum(c1_lower(xs_fill), c2_lower(xs_fill))
    in_both = (xs_fill**2 <= 1.0) & ((xs_fill - 1.0)**2 <= 1.0)

    axes[0].fill_between(xs_fill, np.where(in_both, ys_bot, np.nan),
                         np.where(in_both, ys_top, np.nan),
                         color='#D95319', alpha=0.4, label=f'Overlap ≈ {area_numerical:.4f}')
    axes[0].plot(np.cos(theta), np.sin(theta), color='#0072BD', linestyle='-', lw=2, label='Circle 1')
    axes[0].plot(1 + np.cos(theta), np.sin(theta), 'k-', lw=2, label='Circle 2')
    axes[0].plot([x_int, x_int], [-y_int, y_int], color='#D95319', marker='o', linestyle='none', ms=7)
    axes[0].set_aspect('equal')
    axes[0].set_xlim(-1.3, 2.3); axes[0].set_ylim(-1.3, 1.3)
    axes[0].set_title('Two unit circles, d = 1', fontsize=11)
    axes[0].legend(fontsize=9)

    # Right panel: overlap area vs centre distance d
    ds = np.linspace(0, 2, 300)
    areas = 2 * np.arccos(ds / 2) - (ds / 2) * np.sqrt(np.maximum(4 - ds**2, 0))
    axes[1].plot(ds, areas, color='#0072BD', linestyle='-', lw=2)
    axes[1].axvline(d, color='#D95319', linestyle='--', label=f'd = {d}')
    axes[1].axhline(exact, color='#D95319', linestyle=':', alpha=0.7)
    axes[1].set_title('Overlap area vs. separation', fontsize=11)
    axes[1].legend(fontsize=9)

    fig.suptitle('Area of Overlap of Two Unit Circles', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'two_circles.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("two_circles: done")
    return True

if __name__ == "__main__":
    run()
