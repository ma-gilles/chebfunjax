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


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    # Circle 1: quarter-circle of radius 1 about (-1, 1)
    # Upper arc: y = 1 + sqrt(1 - (x+1)^2) restricted to visible region
    # Circle 2: quarter-circle of radius 2 about (1, -1)
    # Upper arc: y = -1 + sqrt(4 - (x-1)^2)

    def big_circle(x):
        """y = -1 + sqrt(4 - (x-1)^2), radius=2, center=(1,-1)."""
        val = 4 - (x - 1)**2
        if np.isscalar(val):
            return -1 + np.sqrt(max(val, 0))
        return -1 + np.sqrt(np.maximum(val, 0))

    def little_circle(x):
        """y = 2 - sqrt(1 - (x+1)^2), radius=1, center=(-1,1)."""
        val = 1 - (x + 1)**2
        if np.isscalar(val):
            return 2 - np.sqrt(max(val, 0))
        return 2 - np.sqrt(np.maximum(val, 0))

    # Find intersection points on x in [-1, 0]
    diff_fn = lambda x: big_circle(x) - little_circle(x)
    x1 = brentq(diff_fn, -1.0, -0.5)
    x2 = brentq(diff_fn, -0.5, 0.0)
    y1 = big_circle(x1)
    y2 = big_circle(x2)
    print(f"Intersection points: ({x1:.6f}, {y1:.6f}) and ({x2:.6f}, {y2:.6f})")

    # Area of overlap = integral_{x1}^{x2} [big_circle(x) - little_circle(x)] dx
    xs_int = np.linspace(x1, x2, 10000)
    big_vals = np.array([big_circle(xi) for xi in xs_int])
    little_vals = np.array([little_circle(xi) for xi in xs_int])
    area_numerical = np.trapezoid(big_vals - little_vals, xs_int)

    # Exact answer from Professor Povey
    exact = np.arccos(5 * np.sqrt(2) / 8) + 4 * np.arccos(11 * np.sqrt(2) / 16) - np.sqrt(7) / 2
    print(f"Area of overlap:")
    print(f"  Numerical = {area_numerical:.10f}")
    print(f"  Exact     = {exact:.10f}")
    print(f"  Error     = {abs(area_numerical - exact):.2e}")
    assert abs(area_numerical - exact) < 1e-4

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Draw the region
    xs_big = np.linspace(-1, 1, 500)
    ys_big = np.array([big_circle(xi) for xi in xs_big])
    xs_little = np.linspace(-1, 0, 500)
    ys_little = np.array([little_circle(xi) for xi in xs_little])

    # Bounding box
    axes[0].plot([-1, 1, 1, -1, -1], [0, 0, 2, 2, 0], 'k-', linewidth=1.5)

    # Fill overlap region
    xs_fill = np.linspace(x1, x2, 300)
    axes[0].fill_between(xs_fill,
                         [little_circle(xi) for xi in xs_fill],
                         [big_circle(xi) for xi in xs_fill],
                         color='red', alpha=0.5, label=f'Overlap area={area_numerical:.4f}')
    axes[0].plot(xs_big, ys_big, 'k-', linewidth=2, label='Big circle (r=2)')
    axes[0].plot(xs_little, ys_little, 'k--', linewidth=2, label='Little circle (r=1)')
    axes[0].plot([x1, x2], [y1, y2], '.b', markersize=10, label='Intersections')
    axes[0].set_xlim(-1, 1); axes[0].set_ylim(0, 2)
    axes[0].set_aspect('equal')
    axes[0].set_title('Two overlapping circles', fontsize=11)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Plot area integrand
    axes[1].plot(xs_int, big_vals - little_vals, 'r-', linewidth=2)
    axes[1].fill_between(xs_int, big_vals - little_vals, alpha=0.3, color='red')
    axes[1].axvline(x1, color='b', linestyle='--', alpha=0.7)
    axes[1].axvline(x2, color='b', linestyle='--', alpha=0.7)
    axes[1].set_title(f'big_circle - little_circle\n'
                      f'Integral = {area_numerical:.6f} (exact = {exact:.6f})', fontsize=10)
    axes[1].set_xlabel('x'); axes[1].grid(True, alpha=0.3)

    fig.suptitle('Area of Overlap of Two Circles', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'two_circles.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("two_circles: done")
    return True


if __name__ == "__main__":
    run()
