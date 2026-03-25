"""Polynomial level curve of constant width.

Demonstrates a degree-8 polynomial whose zero set is a curve of constant
width, like the British 50p coin. Translated from geom/ConstantWidth.m.

Original: https://www.chebfun.org/examples/geom/ConstantWidth.html
Author: Nick Trefethen, May 2022
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from matplotlib.patches import Polygon
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def constant_width_poly(x, y):
    """The Rabinowitz polynomial whose zero set has constant width."""
    r2 = x**2 + y**2
    xy_term = x**2 - 3 * y**2
    return (r2**4 - 45 * r2**3 - 41283 * r2**2
            + 7950960 * r2
            + 16 * xy_term**3
            + 48 * r2 * xy_term**2
            + x * xy_term * (16 * r2**2 - 5544 * r2 + 266382)
            - 373248000)


def extract_contour_points(X, Y, Z, level=0):
    """Extract contour points at given level."""
    import matplotlib.pyplot as plt
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(X, Y, Z, levels=[level])
    points = []
    for path in cs.allsegs[0]:
        points.append(path)
    plt.close(fig_tmp)
    return points


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    # Evaluate polynomial on grid
    n_grid = 400
    xs = np.linspace(-12, 12, n_grid)
    ys = np.linspace(-12, 12, n_grid)
    X, Y = np.meshgrid(xs, ys)
    Z = constant_width_poly(X, Y)

    # Extract zero contour
    contour_pts = extract_contour_points(X, Y, Z, level=0)
    print(f"Number of contour segments: {len(contour_pts)}")

    # Compute widths in different directions
    if contour_pts:
        all_pts = np.vstack(contour_pts)
        all_complex = all_pts[:, 0] + 1j * all_pts[:, 1]

        print("\nWidth measurements in different directions:")
        print(f"{'theta/pi':>10} {'width':>12}")
        for frac in [0, 1/5, 2/5, 3/5, 4/5]:
            theta = np.pi * frac
            a = np.exp(1j * theta)
            proj = np.real(a * all_complex)
            width = np.max(proj) - np.min(proj)
            print(f"{frac:>10.4f} {width:>12.6f}")
    else:
        print("No contour found - using x-axis verification")

    # Verify on x-axis: p(x, 0) = 0 at x=-8 and x=10
    def poly_x_axis(x):
        return (x**8 + 16 * x**7 + 19 * x**6 - 5544 * x**5
                - 41283 * x**4 + 266382 * x**3 + 7950960 * x**2 - 373248000)

    print(f"\np(-8, 0) = {poly_x_axis(-8):.2e}  (should be ~0)")
    print(f"p(10, 0) = {poly_x_axis(10):.2e}  (should be ~0)")
    print(f"Exact width should be 18 = 10 - (-8)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Contour fill
    im = axes[0].contourf(X, Y, Z, levels=[-1e10, 0], colors=['#b87333'], alpha=0.9)
    axes[0].contour(X, Y, Z, levels=[0], colors=['k'], linewidths=2)
    axes[0].set_aspect('equal')
    axes[0].set_xlim(-12, 12); axes[0].set_ylim(-12, 12)
    axes[0].set_title('Polynomial curve of constant width\n(Rabinowitz, 1997)', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # X-axis cross section
    xs_1d = np.linspace(-12, 12, 500)
    p_vals = poly_x_axis(xs_1d)
    axes[1].plot(xs_1d, p_vals, 'b-', linewidth=2)
    axes[1].axhline(0, color='k', linestyle='--', linewidth=1)
    axes[1].axvline(-8, color='r', linestyle='--', linewidth=1.5, label='x=-8')
    axes[1].axvline(10, color='g', linestyle='--', linewidth=1.5, label='x=10')
    axes[1].set_title('p(x,0) on x-axis\nZeros at x=-8 and x=10 (width=18)', fontsize=10)
    axes[1].set_xlabel('x'); axes[1].set_ylim(-1e9, 1e9)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    fig.suptitle('Polynomial Curve of Constant Width', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'constant_width.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("constant_width: done")
    return True


if __name__ == "__main__":
    run()
