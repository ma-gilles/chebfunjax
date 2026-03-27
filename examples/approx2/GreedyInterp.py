"""A greedy algorithm for choosing interpolation points.

Greedily selects interpolation points by placing the next point at
the location of maximum error, converging to Chebyshev-like clustering.

Credit: Nick Trefethen, November 2011.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/GreedyInterp.html
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


_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def barycentric_interp(xpts, ypts, xx):
    """Barycentric polynomial interpolation via numpy."""
    from numpy.polynomial.chebyshev import chebpts2
    n = len(xpts)
    # Use scipy's barycentric interpolation for stability
    from scipy.interpolate import BarycentricInterpolator
    bary = BarycentricInterpolator(xpts, ypts)
    return bary(xx)


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Use 200 dense points (1000 is slow with 128 iterations)
    xx_dense = np.linspace(-1.0, 1.0, 200)
    f_dense = np.abs(xx_dense)

    # Start: global max of |x| is at x = ±1
    pts = [1.0]

    errors_greedy = []
    max_steps = 40  # reduce from 128 for speed
    for n in range(max_steps):
        x_pts = np.array(pts)
        y_pts = np.abs(x_pts)
        try:
            p_vals = barycentric_interp(x_pts, y_pts, xx_dense)
        except Exception:
            from numpy.polynomial import polynomial as nppoly
            if len(x_pts) >= 2:
                coeffs = np.polyfit(x_pts, y_pts, len(x_pts) - 1)
                p_vals = np.polyval(coeffs, xx_dense)
            else:
                p_vals = np.full_like(xx_dense, y_pts[0])

        err_vals = f_dense - p_vals
        maxval = np.max(np.abs(err_vals))
        maxpos = xx_dense[np.argmax(np.abs(err_vals))]
        errors_greedy.append(maxval)
        # Avoid duplicate points
        if maxpos not in pts:
            pts.append(maxpos)
        else:
            # Perturb slightly
            pts.append(maxpos + 1e-6 * (np.random.rand() - 0.5))
        if maxval < 1e-12:
            break

    # For comparison: error with Chebyshev points (scipy barycentric)
    errors_cheb = []
    ns_cheb = [1, 2, 4, 8, 16, 32]
    for n in ns_cheb:
        pts_cheb = np.cos(np.pi * np.arange(n + 1) / n)
        y_cheb = np.abs(pts_cheb)
        try:
            p_cheb = barycentric_interp(pts_cheb, y_cheb, xx_dense)
            err_cheb = np.max(np.abs(f_dense - p_cheb))
        except Exception:
            err_cheb = np.nan
        errors_cheb.append(err_cheb)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.semilogy(np.arange(1, len(errors_greedy) + 1), errors_greedy, 'b.-',
                lw=1.5, ms=4, label='greedy')
    ax.semilogy(np.array(ns_cheb) + 1, errors_cheb, 'r.--', ms=8, label='Chebyshev')
    ax.set_title('Error vs. number of interpolation points', fontsize=11)
    ax.set_xlabel('n+1 points')
    ax.set_ylabel('max error')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Distribution of greedy points (first 30)
    ax2 = axes[1]
    pts_arr = np.array(pts[:min(30, len(pts))])
    ax2.scatter(pts_arr, np.zeros_like(pts_arr), c=np.arange(len(pts_arr)),
                cmap='viridis', s=50)
    ax2.set_title(f'First {len(pts_arr)} greedy interpolation points', fontsize=11)
    ax2.set_xlabel('x')
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'GreedyInterp.png'), dpi=150)
    plt.close(fig)

    print(f"GreedyInterp: {len(errors_greedy)} steps, final error = {errors_greedy[-1]:.2e}")
    return True


if __name__ == '__main__':
    run()
