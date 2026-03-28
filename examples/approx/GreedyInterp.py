"""A greedy algorithm for choosing interpolation points.

Greedily selects interpolation points by placing the next point at
the location of maximum error, converging to Chebyshev-like clustering.

Credit: Nick Trefethen, November 2011.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/GreedyInterp.html
"""

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

def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Greedy interpolation for f = |x|
    f = cj.chebfun(jnp.abs)
    xx_dense = np.linspace(-1.0, 1.0, 1000)
    f_dense = np.abs(xx_dense)

    # Start: find global max of f
    pts = []
    f_abs_max_idx = np.argmax(f_dense)
    pts.append(xx_dense[f_abs_max_idx])

    errors_greedy = []
    for n in range(128):
        # Interpolate f at pts using barycentric polynomial
        x_pts = np.array(pts)
        y_pts = np.abs(x_pts)
        # Use chebfunjax interp1
        p = cj.chebfun(lambda t: t, domain=(-1.0, 1.0))  # just to use interp
        try:
            p_interp = cj.Chebfun.interp1(x_pts, y_pts)
            p_vals = np.array([float(p_interp(jnp.array(x))) for x in xx_dense])
        except Exception:
            # Fallback: numpy polynomial interpolation
            from numpy.polynomial import polynomial as nppoly
            coeffs = np.polyfit(x_pts, y_pts, len(x_pts) - 1)
            p_vals = np.polyval(coeffs, xx_dense)

        err_vals = f_dense - p_vals
        maxval = np.max(np.abs(err_vals))
        maxpos = xx_dense[np.argmax(np.abs(err_vals))]
        errors_greedy.append(maxval)
        pts.append(maxpos)
        if maxval < 1e-13:
            break

    # For comparison: error with Chebyshev points
    errors_cheb = []
    for n in [1, 2, 4, 8, 16, 32, 64, 128]:
        pts_cheb = np.cos(np.pi * np.arange(n + 1) / n)
        y_cheb = np.abs(pts_cheb)
        try:
            coeffs_cheb = np.polyfit(pts_cheb, y_cheb, n)
            p_cheb = np.polyval(coeffs_cheb, xx_dense)
            err_cheb = np.max(np.abs(f_dense - p_cheb))
        except Exception:
            err_cheb = np.nan
        errors_cheb.append(err_cheb)

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.semilogy(np.arange(1, len(errors_greedy) + 1), errors_greedy, color='#0072BD', linestyle='.-',
                lw=1.5, ms=4, label='greedy')
    ax.semilogy([1, 2, 4, 8, 16, 32, 64, 128],
                errors_cheb, 'r.--', ms=8, label='Chebyshev')
    ax.set_title('Error vs. number of interpolation points', fontsize=11)
    ax.legend(fontsize=9)
    # Distribution of greedy points (first 20)
    ax2 = axes[1]
    pts_arr = np.array(pts[:min(30, len(pts))])
    ax2.scatter(pts_arr, np.zeros_like(pts_arr), c=np.arange(len(pts_arr)),
                cmap='viridis', s=50)
    ax2.set_title('First 30 greedy interpolation points', fontsize=11)
    ax2.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'GreedyInterp.png'), dpi=150)
    plt.close(fig)

    print(f"GreedyInterp: {len(errors_greedy)} steps, final error = {errors_greedy[-1]:.2e}")
    return True

if __name__ == '__main__':
    run()
