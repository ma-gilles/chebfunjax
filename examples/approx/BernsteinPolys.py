"""Bernstein polynomials.

Bernstein polynomial approximations converge to any continuous function
but do so very slowly, independent of the smoothness of f.

Credit: Nick Trefethen, May 2012.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/BernsteinPolys.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from math import comb
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')

def bernstein(f_vals_fn, n, x_pts):
    """Evaluate degree-n Bernstein polynomial at x_pts."""
    result = np.zeros_like(x_pts)
    for k in range(n + 1):
        coeff = comb(n, k)
        result += f_vals_fn(k / n) * coeff * x_pts**k * (1 - x_pts)**(n - k)
    return result

def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Define f: min(|s-0.3|, 2|s-0.7|) + bump
    def f_func(s):
        s = np.asarray(s)
        base = np.minimum(np.abs(s - 0.3), 2 * np.abs(s - 0.7))
        return s + np.maximum(0.0, 1.0 - 5 * base)

    xx = np.linspace(0.0, 1.0, 300)
    f_vals = f_func(xx)

    fig, axes = plt.subplots(1, 3)
    for i, n in enumerate([25, 50, 100]):
        ax = axes[i]
        Bn_vals = bernstein(f_func, n, xx)
        ax.plot(xx, f_vals, 'b', lw=1.8, label='f(x)')
        ax.plot(xx, Bn_vals, 'r', lw=1.5, label=f'B_{n}(x)')
        err = np.max(np.abs(Bn_vals - f_vals))
        ax.set_title(f'n = {n}, max err = {err:.3f}', fontsize=10)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    fig.suptitle('Bernstein polynomial approximations (slow convergence)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'BernsteinPolys.png'), dpi=150)
    plt.close(fig)

    # Also show smooth function: Bernstein is equally slow
    def f_smooth(s):
        s = np.asarray(s)
        return s + np.exp(-50 * (s - 0.3)**2) + np.exp(-200 * (s - 0.7)**2)

    f_smooth_vals = f_smooth(xx)
    fig2, axes2 = plt.subplots(1, 3)
    for i, n in enumerate([25, 50, 100]):
        ax = axes2[i]
        Bn_vals = bernstein(f_smooth, n, xx)
        ax.plot(xx, f_smooth_vals, 'b', lw=1.8, label='f smooth')
        ax.plot(xx, Bn_vals, 'r', lw=1.5, label=f'B_{n}')
        err = np.max(np.abs(Bn_vals - f_smooth_vals))
        ax.set_title(f'n = {n}, max err = {err:.3f}', fontsize=10)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    fig2.suptitle('Bernstein approx of smooth function (no improvement!)', fontsize=11)
    fig2.tight_layout()
    fig2.savefig(os.path.join(_OUTDIR, 'BernsteinPolys_smooth.png'), dpi=150)
    plt.close(fig2)

    print("BernsteinPolys: done.")
    return True

if __name__ == '__main__':
    run()
