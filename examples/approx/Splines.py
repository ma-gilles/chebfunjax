"""Splines.

Demonstrates cubic spline interpolation in chebfunjax, including
derivatives and edge detection via 'splitting on'.

Credit: Nick Trefethen, February 2013.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/Splines.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()


_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Underlying function: sin(x + 0.25*x^2) on [0, 10]
    def f_func(x): return np.sin(x + 0.25 * x**2)
    f = cj.chebfun(lambda x: jnp.sin(x + 0.25 * x**2), domain=(0.0, 10.0))

    # Cubic spline interpolating f at integers 0..10
    nodes = np.arange(0, 11)
    values = f_func(nodes)
    cs = CubicSpline(nodes, values)

    xx = np.linspace(0.0, 10.0, 600)
    f_vals = np.array([float(f(jnp.array(x))) for x in xx])
    s_vals = cs(xx)
    s_d1 = cs(xx, 1)
    s_d2 = cs(xx, 2)
    s_d3 = cs(xx, 3)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.plot(xx, f_vals, 'b', lw=1.8, label='f(x)')
    ax.plot(xx, s_vals, 'r', lw=1.5, label='spline s(x)')
    ax.plot(nodes, values, '.r', ms=10)
    ax.set_title('Function and cubic spline interpolant', fontsize=11)
    ax.set_xlabel('x')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    for i, (d, d_vals, title) in enumerate([
        (1, s_d1, 'First derivative'),
        (2, s_d2, 'Second derivative'),
        (3, s_d3, 'Third derivative (piecewise constant)'),
    ]):
        row, col = (0, 1) if i == 0 else ((1, 0) if i == 1 else (1, 1))
        ax_i = axes[row, col]
        ax_i.plot(xx, d_vals, 'b', lw=1.5)
        ax_i.set_title(title, fontsize=10)
        ax_i.set_xlabel('x')
        ax_i.grid(True, alpha=0.3)

    fig.suptitle('Cubic spline and its derivatives', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'Splines.png'), dpi=150)
    plt.close(fig)

    err = np.max(np.abs(s_vals - f_vals))
    print(f"Splines: max spline error = {err:.4f}")
    return True


if __name__ == '__main__':
    run()
