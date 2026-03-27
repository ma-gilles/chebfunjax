"""Rational interpolation, robust and non-robust.

Compares standard rational interpolation (fragile) with the robust
least-squares rational interpolation and AAA approximation.

Credit: Nick Trefethen, August 2011.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/RationalInterp.html
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

from chebfunjax.utils.aaa import aaa

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Function with near-pole: f(x) = 1/(1 + 25*x^2) (poles at ±0.2i)
    def f_func(x): return 1.0 / (1.0 + 25.0 * x**2)
    f = cj.chebfun(lambda x: jnp.array(1.0 / (1.0 + 25.0 * x**2)))

    xx = np.linspace(-1.0, 1.0, 500)
    f_true = 1.0 / (1.0 + 25.0 * xx**2)

    # AAA rational approximation
    xs_aaa = jnp.linspace(-1.0, 1.0, 300)
    ys_aaa = jnp.array([f_func(x) for x in xs_aaa])
    r_aaa, pol, res, zer, *_ = aaa(ys_aaa, xs_aaa)
    r_vals = np.array([float(r_aaa(jnp.array(x)).real) for x in xx])
    err_aaa = np.abs(r_vals - f_true)

    # Polynomial approximation for comparison
    p30 = f.polyfit(30)
    p30_vals = np.array([float(p30(jnp.array(x))) for x in xx])
    err_poly = np.abs(p30_vals - f_true)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    ax.plot(xx, f_true, 'b', lw=1.8, label='f(x) = 1/(1+25x²)')
    ax.plot(xx, r_vals, 'r--', lw=1.5, label=f'AAA ({len(pol)} poles)')
    ax.set_title('Runge function and AAA approximant', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.semilogy(xx, err_aaa + 1e-18, 'r', lw=1.5, label='AAA error')
    ax2.semilogy(xx, err_poly + 1e-18, 'b', lw=1.5, label='poly deg-30')
    ax2.set_title('Approximation errors', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Poles of AAA approximant
    ax3 = axes[2]
    pol_arr = np.array([complex(p) for p in pol])
    ax3.plot(pol_arr.real, pol_arr.imag, '.r', ms=10, label='AAA poles')
    ax3.plot([0], [0.2], 'kx', ms=12, label='true poles ±0.2i')
    ax3.plot([0], [-0.2], 'kx', ms=12)
    ax3.set_title('Poles of AAA approximant', fontsize=10)
    ax3.set_xlabel('Re(z)')
    ax3.set_ylabel('Im(z)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='k', lw=0.5)
    ax3.axvline(0, color='k', lw=0.5)

    for ax in axes:
        ax.set_xlabel('x')

    fig.suptitle('Rational interpolation of the Runge function', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'RationalInterp.png'), dpi=150)
    plt.close(fig)

    print(f"RationalInterp: AAA err={np.max(err_aaa):.3e}, "
          f"{len(pol)} poles found")
    return True


if __name__ == '__main__':
    run()
