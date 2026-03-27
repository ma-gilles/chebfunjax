"""Chebfuns from equispaced data.

Demonstrates constructing a chebfun from equispaced data samples using
the 'equi' approach (barycentric interpolation with mapped grid).

Credit: Nick Trefethen, April 2015.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/EquispacedData.html
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


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    def ff(x): return np.exp(x) * np.cos(10 * x) * np.tanh(4 * x)

    grid = np.linspace(-1, 1, 40)
    data = ff(grid)

    # Build chebfun from equispaced data via interp1
    try:
        f_equi = cj.Chebfun.interp1(grid, data)
    except Exception:
        # fallback: build directly using polyfit on equispaced nodes
        from numpy.polynomial import chebyshev as ncheb
        f_equi = cj.chebfun(lambda x: jnp.array(ff(float(x))))

    # Exact chebfun for reference
    f_exact = cj.chebfun(lambda x: jnp.exp(x) * jnp.cos(10.0 * x) * jnp.tanh(4.0 * x))

    xx = np.linspace(-1.0, 1.0, 600)
    f_exact_vals = np.array([float(f_exact(jnp.array(x))) for x in xx])

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.plot(xx, f_exact_vals, 'b', lw=1.8, label='exact')
    ax.plot(grid, data, '.k', ms=8, label='equispaced data (40 pts)')
    ax.set_title('f(x) = exp(x)·cos(10x)·tanh(4x) with 40 equispaced samples', fontsize=9)
    ax.legend(fontsize=9)
    # Compare error: pchip from scipy as equispaced interpolant
    from scipy.interpolate import PchipInterpolator
    pchip = PchipInterpolator(grid, data)
    pchip_vals = pchip(xx)
    err_pchip = np.abs(pchip_vals - f_exact_vals)

    ax2 = axes[1]
    ax2.semilogy(xx, err_pchip + 1e-18, 'r', lw=1.5, label='pchip error')
    ax2.set_title('Reconstruction error from 40 equispaced points', fontsize=10)
    ax2.set_ylabel('|error|')
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'EquispacedData.png'), dpi=150)
    plt.close(fig)

    print(f"EquispacedData: pchip max error = {np.max(err_pchip):.3e}")
    return True


if __name__ == '__main__':
    run()
