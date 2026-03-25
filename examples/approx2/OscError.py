"""Approximations and oscillation of error.

Compares L∞ (interpolation), L2 (polyfit), and exact representations of a
continuous function, showing how errors oscillate differently.

Credit: Mohsin Javed, October 2013.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/OscError.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Continuous function
    def f_func(x): return jnp.exp(x) + jnp.cos(5.0 * x)
    f = cj.chebfun(f_func)
    n = 20

    # Chebfun interpolant (essentially exact)
    f_approx = cj.chebfun(f_func, n=n)

    # L2 polyfit
    p_L2 = f.polyfit(n)

    # Chebyshev interpolation in n+1 Chebpts (n-degree polynomial interpolant)
    cheb_nodes = np.cos(np.pi * np.arange(n + 1) / n)
    y_nodes = np.array([float(f(jnp.array(x))) for x in cheb_nodes])
    coeffs_interp = np.polyfit(cheb_nodes, y_nodes, n)

    xx = np.linspace(-1.0, 1.0, 500)
    f_true = np.array([float(f(jnp.array(x))) for x in xx])
    f_interp_vals = np.polyval(coeffs_interp, xx)
    f_L2_vals = np.array([float(p_L2(jnp.array(x))) for x in xx])
    f_approx_vals = np.array([float(f_approx(jnp.array(x))) for x in xx])

    err_interp = f_interp_vals - f_true
    err_L2 = f_L2_vals - f_true

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    ax.plot(xx, f_true, 'b', lw=1.8, label='f(x)')
    ax.plot(xx, f_interp_vals, 'r--', lw=1.3, label=f'Cheb interp n={n}')
    ax.set_title(f'Function and degree-{n} interpolant', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(xx, err_interp, 'r', lw=1.5, label='interp error')
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_title('Interpolation error (equioscillates)', fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(xx, err_L2, 'b', lw=1.5, label='L2 error')
    ax3.axhline(0, color='k', lw=0.5)
    ax3.set_title('L2 error (smaller near ends)', fontsize=10)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f'Error oscillation: interpolation vs. L2 (n={n})', fontsize=12)
    for ax in axes:
        ax.set_xlabel('x')

    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'OscError.png'), dpi=150)
    plt.close(fig)

    print(f"OscError: interp max err={np.max(np.abs(err_interp)):.3e}, "
          f"L2 max err={np.max(np.abs(err_L2)):.3e}")
    return True


if __name__ == '__main__':
    run()
