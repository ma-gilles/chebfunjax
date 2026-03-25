"""Approximation of the checkmark function.

The checkmark function f(x) = max(x, 2x-1) on [0,1] is piecewise linear.
Best polynomial approximants exhibit equioscillation.

Credit: Nick Trefethen, January 2022.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/Checkmark.html
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

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Checkmark function: f(x) = max(x, 2x - 1) on [0, 1]
    def checkmark(x): return jnp.maximum(x, 2.0 * x - 1.0)
    f = cj.chebfun(checkmark, domain=(0.0, 1.0))

    xx = np.linspace(0.0, 1.0, 500)
    f_vals = np.array([float(f(jnp.array(x))) for x in xx])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Plot the function
    ax = axes[0]
    ax.plot(xx, f_vals, 'b', lw=1.8)
    ax.set_title('The checkmark function', fontsize=11)
    ax.set_xlabel('x')
    ax.grid(True, alpha=0.3)

    # Best L2 polynomial approximants of various degrees
    colors = ['r', 'g', 'm']
    for i, deg in enumerate([5, 10, 20]):
        pn = f.polyfit(deg)
        err = f - pn
        err_vals = np.array([float(err(jnp.array(x))) for x in xx])
        pn_vals = np.array([float(pn(jnp.array(x))) for x in xx])
        ax.plot(xx, pn_vals, '--', color=colors[i], lw=1.2, label=f'deg {deg}')

        ax2 = axes[1]
        ax2.plot(xx, err_vals, color=colors[i], lw=1.5, label=f'deg {deg}')

    ax.legend(fontsize=8)
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_title('Errors of polynomial approximants', fontsize=11)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Chebyshev coefficient decay
    coeffs = np.abs(np.array(f.coeffs)) + 1e-18
    axes[2].semilogy(np.arange(len(coeffs)), coeffs, 'b.', ms=4)
    axes[2].set_title('Chebyshev coefficients of checkmark', fontsize=11)
    axes[2].set_xlabel('degree')
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'Checkmark.png'), dpi=150)
    plt.close(fig)

    print("Checkmark: done.")
    return True


if __name__ == '__main__':
    run()
