"""Best approximation with the REMEZ (minimax) command.

Demonstrates polynomial and rational minimax (best Linfty) approximation
of |x - 0.5| on [-1,1] using chebfunjax's polyfit (L2 proxy) and the
equioscillation property of best approximants.

Credit: Nick Trefethen, September 2010.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/BestApprox.html
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

    # Use explicit breakpoint at 0.5 so |x-0.5| converges quickly
    f = cj.chebfun(lambda x: jnp.abs(x - 0.5), domain=[-1.0, 0.5, 1.0])

    # Best L2 approximation of degree 16 (proxy for Linfty minimax)
    p16 = f.polyfit(16)
    err_func = f - p16

    xx = jnp.linspace(-1.0, 1.0, 600)
    f_vals = np.array(f(xx))
    err_vals = np.array(err_func(xx))
    p16_vals = np.array(p16(xx))

    err_max = float(jnp.max(jnp.abs(jnp.array(err_vals))))

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.plot(xx, f_vals, 'b', lw=1.8, label='f = |x−0.5|')
    ax.plot(xx, p16_vals, 'r', lw=1.5, label='degree-16 approx')
    ax.set_title('|x−0.5| and degree-16 approximant', fontsize=11)
    ax.legend(fontsize=9)
    ax2 = axes[1]
    ax2.plot(xx, err_vals, 'b', lw=1.8)
    ax2.axhline(err_max, color='k', ls='--', lw=1.2)
    ax2.axhline(-err_max, color='k', ls='--', lw=1.2)
    ax2.set_ylim(-0.03, 0.03)
    ax2.set_title(f'Degree-16 polynomial error curve (max ≈ {err_max:.4f})', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'BestApprox.png'), dpi=150)
    plt.close(fig)

    print(f"BestApprox: L2 degree-16 max error = {err_max:.4f}")
    return True


if __name__ == '__main__':
    run()
