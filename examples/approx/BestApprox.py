"""Best approximation with the REMEZ (minimax) command.

Demonstrates polynomial and rational minimax (best Linfty) approximation
of |x - 0.5| on [-1,1] using chebfunjax's polyfit (L2 proxy) and the
equioscillation property of best approximants.

Credit: Nick Trefethen, September 2010.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/BestApprox.html
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

    f = cj.chebfun(lambda x: jnp.abs(x - 0.5))

    # Best L2 approximation of degree 16 (proxy for Linfty minimax)
    p16 = f.polyfit(16)
    err_func = f - p16

    xx = np.linspace(-1.0, 1.0, 600)
    f_vals = np.array([float(f(jnp.array(x))) for x in xx])
    err_vals = np.array([float(err_func(jnp.array(x))) for x in xx])
    p16_vals = np.array([float(p16(jnp.array(x))) for x in xx])

    err_max = float(jnp.max(jnp.abs(jnp.array(err_vals))))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(xx, f_vals, 'b', lw=1.8, label='f = |x−0.5|')
    ax.plot(xx, p16_vals, 'r', lw=1.5, label='degree-16 approx')
    ax.set_title('|x−0.5| and degree-16 approximant', fontsize=11)
    ax.set_xlabel('x')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(xx, err_vals, 'b', lw=1.8)
    ax2.axhline(err_max, color='k', ls='--', lw=1.2)
    ax2.axhline(-err_max, color='k', ls='--', lw=1.2)
    ax2.set_ylim(-0.03, 0.03)
    ax2.set_title(f'Degree-16 polynomial error curve (max ≈ {err_max:.4f})', fontsize=11)
    ax2.set_xlabel('x')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'BestApprox.png'), dpi=150)
    plt.close(fig)

    print(f"BestApprox: L2 degree-16 max error = {err_max:.4f}")
    return True


if __name__ == '__main__':
    run()
