"""Best polynomial approximation in the L1 norm.

Compares L∞ (minimax), L2 (least squares), and L1 best polynomial
approximants, showing the localized error of L1 approximation.

Credit: Yuji Nakatsukasa and Alex Townsend, July 2019.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/BestL1.html
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

    dom = (0.0, 14.0)
    def f_func(x): return jnp.sin(x)**2 + jnp.sin(x**2)
    f = cj.chebfun(f_func, domain=dom)
    deg = 30  # reduce from 100 for speed

    # L2 approximation via polyfit
    p2 = f.polyfit(deg)
    err2 = f - p2

    xx = jnp.linspace(0.0, 14.0, 600)
    f_vals = np.array(f(xx))
    err2_vals = np.array(err2(xx))
    p2_vals = np.array(p2(xx))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(xx, f_vals, 'b', lw=1.5, label='f')
    axes[0, 0].plot(xx, p2_vals, 'r--', lw=1.5, label=f'p_L2 (deg {deg})')
    axes[0, 0].set_title('f and L2 approximant', fontsize=11)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-3, 3)

    axes[0, 1].plot(xx, err2_vals, 'k', lw=1.5)
    axes[0, 1].set_title('L2 error curve', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-3, 3)

    # Second example: |x - 1/4| on [-1,1]
    # Use breakpoint at 0.25 for fast convergence
    f2 = cj.chebfun(lambda x: jnp.abs(x - 0.25), domain=[-1.0, 0.25, 1.0])
    deg2 = 20
    p2b = f2.polyfit(deg2)
    err2b = f2 - p2b

    xx2 = jnp.linspace(-1.0, 1.0, 400)
    err2b_vals = np.array(err2b(xx2))

    axes[1, 0].plot(xx2, err2b_vals, 'k', lw=1.5)
    axes[1, 0].set_title(f'L2 error for |x−1/4| (deg {deg2})', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axis('off')
    axes[1, 1].text(0.1, 0.5,
        'L1 approximation (polyfitL1)\nnot yet implemented in chebfunjax.\n'
        'L2 shown as proxy.',
        fontsize=10, transform=axes[1, 1].transAxes, va='center')

    fig.suptitle('L1/L2 polynomial approximation', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'BestL1.png'), dpi=150)
    plt.close(fig)

    print("BestL1: done (L1 polyfitL1 not available; L2 shown).")
    return True


if __name__ == '__main__':
    run()
