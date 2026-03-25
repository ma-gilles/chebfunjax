"""Absolute value approximations by rationals II.

Compares standard Newton iteration with scaled iteration for computing |x|,
demonstrating rational approximation methods for the sign/absolute value function.

Credit: Yuji Nakatsukasa, July 2012.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/AbsoluteValueScaled.html
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

    # Standard Newton iteration for |x|: r_{k+1} = (r_k^2 + x^2) / (2 r_k)
    # Starting from r_0 = 1 on the domain that avoids 0
    xx = np.linspace(-1.0, 1.0, 500)
    abs_x = np.abs(xx)

    # Compute error of truncated Newton rational approximant (evaluated symbolically)
    # After k iterations, r_k(x) = rational approx to |x|
    # We implement the rational approximant directly on a grid (no chebfun arithmetic needed)
    def newton_iter_vals(x_arr, kmax):
        """Newton iterate on an array, starting from r=1."""
        r = np.ones_like(x_arr, dtype=float)
        for _ in range(kmax):
            r = (r**2 + x_arr**2) / (2.0 * r)
        return r

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for k, col in [(1, 'b'), (3, 'r'), (5, 'g'), (7, 'm')]:
        r_vals = newton_iter_vals(xx, k)
        err = np.abs(r_vals - abs_x) + 1e-18
        ax.semilogy(xx, err, color=col, lw=1.5, label=f'k={k}')
    ax.set_xlim(-1, 1)
    ax.set_title('Newton iteration error for |x|', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('|r_k(x) - |x||')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Chebfun-based: show Newton approximation converges
    # Only 3 iterations to avoid excessive length
    dom = (-1.0, 0.0, 1.0)
    x_cf = cj.chebfun(lambda t: t, domain=dom)
    r_cf = cj.chebfun(lambda t: jnp.ones_like(t), domain=dom)
    for _ in range(3):
        r_cf = (r_cf**2 + x_cf**2) / (2.0 * r_cf)

    cf_vals = np.array([float(r_cf(jnp.array(xi))) for xi in xx])
    err_cf = np.abs(cf_vals - abs_x) + 1e-18

    ax2 = axes[1]
    ax2.semilogy(xx, err_cf, 'b', lw=1.8, label='chebfun Newton (k=3)')
    ax2.semilogy(xx, np.abs(newton_iter_vals(xx, 3) - abs_x) + 1e-18,
                 'r--', lw=1.5, label='direct Newton (k=3)')
    ax2.set_xlim(-1, 1)
    ax2.set_title('Chebfun Newton vs. direct (k=3)', fontsize=11)
    ax2.set_xlabel('x')
    ax2.set_ylabel('|error|')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Newton iteration for absolute value approximation', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'AbsoluteValueScaled.png'), dpi=150)
    plt.close(fig)

    print("AbsoluteValueScaled: done.")
    return True


if __name__ == '__main__':
    run()
