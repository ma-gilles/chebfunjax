"""Absolute value approximations by rationals.

Newton's method applied to r² = x² starting from r=1 generates a sequence
of rational functions of type (2^k, 2^k) converging to |x|.

Credit: Nick Trefethen, May 2011.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/AbsoluteValue.html
"""

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

    # Newton iteration r := (r^2 + x^2) / (2r) starting from r=1
    # on piecewise domain [-1,0,1] to keep lengths manageable
    dom = (-1.0, 0.0, 1.0)
    x = cj.chebfun(lambda t: t, domain=dom)
    r = cj.chebfun(lambda t: jnp.ones_like(t), domain=dom)

    xx = np.linspace(-1.0, 1.0, 500)
    abs_x = np.abs(xx)

    fig, axes = plt.subplots(2, 3)
    axes = axes.flatten()

    print(f"{'Step':>4}  {'||r - |x|||_inf':>20}  {'length':>8}")
    for k in range(6):
        err = float(jnp.max(jnp.abs(jnp.array(
            [float(r(jnp.array(xi))) for xi in xx]) - jnp.array(abs_x))))
        print(f"  {k:>4}  {err:>20.4e}  {len(r):>8}")

        ax = axes[k]
        r_vals = np.array([float(r(jnp.array(xi))) for xi in xx])
        ax.plot(xx, r_vals, 'b', lw=1.8)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.2, 1.2)
        ax.set_title(f'step {k}: err={err:.1e}, len={len(r)}', fontsize=9)

        r = (r**2 + x**2) / (2.0 * r)

    fig.suptitle("Newton iteration converging to |x|", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'AbsoluteValue.png'), dpi=150)
    plt.close(fig)

    # Error plot after 6 steps
    fig2, ax2 = plt.subplots()
    err_vals = np.array([abs(float(r(jnp.array(xi))) - abs(xi)) + 1e-18
                         for xi in xx])
    ax2.semilogy(xx, err_vals, 'b', lw=1.8)
    ax2.set_xlim(-1, 1)
    ax2.set_title('Error after 6 Newton steps', fontsize=11)
    fig2.tight_layout()
    fig2.savefig(os.path.join(_OUTDIR, 'AbsoluteValue_err.png'), dpi=150)
    plt.close(fig2)

    print("AbsoluteValue: done.")
    return True

if __name__ == '__main__':
    run()
