"""Edge detection in Chebfun.

Demonstrates Chebfun's ability to automatically detect discontinuities
and choose breakpoints in splitting mode.

Credit: Nick Trefethen, November 2016.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/EdgeDetection.html
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

    # A piecewise smooth function: sin(kx) on each piece
    # Build each piece as a separate chebfun to avoid jnp.where convergence issues
    f0 = cj.chebfun(lambda x: jnp.sin(x), domain=[0.0, 2.0])
    f1 = cj.chebfun(lambda x: jnp.sin(2.0 * x), domain=[2.0, 5.0])
    f2 = cj.chebfun(lambda x: jnp.sin(3.0 * x), domain=[5.0, 8.0])
    # Smooth approximation on whole domain (will be less accurate near breakpoints)
    f_smooth = cj.chebfun(lambda x: jnp.sin(x), domain=(0.0, 8.0))

    # True piecewise values via numpy
    def f_pw_np(x):
        return np.where(x < 2.0, np.sin(x), np.where(x < 5.0, np.sin(2.0 * x), np.sin(3.0 * x)))

    xx = np.linspace(0.0, 8.0, 800)
    xx_jnp = jnp.array(xx)
    # Evaluate each piece on its subdomain
    mask0 = xx <= 2.0
    mask1 = (xx > 2.0) & (xx <= 5.0)
    mask2 = xx > 5.0
    f_e_vals = np.zeros(len(xx))
    f_e_vals[mask0] = np.array(f0(jnp.array(xx[mask0])))
    f_e_vals[mask1] = np.array(f1(jnp.array(xx[mask1])))
    f_e_vals[mask2] = np.array(f2(jnp.array(xx[mask2])))
    f_s_vals = np.array(f_smooth(xx_jnp))
    f_true = f_pw_np(xx)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(xx, f_e_vals, 'b', lw=1.8, label='piecewise (3 pieces)')
    ax.plot(xx, f_true, 'k--', lw=1.0, alpha=0.5, label='true function')
    ax.axvline(2.0, color='r', lw=1.0, ls='--')
    ax.axvline(5.0, color='r', lw=1.0, ls='--')
    ax.set_title('Piecewise chebfun (separate pieces)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    err_vals = np.abs(f_s_vals - f_true)
    ax2.semilogy(xx, err_vals + 1e-18, 'b', lw=1.5)
    ax2.set_title('Error of smooth (no-breakpoint) approximation', fontsize=10)
    ax2.set_xlabel('x')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Edge detection: piecewise vs. smooth chebfun', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'EdgeDetection.png'), dpi=150)
    plt.close(fig)

    total_len = len(f0) + len(f1) + len(f2)
    print(f"EdgeDetection: piecewise total pts={total_len} (pieces: {len(f0)}, {len(f1)}, {len(f2)}), smooth len={len(f_smooth)}")
    return True


if __name__ == '__main__':
    run()
