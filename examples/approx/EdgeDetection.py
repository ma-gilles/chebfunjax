"""Edge detection in Chebfun.

Demonstrates Chebfun's ability to automatically detect discontinuities
and choose breakpoints in splitting mode.

Credit: Nick Trefethen, November 2016.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/EdgeDetection.html
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

    # A piecewise smooth function with known breakpoints
    def f_pw(x):
        # sin(x) on [0,2], sin(2x) on [2,5], sin(3x) on [5,8]
        return jnp.where(x < 2.0,
               jnp.sin(x),
               jnp.where(x < 5.0, jnp.sin(2.0 * x), jnp.sin(3.0 * x)))

    # With explicit breakpoints (pass as list directly, not as Domain object)
    f_explicit = cj.chebfun(f_pw, domain=[0.0, 2.0, 5.0, 8.0])

    # Without breakpoints (smooth approximation)
    f_smooth = cj.chebfun(f_pw, domain=(0.0, 8.0))

    xx = np.linspace(0.0, 8.0, 800)
    f_e_vals = np.array([float(f_explicit(jnp.array(x))) for x in xx])
    f_s_vals = np.array([float(f_smooth(jnp.array(x))) for x in xx])
    f_true = np.array([float(f_pw(jnp.array(x))) for x in xx])

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.plot(xx, f_e_vals, 'b', lw=1.8, label='piecewise (breakpoints at 2,5)')
    ax.plot(xx, f_true, 'k--', lw=1.0, alpha=0.5, label='true function')
    ax.axvline(2.0, color='r', lw=1.0, ls='--')
    ax.axvline(5.0, color='r', lw=1.0, ls='--')
    ax.set_title('Piecewise chebfun with explicit breakpoints', fontsize=10)
    ax.legend(fontsize=8)
    ax2 = axes[1]
    err_vals = np.abs(f_s_vals - f_true)
    ax2.semilogy(xx, err_vals + 1e-18, 'b', lw=1.5)
    ax2.set_title('Error of smooth (no-breakpoint) approximation', fontsize=10)
    fig.suptitle('Edge detection: piecewise vs. smooth chebfun', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'EdgeDetection.png'), dpi=150)
    plt.close(fig)

    print(f"EdgeDetection: piecewise len={len(f_explicit)}, smooth len={len(f_smooth)}")
    return True

if __name__ == '__main__':
    run()
