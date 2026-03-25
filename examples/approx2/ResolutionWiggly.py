"""Resolution of wiggly functions.

The function sin(x)^2 + sin(x^2) on [0,14] is highly oscillatory and
tests the resolution of approximation methods.

Credit: Nick Hale and Nick Trefethen, October 2013.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/ResolutionWiggly.html
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

    dom = (0.0, 14.0)
    def wiggly(x): return jnp.sin(x)**2 + jnp.sin(x**2)
    f = cj.chebfun(wiggly, domain=dom)

    xx = np.linspace(0.0, 14.0, 1200)
    f_vals = np.array([float(f(jnp.array(x))) for x in xx])
    true_vals = np.sin(xx)**2 + np.sin(xx**2)

    # Polynomial approximations of increasing degree
    degrees = [50, 100, 200]
    fig, axes = plt.subplots(len(degrees) + 1, 1, figsize=(11, 12))

    ax = axes[0]
    ax.plot(xx, f_vals, 'b', lw=1.5)
    ax.set_title(f'Adaptive chebfun: f(x) = sin²(x) + sin(x²), len={len(f)}',
                 fontsize=10)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.3)

    for i, n in enumerate(degrees):
        ax_i = axes[i + 1]
        pn = f.polyfit(n)
        pn_vals = np.array([float(pn(jnp.array(x))) for x in xx])
        err = np.max(np.abs(pn_vals - true_vals))
        ax_i.plot(xx, pn_vals - true_vals, 'r', lw=1.0)
        ax_i.set_title(f'L2 deg {n} error (max={err:.3f})', fontsize=10)
        ax_i.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel('x')

    fig.suptitle('Resolution of wiggly function sin²(x)+sin(x²)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'ResolutionWiggly.png'), dpi=150)
    plt.close(fig)

    print(f"ResolutionWiggly: len(f)={len(f)}")
    return True


if __name__ == '__main__':
    run()
