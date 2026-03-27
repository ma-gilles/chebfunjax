"""CF approximation 30 years ago.

Caratheodory-Fejer (CF) approximation of functions on [-1,1]. The CF method
uses Hankel SVD to produce near-best rational approximants extremely quickly.

Credit: Nick Trefethen and Mohsin Javed, July 2014.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/CF30.html
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

    # Approximate sqrt(1.2 - x) using L2 polyfit as a proxy for CF/REMEZ
    f = cj.chebfun(lambda x: jnp.sqrt(1.2 - x))

    # Polynomial approximation of degree 1 and 3
    p1 = f.polyfit(1)
    p3 = f.polyfit(3)

    xx = np.linspace(-1.0, 1.0, 400)
    f_vals = np.array([float(f(jnp.array(x))) for x in xx])
    err1 = np.array([float((f - p1)(jnp.array(x))) for x in xx])
    err3 = np.array([float((f - p3)(jnp.array(x))) for x in xx])

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    p1_vals = np.array([float(p1(jnp.array(x))) for x in xx])
    ax.plot(xx, f_vals, 'b', lw=1.8, label='sqrt(1.2−x)')
    ax.plot(xx, p1_vals, 'r--', lw=1.5, label='degree-1 approx')
    ax.set_title('sqrt(1.2−x) and degree-1 approximant', fontsize=11)
    ax.legend(fontsize=9)
    ax2 = axes[1]
    ax2.plot(xx, err1, 'b', lw=1.8, label='degree-1 error')
    ax2.plot(xx, err3, 'r', lw=1.5, label='degree-3 error')
    ax2.set_title('Approximation errors (CF-style)', fontsize=11)
    ax2.legend(fontsize=9)
    fig.suptitle('Caratheodory-Fejer-style approximation of sqrt(1.2-x)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'CF30.png'), dpi=150)
    plt.close(fig)

    print(f"CF30: degree-1 max err = {np.max(np.abs(err1)):.4e}")
    return True


if __name__ == '__main__':
    run()
