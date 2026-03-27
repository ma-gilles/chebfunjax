"""AAA rational approximation.

Demonstrates the AAA (Adaptive Antoulas-Anderson) algorithm for rational
approximation of functions on intervals and in the complex plane.

Credit: Nick Trefethen, December 2016.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/AAAApprox.html
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

from chebfunjax.utils.aaa import aaa
from scipy.special import gamma as scipy_gamma

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')

def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # --- 1. AAA approximation of gamma on [-1,1] ---------------------------------
    xs = jnp.linspace(-1.0, 1.0, 500)
    ys = jnp.array([float(scipy_gamma(float(x))) for x in xs])
    r, pol, res, zer, zj, fj, w = aaa(ys, xs)

    # Plot rational approximant vs exact
    fig, ax = plt.subplots()
    xx = np.linspace(-1.0, 1.0, 600)
    ax.plot(xx, [float(scipy_gamma(x)) for x in xx], 'b', lw=1.8, label='Γ(x)')
    ax.plot(xx, [float(r(jnp.array(x)).real) for x in xx], 'r--', lw=1.5,
            label='AAA approx')
    ax.set_ylim(-8, 8)
    ax.set_xlim(-1, 1)
    ax.legend(fontsize=9)
    ax.set_title('AAA approximation of Γ(x) on [−1, 1]', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'AAAApprox.png'), dpi=150)
    plt.close(fig)

    # --- 2. AAA approximation of exp(x) on [-1,1] error curve -------------------
    xs2 = jnp.linspace(-1.0, 1.0, 500)
    ys2 = jnp.exp(xs2)
    r2, pol2, *_ = aaa(ys2, xs2)

    fig2, ax2 = plt.subplots()
    xx2 = np.linspace(-1.0, 1.0, 600)
    err2 = np.array([float((jnp.exp(jnp.array(x)) - r2(jnp.array(x))).real) for x in xx2])
    ax2.semilogy(xx2, np.abs(err2) + 1e-18, 'b', lw=1.8)
    ax2.set_title('AAA approximation error for exp(x)', fontsize=11)
    fig2.tight_layout()
    fig2.savefig(os.path.join(_OUTDIR, 'AAAApprox_err.png'), dpi=150)
    plt.close(fig2)

    print("AAAApprox: done. Poles found:", len(pol))
    return True

if __name__ == '__main__':
    run()
