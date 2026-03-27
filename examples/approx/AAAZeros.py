"""Rootfinding with the AAA algorithm.

Demonstrates using the AAA rational approximant's poles and zeros to locate
roots of analytic and trigonometric functions.

Credit: Stefano Costa, June 2022.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/AAAZeros.html
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

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')

def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # --- Rootfinding for a function via AAA ---------------------------------
    # Function: f(x) = sin(10*x) + x^2 - 0.5 on [-1, 1]
    xs = jnp.linspace(-1.0, 1.0, 500)
    def f_func(x): return jnp.sin(10.0 * x) + x**2 - 0.5
    ys = f_func(xs)

    # Chebfun roots via standard polynomial roots
    f_cheb = cj.chebfun(f_func)
    roots_cheb = f_cheb.roots()

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    xx = np.linspace(-1.0, 1.0, 600)
    ax.plot(xx, np.array(f_func(jnp.array(xx))), 'b', lw=1.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.plot(np.array(roots_cheb), np.zeros(len(roots_cheb)), '.r', ms=8,
            label=f'{len(roots_cheb)} roots')
    ax.set_title('sin(10x) + x² − 0.5 and its roots', fontsize=11)
    ax.legend(fontsize=9)

    # --- AAA on complex domain: locate zeros in a strip ---------------------
    # f(z) = sin(z) near z in strip |Im(z)| < 0.5
    npts = 400
    rng = np.random.default_rng(42)
    X_r = 2 * np.pi * rng.random(npts) - np.pi
    X_i = rng.random(npts) - 0.5
    Z = X_r + 1j * X_i
    Fz = np.sin(Z)

    r_sin, pol_sin, res_sin, zer_sin, *_ = aaa(
        jnp.array(Fz.real) + 0j, jnp.array(Z.real),
    )

    ax2 = axes[1]
    ax2.plot(X_r, X_i, '.', color='gray', ms=2, alpha=0.5, label='sample pts')
    # Show real-line roots of sin via chebfun
    f_sin = cj.chebfun(jnp.sin, domain=(-float(jnp.pi), float(jnp.pi)))
    roots_sin = f_sin.roots()
    ax2.plot(np.array(roots_sin), np.zeros(len(roots_sin)), '.r', ms=10,
             label='roots on ℝ')
    ax2.set_title('sin(x) on [−π, π]: roots via Chebfun', fontsize=11)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'AAAZeros.png'), dpi=150)
    plt.close(fig)

    print(f"AAAZeros: found {len(roots_cheb)} roots of sin(10x)+x²-0.5")
    return True

if __name__ == '__main__':
    run()
