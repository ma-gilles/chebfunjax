"""AAA approximation of a spline.

Demonstrates AAA rational approximation of a piecewise polynomial (spline),
showing how poles cluster near the spline knots.

Credit: Nick Trefethen, April 2021.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/AAASpline.html
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

from chebfunjax.utils.aaa import aaa

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Build a spline-like function: sin(x + x^2/4) sampled at integers 0..10
    nodes = np.arange(0, 11)
    data = np.sin(nodes + nodes**2 / 4.0)

    # Build the spline using chebfunjax
    s = cj.chebfun(lambda x: x * 0.0, domain=(0.0, 10.0))  # placeholder
    # Use scipy for the spline interpolation
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(nodes, data)

    # Sample spline on a fine grid (use 200 pts for speed; mmax=30 to avoid OOM)
    X = np.linspace(0, 10, 200)
    Y = cs(X)

    # AAA approximation (reduced mmax for CPU performance)
    r, poles, *_ = aaa(jnp.array(Y), jnp.array(X), mmax=30, tol=1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: the spline function
    ax = axes[0]
    ax.plot(X, Y, 'b', lw=1.8, label='spline s(x)')
    ax.plot(nodes, data, '.k', ms=10, label='nodes')
    ax.set_title('Spline being approximated', fontsize=11)
    ax.set_xlabel('x')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Right: poles in complex plane
    ax2 = axes[1]
    poles_np = np.array([complex(p) for p in poles])
    ax2.plot(poles_np.real, poles_np.imag, '.r', ms=8)
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_xlim(1.5, 8.5)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'Poles of AAA approximant ({len(poles)} total)', fontsize=11)
    ax2.set_xlabel('Re(z)')
    ax2.set_ylabel('Im(z)')

    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'AAASpline.png'), dpi=150)
    plt.close(fig)

    err = np.max(np.abs(Y - np.array([float(r(jnp.array(float(x))).real) for x in X])))
    print(f"AAASpline: max error = {err:.2e}, {len(poles)} poles found")
    return True


if __name__ == '__main__':
    run()
