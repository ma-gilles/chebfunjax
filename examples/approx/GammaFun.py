"""The gamma function and its poles.

Demonstrates Chebfun's capabilities for functions with poles by exploring
the gamma function Γ(x) on [-4, 4].

Credit: Nick Hale, December 2009 (revised June 2019 by Nick Trefethen).
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/GammaFun.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.special import gamma as scipy_gamma
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()


_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Build a piecewise chebfun for gamma(x) avoiding poles at 0, -1, -2, -3, -4
    breakpoints = (-4.0, -3.0, -2.0, -1.0, 0.0, 4.0)

    def gamma_safe(x):
        return jnp.array(float(scipy_gamma(float(x))))

    # Evaluate on each piece separately
    pieces_xx = []
    pieces_yy = []
    for a, b in zip(breakpoints[:-1], breakpoints[1:]):
        mid = 0.5 * (a + b)
        xx = np.linspace(a + 0.05, b - 0.05, 200)
        yy = np.array([scipy_gamma(x) for x in xx])
        pieces_xx.append(xx)
        pieces_yy.append(yy)

    # Also compute |gamma| and sqrt(|gamma|) for the smooth pieces
    xx_all = np.concatenate(pieces_xx)
    yy_all = np.concatenate(pieces_yy)
    abs_yy = np.abs(yy_all)
    sqrt_yy = np.sqrt(abs_yy)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for xx, yy in zip(pieces_xx, pieces_yy):
        ax.plot(xx, yy, 'b', lw=1.8)
    ax.set_ylim(-8, 8)
    ax.set_title('Gamma function Γ(x) on [−4, 4]', fontsize=11)
    ax.set_xlabel('x')
    ax.axhline(0, color='k', lw=0.5)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for xx, yy in zip(pieces_xx, pieces_yy):
        abs_y = np.abs(yy)
        ax2.plot(xx, abs_y, 'b', lw=1.5, label='|Γ(x)|' if xx is pieces_xx[0] else '')
        ax2.plot(xx, np.sqrt(abs_y), 'r', lw=1.5,
                 label='√|Γ(x)|' if xx is pieces_xx[0] else '')
        ax2.plot(xx, 1.0 / np.maximum(abs_y, 1e-10), 'g', lw=1.5,
                 label='1/Γ(x)' if xx is pieces_xx[0] else '')
    ax2.set_ylim(0, 8)
    ax2.set_title('Related functions', fontsize=11)
    ax2.set_xlabel('x')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'GammaFun.png'), dpi=150)
    plt.close(fig)

    print("GammaFun: done.")
    return True


if __name__ == '__main__':
    run()
