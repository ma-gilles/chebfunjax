"""Accuracy of Chebyshev coefficients via aliasing.

Faithful port of approx/AliasingCoefficients.m by Yuji Nakatsukasa (April
2016).  When a bivariate function is interpolated on a coarse tensor
Chebyshev grid, its 2D Chebyshev coefficients are corrupted by aliasing: the
high-degree coefficients fold back onto the low-degree ones.  This computes
the aliasing error matrix ``ptc - pc`` for ``p = sin(x+y) + cos(x-y)`` between
the full chebfun2 coefficients and those of the degree-[5 5] interpolant on a
6x6 grid.

Original: https://www.chebfun.org/examples/approx/AliasingCoefficients.html
Copyright 2016 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): the full aliasing-error matrix reproduces the
published values (e.g. row 1: -8.0518e-10, 7.5521e-09, 1.4432e-07,
-2.2991e-06, -3.2044e-05) using chebfun2 ``chebcoeffs2`` and ``chebpts2``.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.quadrature import chebpts2

chebfun_style()

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    ff = lambda x, y: jnp.sin(x + y) + jnp.cos(x - y)
    p = chebfun2(ff)
    pc = np.asarray(p.chebcoeffs2())

    # Degree-[5 5] interpolant from a 6x6 tensor Chebyshev grid.
    X, Y = chebpts2(6)
    vals = np.asarray(ff(jnp.asarray(np.asarray(X)),
                         jnp.asarray(np.asarray(Y))))
    pt = chebfun2(vals)
    ptc = np.asarray(pt.chebcoeffs2())

    r, c = ptc.shape
    alias = ptc - pc[:r, :c]

    print("ans =")
    for i in range(r):
        print("  " + "  ".join(f"{alias[i, j]:.4e}" for j in range(c)))

    # ------------------------------------------------------------------
    # Plot: the aliasing-error matrix magnitude.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(np.log10(np.abs(alias) + 1e-20), cmap="viridis")
    ax.set_title("log10 |aliasing error| in T_i(x) T_j(y) coeffs", fontsize=10)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'AliasingCoefficients.png'), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return True


if __name__ == '__main__':
    run()
