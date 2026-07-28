"""Accuracy of Legendre coefficients via aliasing.

Faithful port of approx/AliasingCoefficientsLeg.m by Yuji Nakatsukasa (April
2016).  The Legendre analogue of AliasingCoefficients: the 2D Legendre
coefficients of ``sin(x+y)+cos(x-y)`` are compared with those of its
interpolant on a 6x6 Gauss-Legendre grid, and the aliasing-error matrix is
reported.

Original: https://www.chebfun.org/examples/approx/AliasingCoefficientsLeg.html
Copyright 2016 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): the full aliasing-error matrix reproduces the
published values (row 1: 1.2730e-12, 2.2176e-11, 6.2811e-10, 1.4621e-08,
2.9536e-07, 5.1823e-06) using chebfun2 ``chebcoeffs2``, ``cheb2leg``,
``legpts``, and ``legvals2legcoeffs``.
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
from chebfunjax.utils.quadrature import legpts
from chebfunjax.utils.transforms import cheb2leg, legvals2legcoeffs

chebfun_style()

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def _cols(fn, M):
    """Apply a 1D transform ``fn`` to every column of ``M``."""
    return np.stack([np.asarray(fn(jnp.asarray(M[:, j])))
                     for j in range(M.shape[1])], axis=1)


def _both(fn, M):
    """Apply ``fn`` along both dimensions (MATLAB ``fn(fn(M)')'``)."""
    return _cols(fn, _cols(fn, M).T).T


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    fori = lambda x, y: jnp.sin(x + y) + jnp.cos(x - y)

    # Full 2D Legendre coefficients of f.
    f = chebfun2(fori)
    fc = _both(cheb2leg, np.asarray(f.chebcoeffs2()))

    # Interpolant on a 6x6 Gauss-Legendre grid -> its Legendre coefficients.
    k = 6
    s = np.asarray(legpts(k))[0]
    xx = np.tile(s.reshape(1, -1), (k, 1))
    yy = np.tile(s.reshape(-1, 1), (1, k))
    V = np.asarray(fori(jnp.asarray(xx), jnp.asarray(yy)))
    ptc = _both(legvals2legcoeffs, V)

    alias = np.abs(fc[:k, :k] - ptc)
    print("ans =")
    for i in range(k):
        print("  " + "  ".join(f"{alias[i, j]:.4e}" for j in range(k)))

    # ------------------------------------------------------------------
    # Plot: the Legendre aliasing-error matrix magnitude.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(np.log10(alias + 1e-20), cmap="viridis")
    ax.set_title("log10 |Legendre aliasing error|", fontsize=10)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'AliasingCoefficientsLeg.png'), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return True


if __name__ == '__main__':
    run()
