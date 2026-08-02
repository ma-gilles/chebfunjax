"""Chebyshev polynomials as plotted by Fornberg and Higham.

Faithful replica of cheb/ChebPolysHigham.m by Nick Trefethen
(December 2011): the 3D waterfall plots of Chebyshev and Legendre
polynomials popularized by Fornberg's and Higham & Higham's books.

Original: https://www.chebfun.org/examples/cheb/ChebPolysHigham.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.polynomials import chebpoly, legpoly

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'cheb')

KS = [0, 2, 4, 10, 20, 40, 60]
XS = np.linspace(-1, 1, 1200)


def _waterfall(coeff_fn, fname):
    fig = plt.figure(figsize=(8.8, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    for j, k in enumerate(KS, start=1):
        c = coeff_fn(k)
        p = cj.chebfun(jnp.asarray(c), coeffs=True)
        ax.plot(np.full_like(XS, float(j)), XS,
                np.asarray(p(jnp.asarray(XS))), lw=1.6)
    ax.set_xlim(1, len(KS))
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1.0, 0.75, 4.0 / 6))
    ax.view_init(elev=28, azim=-72 - 90)
    ax.set_xticks(range(1, len(KS) + 1))
    ax.set_xticklabels(KS)
    ax.set_xlabel("k", fontsize=14)
    ax.set_ylabel("x", fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    _waterfall(chebpoly, "ChebPolysHigham_repl_01.png")
    _waterfall(legpoly, "ChebPolysHigham_repl_02.png")
    print("waterfalls plotted for k =", KS)


if __name__ == "__main__":
    run()
