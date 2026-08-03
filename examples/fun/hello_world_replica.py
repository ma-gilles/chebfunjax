"""Hello World, in low rank.

Faithful replica of fun/HelloWorld.m by Alex Townsend (March 2013):
the words HELLO WORLD as a rank-10 matrix, represented and
progressively approximated by a chebfun2.

Original: https://www.chebfun.org/examples/fun/HelloWorld.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'fun')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"HelloWorld_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _matrix():
    A = np.zeros((15, 40))
    A[1:9, 1:3] = 1
    A[4:6, 3:5] = 1
    A[1:9, 5:7] = 1
    A[2:10, 9:11] = 1
    A[2:4, 9:15] = 1
    A[5:7, 9:15] = 1
    A[8:10, 9:15] = 1
    A[3:11, 17:19] = 1
    A[9:11, 17:24] = 1
    A[4:12, 25:27] = 1
    A[10:12, 25:31] = 1
    A[5:13, 33:35] = 1
    A[5:13, 37:39] = 1
    A[5:7, 35:37] = 1
    A[11:13, 35:37] = 1
    return A


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    A = _matrix()
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ii, jj = np.nonzero(A)
    ax.plot(jj, ii, 'bs', ms=5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    _save(fig)
    print("ans =")
    print(f"    {np.linalg.matrix_rank(A)}")

    f = cj.chebfun2(jnp.asarray(A))
    # evaluate on the same chebyshev grid and compare
    m, n = A.shape
    ty = np.cos(np.pi * np.arange(m - 1, -1, -1) / (m - 1))
    tx = np.cos(np.pi * np.arange(n - 1, -1, -1) / (n - 1))
    TX, TY = np.meshgrid(tx, ty)
    X = np.asarray(f(jnp.asarray(TX), jnp.asarray(TY)))
    print("ans =")
    print(f"     {np.linalg.norm(A - X, 2):.15e}")

    B = np.flipud(A)
    x = np.linspace(-1, 1, 200)
    XX, YY = np.meshgrid(x, x)
    U, S, Vt = np.linalg.svd(B)
    for k in [1, 3, 5, 7, 10]:
        Bk = (U[:, :k] * S[:k]) @ Vt[:k]
        fk = cj.chebfun2(jnp.asarray(Bk))
        Z = np.asarray(fk(jnp.asarray(XX), jnp.asarray(YY)))
        fig, ax = plt.subplots(figsize=(9.2, 4.0))
        ax.contour(XX, YY, Z, levels=np.arange(0.1, 1.0, 0.1))
        ax.set_axis_off()
        ax.set_title(f"Rank {k}", fontsize=15)
        _save(fig)


if __name__ == "__main__":
    run()
