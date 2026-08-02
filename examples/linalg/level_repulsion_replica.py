"""Eigenvalue level repulsion.

Faithful replica of linalg/LevelRepulsion.m by Nick Trefethen
(October 2010): the eigenvalues of the symmetric interpolation
(1-t)A + tB avoid crossing as t varies — neighboring eigenvalue
curves repel — illustrated by chebfuns of the sorted eigenvalues.

MATLAB seeds rng(1); randn draws are not bit-reproducible between
MATLAB and numpy, so the matrices (and hence the minimal gap) differ;
the level-repulsion phenomenon is what reproduces.

Original: https://www.chebfun.org/examples/linalg/LevelRepulsion.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')


def run():
    os.makedirs(_IMG, exist_ok=True)

    n = 10
    rs = np.random.RandomState(1)
    A = rs.randn(n, n)
    A = A + A.T
    B = rs.randn(n, n)
    B = B + B.T

    def eigk(t_arr, k):
        t_arr = np.atleast_1d(np.asarray(t_arr, dtype=float))
        out = np.empty_like(t_arr)
        for i, t in enumerate(t_arr.ravel()):
            w = np.linalg.eigvalsh((1 - t) * A + t * B)
            out.ravel()[i] = np.sort(w)[k]
        return out.reshape(np.shape(t_arr))

    t0 = time.time()
    E = [cj.chebfun(lambda x, _k=k: jnp.asarray(eigk(np.asarray(x),
                                                     _k)),
                    domain=(0.0, 1.0))
         for k in range(n)]
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")

    xs = np.linspace(0, 1, 600)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for e in E:
        ax.plot(xs, np.asarray(e(xs)), lw=1.2)
    ax.grid(True)
    ax.set_title("Eigenvalues of (1-t)A + tB", fontsize=12)
    ax.set_xlabel("t")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "LevelRepulsion_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # the closest pair of curves at an interior near-crossing
    gaps = [(k, (E[k + 1] - E[k]).min()) for k in range(n - 1)]
    interior = [g for g in gaps
                if 1e-3 < float(g[1][0]) < 1 - 1e-3]
    pool = interior if interior else gaps
    k_min, (minpos, minval) = min(pool, key=lambda g: float(g[1][1]))
    minpos, minval = float(minpos), float(minval)
    print("minval =")
    print(f"   {minval:.15f}")
    print("minpos =")
    print(f"   {minpos:.15f}")

    E5, E6 = E[k_min], E[k_min + 1]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    xs2 = np.linspace(max(0, minpos - 0.05), min(1, minpos + 0.05),
                      400)
    for e in E:
        ax.plot(xs2, np.asarray(e(xs2)), lw=1.2)
    v5 = float(E5(minpos))
    ax.axis([minpos - 0.05, minpos + 0.05, v5 - 0.4, v5 + 0.4])
    ax.grid(True)
    ax.set_title(f"Zooming in: the gap width is {minval:.5g}",
                 fontsize=12)
    ax.plot(minpos, v5, '.k', ms=10)
    ax.plot(minpos, float(E6(minpos)), '.k', ms=10)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "LevelRepulsion_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
