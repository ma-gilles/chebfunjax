"""Eigenvalue landscapes.

Faithful replica of linalg/EigLandscapes.m by Nick Trefethen
(June 2016): the smallest eigenvalues of a two-parameter Hermitian
family B + xC + yD as chebfun2 surfaces — analytic (generic complex
Hermitian) vs kinked where eigenvalues cross (real symmetric,
requiring fixed-grid construction).

randn draws are not bit-reproducible between MATLAB and numpy; the
landscapes are our own draws of the same families.

Original: https://www.chebfun.org/examples/linalg/EigLandscapes.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

N = 8
FIG = [0]


def _eig_op(B, C, D, k):
    def op(X, Y):
        X = np.atleast_1d(np.asarray(X, dtype=float))
        Y = np.atleast_1d(np.asarray(Y, dtype=float))
        out = np.empty(X.shape)
        for i in np.ndindex(X.shape):
            w = np.linalg.eigvalsh(B + X[i] * C + Y[i] * D)
            out[i] = w[k]
        return jnp.asarray(out)
    return op


def _surf(fs, fname, view=None, axis3d=None):
    FIG[0] += 1
    fig = plt.figure(figsize=(8.6, 6.4))
    ax = fig.add_subplot(projection="3d")
    xs = np.linspace(-1, 1, 80)
    X, Y = np.meshgrid(xs, xs)
    for f in fs:
        Z = np.asarray(f(jnp.asarray(X), jnp.asarray(Y)))
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if view:
        ax.view_init(elev=view[1], azim=view[0])
    if axis3d:
        ax.set_xlim(axis3d[0], axis3d[1])
        ax.set_ylim(axis3d[2], axis3d[3])
        ax.set_zlim(axis3d[4], axis3d[5])
    ax.set_title(fname, fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"EigLandscapes_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    rs = np.random.RandomState(1)

    def herm():
        M = rs.randn(N, N) + 1j * rs.randn(N, N)
        return M + M.conj().T

    B, C, D = herm(), herm(), herm()
    f1 = cj.chebfun2(_eig_op(B, C, D, 0))
    f2 = cj.chebfun2(_eig_op(B, C, D, 1))
    _surf([f1, f2], "first two eigenvalues")

    gap = f2 - f1
    mingap, _loc = gap.min2()
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    xs = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(xs, xs)
    Z = np.asarray(gap(jnp.asarray(X), jnp.asarray(Y)))
    cs = ax.contour(X, Y, Z, levels=np.arange(0, 3.01, 0.25))
    fig.colorbar(cs, ax=ax)
    cs.set_clim(0, 3)
    ax.set_aspect("equal")
    ax.set_title(f"min gap = {float(mingap):.5g}", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"EigLandscapes_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"min gap (Hermitian family) = {float(mingap):.6f}")

    # real symmetric: eigenvalues can cross, surfaces have kinks;
    # fixed 512-point grids as in the MATLAB example
    rs2 = np.random.RandomState(8)

    def sym():
        M = rs2.randn(N, N)
        return M + M.T

    B, C, D = sym(), sym(), sym()
    npts = 512
    f1 = cj.chebfun2(_eig_op(B, C, D, 0), n=npts)
    f2 = cj.chebfun2(_eig_op(B, C, D, 1), n=npts)
    _surf([f1, f2], "first two eigenvalues (real symmetric)")
    _surf([f1, f2], "zoom toward a conical intersection",
          view=(-27, 18))


if __name__ == "__main__":
    run()
