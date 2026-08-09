"""Poloidal-toroidal decomposition of a vector field.

Faithful replica of sphere/PTDecomposition.m: a divergence-free field
in the ball splits as w = curl(curl(P r)) + curl(T r); the ballfun
commands PT2ballfunv and PTdecomposition round-trip the scalars.

Original: https://www.chebfun.org/examples/sphere/PTDecomposition.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')


def _quiver(ax, v, title, nr=3, nl=14, nt=7):
    rs = np.linspace(0.35, 0.95, nr)
    lam = np.linspace(-np.pi, np.pi, nl, endpoint=False)
    th = np.linspace(0.35, np.pi - 0.35, nt)
    R, L, T = np.meshgrid(rs, lam, th, indexing="ij")
    sh = R.shape
    vx = np.asarray(v.components[0](
        R.reshape(1, -1), L.reshape(1, -1), T.reshape(1, -1))).reshape(sh)
    vy = np.asarray(v.components[1](
        R.reshape(1, -1), L.reshape(1, -1), T.reshape(1, -1))).reshape(sh)
    vz = np.asarray(v.components[2](
        R.reshape(1, -1), L.reshape(1, -1), T.reshape(1, -1))).reshape(sh)
    X = R * np.cos(L) * np.sin(T)
    Y = R * np.sin(L) * np.sin(T)
    Z = R * np.cos(T)
    ax.quiver(X, Y, Z, vx, vy, vz, length=0.18, lw=0.6,
              color="tab:blue")
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(title)


def _slice(ax, c, title, n=140):
    g = np.linspace(-1, 1, n)
    A, B = np.meshgrid(g, g)
    inside = A**2 + B**2 <= 1
    xi, zi = A[inside], B[inside]
    ri = np.sqrt(xi**2 + zi**2)
    lami = np.where(xi >= 0, 0.0, np.pi)
    thi = np.arccos(np.clip(np.where(ri > 0, zi / np.maximum(ri, 1e-300),
                                     1.0), -1, 1))
    V = np.full(A.shape, np.nan)
    V[inside] = np.asarray(c(ri.reshape(1, -1), lami.reshape(1, -1),
                             thi.reshape(1, -1))).ravel()
    ax.imshow(V, origin="lower", cmap="viridis", extent=(-1, 1, -1, 1))
    ax.set_axis_off()
    ax.set_title(title)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    Pw = Ballfun.from_function(lambda x, y, z: np.cos(x * y))
    Tw = Ballfun.from_function(lambda x, y, z: np.sin(y * z))
    w = Ballfunv.PT2ballfunv(Pw, Tw)

    fig = plt.figure(figsize=(6.4, 6.0))
    ax = fig.add_subplot(projection="3d")
    _quiver(ax, w, "")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "PTDecomposition_repl_01.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("ans =")
    print(f"     {float(w.div().norm()):.15e}")

    P2, T2 = w.PTdecomposition()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    _slice(axes[0], P2, "poloidal scalar")
    _slice(axes[1], T2, "toroidal scalar")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "PTDecomposition_repl_02.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    Pv, Tv = Ballfunv.PT2ballfunv(P2, T2, nargout=2)
    fig = plt.figure(figsize=(13.5, 4.6))
    for i, (vv, ttl) in enumerate([
            (w, "divergence-free field"), (Pv, "poloidal component"),
            (Tv, "toroidal component")]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        _quiver(ax, vv, ttl)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "PTDecomposition_repl_03.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    v = Ballfunv.PT2ballfunv(P2, T2)
    print("ans =")
    print(f"     {float((v - w).norm()):.15e}")


if __name__ == "__main__":
    run()
