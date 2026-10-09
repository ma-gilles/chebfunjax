"""Poloidal-toroidal decomposition of a vector field.

Translation of sphere/PTDecomposition.m: a divergence-free field
in the ball splits as w = curl(curl(P r)) + curl(T r); the ballfun
commands PT2ballfunv and PTdecomposition round-trip the scalars.

Original: https://www.chebfun.org/examples/sphere/PTDecomposition.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')


def run():
    os.makedirs(_IMG, exist_ok=True)

    Pw = Ballfun.from_function(lambda x, y, z: np.cos(x * y))
    Tw = Ballfun.from_function(lambda x, y, z: np.sin(y * z))
    w = Ballfunv.PT2ballfunv(Pw, Tw)

    fig = plt.figure(figsize=(6.0, 2.53))
    ax = fig.add_subplot(projection="3d")
    w.quiver(ax=ax)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "PTDecomposition_01.png"), size=(600, 253))
    plt.close(fig)

    print("ans =")
    print(f"     {float(w.div().norm()):.15e}")

    P2, T2 = w.PTdecomposition()
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.53),
                             subplot_kw={"projection": "3d"})
    P2.plot(ax=axes[0], title="poloidal scalar")
    T2.plot(ax=axes[1], title="toroidal scalar")
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "PTDecomposition_02.png"), size=(600, 253))
    plt.close(fig)

    Pv, Tv = Ballfunv.PT2ballfunv(P2, T2, nargout=2)
    fig = plt.figure(figsize=(6.0, 2.53))
    for i, (vv, ttl) in enumerate([
            (w, "divergence-free field"), (Pv, "poloidal component"),
            (Tv, "toroidal component")]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        vv.quiver(ax=ax, title=ttl)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "PTDecomposition_03.png"), size=(600, 253))
    plt.close(fig)

    v = Ballfunv.PT2ballfunv(P2, T2)
    print("ans =")
    print(f"     {float((v - w).norm()):.15e}")


if __name__ == "__main__":
    run()
