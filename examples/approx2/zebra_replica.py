"""Zebra plots.

Faithful replica of approx2/Zebra.m (Trefethen, 2017): plus/minus
"zebra" plots -- on the disk (recolored to bumblebee yellow/black),
on the sphere (sphharm(15,5)), and on a rectangle for a random
function, plus the contourf equivalent and the higher-accuracy
zero curves from roots.

The random function uses a seeded numpy stream (MATLAB's rng state
is not reproducible outside MATLAB); the zebra rendering is the
content of the example.

Original: https://www.chebfun.org/examples/approx2/Zebra.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.utils.random import randnfun2
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2')


def _save(fig, k):
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Zebra_repl_{k:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Bumblebee plot on the disk: sign of sin(20*(x+y)*(1+y)).
    th = np.linspace(-np.pi, np.pi, 720)
    r = np.linspace(0, 1, 360)
    TH, R = np.meshgrid(th, r)
    X, Y = R * np.cos(TH), R * np.sin(TH)
    F = np.sin(20 * (X + Y) * (1 + Y))
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.contourf(X, Y, F, [-np.inf, 0, np.inf],
                colors=[(1, 1, 0), (0, 0, 0)])
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, 1)

    # Zebra plot on the sphere: Y_15^5.
    f = Spherefun.sphharm(15, 5)
    lam = np.linspace(-np.pi, np.pi, 400)
    thp = np.linspace(0, np.pi, 200)
    LAM, THP = np.meshgrid(lam, thp)
    V = np.asarray(f(LAM, THP))
    XS = np.sin(THP) * np.cos(LAM)
    YS = np.sin(THP) * np.sin(LAM)
    ZS = np.cos(THP)
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(projection="3d")
    cols = np.where(V > 0, 1.0, 0.0)
    ax.plot_surface(XS, YS, ZS, facecolors=plt.cm.gray(cols),
                    rstride=1, cstride=1, linewidth=0, antialiased=False,
                    shade=False)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")
    _save(fig, 2)

    # Zebra plot on a rectangle: a random function.
    fr = randnfun2(.2, (-2, 2, -1, 1), seed=1)
    gx = np.linspace(-2, 2, 800)
    gy = np.linspace(-1, 1, 400)
    GX, GY = np.meshgrid(gx, gy)
    FR = np.asarray(fr(GX, GY))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.contourf(GX, GY, FR, [-np.inf, 0, np.inf],
                colors=[(0, 0, 0), (1, 1, 1)])
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, 3)

    # "Giraffe plot": contourf with a brownish-orange colormap.
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    cs = ax.contourf(GX, GY, FR, [float(FR.min()), 0, float(FR.max())],
                     cmap=ListedColormap([(.8, .4, .2), (1, 1, 1)]))
    fig.colorbar(cs, ax=ax)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, 4)

    # Higher-accuracy boundaries from roots.
    c = fr.roots()
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for cc in c:
        t = np.linspace(float(cc.domain.a), float(cc.domain.b), 600)
        z = np.asarray(cc(t))
        ax.plot(np.real(z), np.imag(z), 'b', lw=1.0)
    ax.set_aspect("equal")
    _save(fig, 5)
    print(f"components from roots: {len(c)}")


if __name__ == "__main__":
    run()
