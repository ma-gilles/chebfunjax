"""Random functions in 2D.

Faithful replica of approx2/Random2D.m (Trefethen, 2017): smooth
random functions from randnfun2 at space scales lambda = 0.2, 0.1,
0.05 on a 2x1 rectangle -- zebra plots, contours, high-accuracy zero
contours from roots, a 3D view, and the periodic variant.

Random draws use seeded numpy streams (MATLAB's rng(0) stream is not
reproducible outside MATLAB); the structure vs. lambda is the
content of the example.

Original: https://www.chebfun.org/examples/approx2/Random2D.html
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

from chebfunjax.utils.random import randnfun2
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2')

DOM = (0.0, 2.0, 0.0, 1.0)


def _grid(n=400):
    gx = np.linspace(0, 2, 2 * n)
    gy = np.linspace(0, 1, n)
    return np.meshgrid(gx, gy)


def _zebra(f, k):
    X, Y = _grid()
    Z = np.asarray(f(X, Y))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.contourf(X, Y, Z, [float(Z.min()) - 1, 0, float(Z.max()) + 1],
                colors=[(0, 0, 0), (1, 1, 1)])
    ax.set_aspect("equal")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 2.01, .5))
    ax.set_yticks(np.arange(0, 1.01, .5))
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Random2D_repl_{k:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    lam = 0.2
    f = randnfun2(lam, DOM, seed=0)
    _zebra(f, 1)

    # contour plot
    X, Y = _grid()
    Z = np.asarray(f(X, Y))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.contour(X, Y, Z, 10)
    ax.set_aspect("equal")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 2.01, .5))
    ax.set_yticks(np.arange(0, 1.01, .5))
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Random2D_repl_02.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    # zero contours via roots
    c = f.roots()
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for cc in c:
        t = np.linspace(float(cc.domain.a), float(cc.domain.b), 500)
        z = np.asarray(cc(t))
        ax.plot(np.real(z), np.imag(z), 'b', lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 2.01, .5))
    ax.set_yticks(np.arange(0, 1.01, .5))
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Random2D_repl_03.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"zero-contour components: {len(c)}")

    # 3D view
    fig = plt.figure(figsize=(9.0, 5.4))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", rstride=2, cstride=2,
                    linewidth=0)
    ax.view_init(50, -20 - 90)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Random2D_repl_04.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    # periodic random function
    ft = randnfun2(lam, DOM, seed=1, trig=True)
    _zebra(ft, 5)

    # lambda = 0.1 and 0.05
    _zebra(randnfun2(0.1, DOM, seed=2), 6)
    _zebra(randnfun2(0.05, DOM, seed=3), 7)


if __name__ == "__main__":
    run()
