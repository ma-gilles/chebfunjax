"""Pretty functions approximated by Chebfun2.

Faithful replica of approx2/PrettyFunctions.m (Townsend, 2013):
contour plots with construction pivot locations for six functions,
surface plots for four more, and the waterfall plot of Franke's
function showing the lines on which it was sampled.

Original: https://www.chebfun.org/examples/approx2/PrettyFunctions.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import airy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2')


def _save(fig, k):
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"PrettyFunctions_repl_{k:02d}.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    funcs = [
        lambda x, y: 1 / (1 + 100 * (x**2 - y**2)**2),
        lambda x, y: 1 / (1 + 100 * (.5 - x**2 - y**2)**2),
        lambda x, y: 1 / (1 + 1e3 * ((x**2 - .25)**2 * (y**2 - .25)**2)),
        lambda x, y: np.cos(10 * (x**2 + y)) * np.sin(10 * (x + y**2)),
        lambda x, y: np.real(airy(5 * (x + y**2))[0]
                             * airy(-5 * (x**2 + y**2))[0]),
        lambda x, y: (np.tanh(10 * x) * np.tanh(10 * y)
                      / np.tanh(10)**2 + np.cos(5 * x)),
    ]

    # Six contour plots with pivot locations.
    n = 500
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.0))
    for ax, fj in zip(axes.ravel(), funcs):
        F = Chebfun2.from_function(fj)
        ax.contour(X, Y, fj(X, Y), 10)
        piv = np.array(F.pivot_locations)
        if piv.size:
            ax.plot(piv[:, 0], piv[:, 1], '.k', ms=5)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
    _save(fig, 1)

    # Four surface plots.
    C = 1000
    gfuncs = [
        lambda x, y: (np.tanh(C * x) / np.tanh(C)
                      * np.tanh(C * y) / np.tanh(C)),
        lambda x, y: 1 / (1 + 200 * ((x - .3)**2 * (y + .5)**2)),
        lambda x, y: 1 / (1 + 100 * (x - y)**2),
        lambda x, y: 1 / (1 + 100 * (.5 - x**2 - y**2)**2),
    ]
    gs = np.linspace(-1, 1, 300)
    XS, YS = np.meshgrid(gs, gs)
    fig = plt.figure(figsize=(11.0, 9.0))
    for j, gj in enumerate(gfuncs):
        G = Chebfun2.from_function(gj)
        ax = fig.add_subplot(2, 2, j + 1, projection="3d")
        Z = np.asarray(G(XS, YS))
        ax.plot_surface(XS, YS, Z, cmap="viridis",
                        rstride=2, cstride=2, linewidth=0)
        ax.set_zlim(-1, 3)
    _save(fig, 2)

    # Waterfall: Franke's function, sampled lines from the pivots.
    def h(x, y):
        return (.75 * np.exp(-((9 * x - 2)**2 + (9 * y - 2)**2) / 4)
                + .75 * np.exp(-((9 * x + 1)**2) / 49 - (9 * y + 1) / 10)
                + .5 * np.exp(-((9 * x - 7)**2 + (9 * y - 3)**2) / 4)
                - .2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2))

    H = Chebfun2.from_function(h)
    piv = np.array(H.pivot_locations)
    fig = plt.figure(figsize=(8.6, 6.6))
    ax = fig.add_subplot(projection="3d")
    t = np.linspace(-1, 1, 400)
    for (px, py) in piv:
        ax.plot(np.full_like(t, px), t, h(np.full_like(t, px), t),
                'b', lw=1.0)
        ax.plot(t, np.full_like(t, py), h(t, np.full_like(t, py)),
                'b', lw=1.0)
    ax.view_init(30, -37.5)
    _save(fig, 3)
    print(f"ranks: F1..F6 done; Franke rank {int(H.rank)} "
          f"({len(piv)} pivot lines)")


if __name__ == "__main__":
    run()
