"""Random ponds in a 2D landscape.

Faithful replica of approx2/RandomPonds.m (Trefethen, 2017): a
random landscape f filled with water up to level h -- zebra plots of
f - h in blue/black at h = -1, -2, 0, 1, 2, and a varying water
level h(x, y) = x.

Random draws use a seeded numpy stream (MATLAB's rng state is not
reproducible outside MATLAB); pond morphology vs. h is the content.

Original: https://www.chebfun.org/examples/approx2/RandomPonds.html
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

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.random import randnfun2

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2')

BLUE = (.6, .6, 1)
BLACK = (0, 0, 0)
FIG = [0]


def _ponds(Z, X, Y, title, aspect):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(4.6 * aspect, 4.6))
    ax.contourf(X, Y, Z, [float(Z.min()) - 1, 0, float(Z.max()) + 1],
                colors=[BLUE, BLACK])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"RandomPonds_repl_{FIG[0]:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    dom = (-2, 2, -1, 1)
    f = randnfun2(0.3, dom, seed=0)
    gx = np.linspace(-2, 2, 800)
    gy = np.linspace(-1, 1, 400)
    X, Y = np.meshgrid(gx, gy)
    F = np.asarray(f(X, Y))

    _ponds(F - (-1), X, Y, "h = -1", 2)
    _ponds(F - (-2), X, Y, "h = -2", 2)
    for h in [0, 1, 2]:
        _ponds(F - h, X, Y, f"h = {h}", 2)

    # varying water level h(x,y) = x
    dom = (-3, 3, -1, 1)
    f = randnfun2(.1, dom, seed=1)
    gx = np.linspace(-3, 3, 1500)
    gy = np.linspace(-1, 1, 500)
    X, Y = np.meshgrid(gx, gy)
    F = np.asarray(f(X, Y))
    _ponds(F - X, X, Y, "varying h", 3)


if __name__ == "__main__":
    run()
