"""Low-rank approximation and localized singularities.

Faithful replica of approx2/Localization.m (Trefethen, 2016):
low-rank compression is dramatic when a (near-)singularity is
localized — a sharp spike inside the domain, or a real singularity
just outside a corner — shown by rank vs. length at eps = 1e-10,
with the GE pivot-cross pictures.

Original: https://www.chebfun.org/examples/approx2/Localization.html
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

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2')

EP = 1e-10


def _report(F):
    m, n = F.length()
    print("r =")
    print(f"    {int(F.rank)}")
    print("m =")
    print(f"    {m}")
    print("n =")
    print(f"    {n}")


def _pivot_plot(F, k, xticks, yticks):
    piv = np.array(F.pivot_locations)
    n = piv.shape[0]
    print("n =")
    print(f"    {n}")
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for (px, py) in piv:
        ax.plot([-1, 1], [py, py], '-k', lw=0.8)
        ax.plot([px, px], [-1, 1], '-k', lw=0.8)
    ax.plot(piv[:, 0], piv[:, 1], 'or', markerfacecolor='none', ms=7)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Localization_repl_{k:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Broad spike inside the domain: modest compression.
    f = Chebfun2.from_function(
        lambda x, y: 1 / (1 + (x - .2)**2 + (y - .5)**2), tol=EP)
    _report(f)

    # Sharp spike: dramatic difference between rank and length.
    f = Chebfun2.from_function(
        lambda x, y: 1 / (0.001 + (x - .2)**2 + (y - .5)**2), tol=EP)
    _report(f)
    _pivot_plot(f, 1, [-1, 0.2, 1], [-1, 0.5, 1])

    # Real singularity outside a corner, not very close: little
    # compression.
    g = Chebfun2.from_function(
        lambda x, y: 1 / ((x + 1.2)**2 + (y + 1.2)**2), tol=EP)
    _report(g)

    # Singularity very close to the corner: striking compression.
    g = Chebfun2.from_function(
        lambda x, y: 1 / ((x + 1.02)**2 + (y + 1.02)**2), tol=EP)
    _report(g)
    _pivot_plot(g, 2, [-1, 0.2, 1], [-1, 0.5, 1])


if __name__ == "__main__":
    run()
