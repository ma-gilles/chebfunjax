"""A random surface on a disk.

Faithful replica of stats/RandomSurf.m by Nick Trefethen and Grady
Wright (April 2017): a smooth random function on the unit disk added
to a paraboloid, shown with a zebra plot, contour plot, and surface
plot.

randn draws are not bit-reproducible vs MATLAB; the surface is our
own draw from the same family.

Original: https://www.chebfun.org/examples/stats/RandomSurf.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.random import randnfundisk

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"RandomSurf_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    F = np.asarray(randnfundisk(3, key=jax.random.PRNGKey(1),
                                lam=0.1))
    nr, nt = F.shape
    r = np.linspace(0, 1, nr)
    th = np.linspace(0, 2 * np.pi, nt)
    R, T = np.meshgrid(r, th, indexing="ij")
    X = R * np.cos(T)
    Y = R * np.sin(T)
    Z = F + (2 - 4 * R**2)

    # zebra plot: two-tone by sign
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    ax.pcolormesh(X, Y, np.where(Z >= 0, 1.0, 0.0), cmap="gray",
                  shading="auto")
    ax.set_aspect("equal")
    ax.set_axis_off()
    _save(fig)

    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    cs = ax.contourf(X, Y, Z, levels=20)
    fig.colorbar(cs, ax=ax)
    ax.set_aspect("equal")
    ax.set_axis_off()
    _save(fig)

    fig = plt.figure(figsize=(8.6, 6.6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", rstride=1, cstride=2)
    ax.set_zlim(-10, 10)
    ax.view_init(elev=60, azim=0)
    ax.set_axis_off()
    _save(fig)


if __name__ == "__main__":
    run()
