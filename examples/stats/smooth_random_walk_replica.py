"""Smooth random walk.

Faithful replica of stats/SmoothRandomWalk.m by Nick Trefethen
(May 2017): the indefinite integral of a complex smooth random
function ('big' normalization) approaches 2D Brownian motion as the
wavelength dx shrinks.

randn draws are not bit-reproducible vs MATLAB; each panel is our own
draw of the same random-function family.

Original: https://www.chebfun.org/examples/stats/SmoothRandomWalk.html
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
from chebfunjax.utils.randnfun import randnfun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')

FIG = [0]


def _walk_plot(dx, lw):
    FIG[0] += 1
    f = randnfun(dx, key=jax.random.PRNGKey(1), big=True, cmplx=True)
    g = f.cumsum()
    ts = np.linspace(-1, 1, 4000)
    z = np.asarray(g(ts))
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.plot(z.real, z.imag, 'k', lw=lw)
    ends = np.asarray(g(np.array([-1.0, 1.0])))
    ax.plot(ends.real, ends.imag, '.r', ms=10)
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_title(f"dx = {dx:g}", fontsize=12)
    ax.set_xticks(range(-2, 3))
    ax.set_yticks(range(-2, 3))
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"SmoothRandomWalk_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    dx = 0.1
    _walk_plot(dx, 1.0)
    for k in range(1, 4):
        dx = dx / 4
        _walk_plot(dx, 1 - 0.15 * k)


if __name__ == "__main__":
    run()
