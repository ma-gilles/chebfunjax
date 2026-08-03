"""Writing messages in 3D.

Faithful replica of fun/Writing3D.m by Nick Trefethen
(November 2010): scribble text plotted flat, bent along a sine wave,
and wrapped around a cylinder in 3D.

Original: https://www.chebfun.org/examples/fun/Writing3D.html
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
from chebfunjax.utils.scribble import scribble

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'fun')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Writing3D_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pieces(cf, n=14):
    bps = [float(v) for v in cf.domain.breakpoints]
    for a, b in zip(bps[:-1], bps[1:]):
        t = np.linspace(a, b, n)
        yield np.asarray(cf(t))


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    s = scribble("There is no fun like chebfun.")
    fig, ax = plt.subplots(figsize=(9.8, 2.6))
    for z in _pieces(s):
        ax.plot(z.real, z.imag, 'r', lw=2)
    ax.set_xlim(-1.05, 1.05)
    ax.set_aspect("equal")
    _save(fig)

    fig = plt.figure(figsize=(9.8, 4.4))
    ax = fig.add_subplot(projection="3d")
    for z in _pieces(s, 30):
        ax.plot(z.real, np.sin(6 * z.real), z.imag, 'b', lw=1.6)
    ax.view_init(elev=6, azim=-1.5)
    ax.set_box_aspect((3, 1, 1))
    _save(fig)

    s2 = scribble("There is no fun like chebfun.  "
                  "Try it and you'll see.  "
                  "It does your calculation, "
                  "and makes a cup of tea!")
    fig = plt.figure(figsize=(8.6, 7.0))
    ax = fig.add_subplot(projection="3d")
    for z in _pieces(s2, 30):
        rs = 6 * z.real
        ax.plot(np.cos(rs), np.sin(rs),
                6 * z.imag + 0.05 * rs, lw=1.4)
    ax.set_box_aspect((1, 1, 1))
    _save(fig)


if __name__ == "__main__":
    run()
