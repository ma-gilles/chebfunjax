"""Lissajous curves.

Faithful replica of geom/Lissajous.m by Nick Trefethen
(October 2014): Lissajous figures sin(mt) + i cos(nt + pi d), and a
colorful array of them.

Original: https://www.chebfun.org/examples/geom/Lissajous.html
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

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'geom')

FIG = [0]
TS = np.linspace(0, 2 * np.pi, 3000)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Lissajous_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    m, n = 5, 6
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0))
    axes[0].plot(np.sin(m * TS), np.sin(n * TS), lw=1.6)
    axes[0].set_title(f"m={m}  n={n}  d=0.0", fontsize=12)
    axes[1].plot(np.sin(m * TS), np.sin(n * TS + np.pi / 2), lw=1.6)
    axes[1].set_title(f"m={m}  n={n}  d=0.5", fontsize=12)
    for ax in axes:
        ax.axis([-1, 1, -1, 1])
        ax.set_aspect("equal")
        ax.set_axis_off()
    _save(fig)

    colors = [(1, 0, 0), (0, 0.8, 0), (1, 0.75, 0),
              (0, 1, 1), (1, 0, 1), (0, 0, 0.75)]
    rs = np.random.RandomState(2)
    fig, axes = plt.subplots(6, 6, figsize=(10.5, 10.5))
    for i in range(6):
        for j in range(6):
            ax = axes[i, j]
            mm, nn = i + 1, j + 1
            d = rs.rand()
            z = np.sin(mm * TS) + 1j * np.cos(nn * TS + np.pi * d)
            ax.plot(z.real, z.imag,
                    color=colors[rs.randint(0, 6)], lw=0.9)
            ax.set_aspect("equal")
            ax.set_axis_off()
    _save(fig)


if __name__ == "__main__":
    run()
