"""Rose curves.

Translation of geom/RoseCurves.m by Grady Wright (June 2015):
rhodonea curves cos(m t / n) e^{it} as periodic chebfuns, the
pi/2 length advantage of trig representations, and a 6x6 garden.

Original: https://www.chebfun.org/examples/geom/RoseCurves.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings
from math import lcm

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'geom')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"RoseCurves_{FIG[0]:02d}.png"), size=(680, 680))
    plt.close(fig)


def rose(m, n, trig=True):
    L = 2 * np.pi * lcm(m, n)
    return cj.chebfun(
        lambda t: jnp.cos(m / n * t) * jnp.cos(t)
        + 1j * jnp.cos(m / n * t) * jnp.sin(t),
        domain=(0.0, L), trig=trig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    m, n = 50, 51
    f = rose(m, n, trig=True)
    g = rose(m, n, trig=False)
    print("ans =")
    print(f"   {len(g)/len(f):.15f}")
    print("ans =")
    print(f"   {np.pi/2:.15f}")

    # figure('position', [0 0 680 680])
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    _roses(ax, 6, 1.0)
    _save(fig)

    t0 = time.time()
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    _roses(ax, 12, 0.8)
    _save(fig)
    time_ = time.time() - t0
    print("time =")
    print(f"   {time_:.15f}")


def _roses(ax, N, lw):
    for m in range(1, N + 1):
        for n in range(1, N + 1):
            f = rose(m, n)
            offset = 2.5 * m - 2.5j * n
            (f + offset).plot(ax=ax, color='k', linewidth=lw * 0.75,
                              n_pts=max(600, 200 * lcm(m, n)))
    ax.set_aspect("equal")
    ax.autoscale(tight=True)
    ax.set_axis_off()


if __name__ == "__main__":
    run()
