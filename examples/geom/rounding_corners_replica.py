"""Rounding corners by convolution.

Faithful replica of geom/RoundingCorners.m by Nick Trefethen
(October 2011): mollifying the corners of a piecewise-linear
function and of a planar curve by convolution with a narrow hat
kernel.

Original: https://www.chebfun.org/examples/geom/RoundingCorners.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'geom')

FIG = [0]
AX = (-1.2, 1.2, 0, 2.4)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"RoundingCorners_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pw(ax, f, color='b'):
    bps = [float(v) for v in f.domain.breakpoints]
    for a, b in zip(bps[:-1], bps[1:]):
        t = np.linspace(a, b, 200)
        ax.plot(t, np.asarray(f(t)), color, lw=1.4)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    t = cj.chebfun(lambda s: s)
    f = (t + 0.4).abs().minimum((t - 0.3).abs()) * 3.0
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    _plot_pw(ax, f)
    ax.axis(AX)
    ax.grid(True)
    _save(fig)

    h = 0.1
    g = cj.chebfun(lambda s: (h - jnp.abs(s)) / h**2,
                   domain=(-h, 0.0, h))
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    _plot_pw(ax, g, 'k')
    ax.axis([-1, 1, 0, 12])
    ax.grid(True)
    _save(fig)

    f2 = f.conv(g)
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    _plot_pw(ax, f2)
    ax.axis(AX)
    ax.grid(True)
    _save(fig)

    # the corner-rounded planar curve: convolve x(t) and y(t)
    xr = t.conv(g)
    yr = f.conv(g)
    ts = np.linspace(-1 + h, 1 - h, 900)
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    bps = [float(v) for v in f.domain.breakpoints]
    for a, b in zip(bps[:-1], bps[1:]):
        u = np.linspace(a, b, 200)
        ax.plot(u, np.asarray(f(u)), 'r', lw=1.4)
    ax.axis(AX)
    ax.grid(True)
    _save(fig)

    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.plot(np.asarray(xr(ts)), np.asarray(yr(ts)), 'r', lw=1.4)
    ax.axis(AX)
    ax.grid(True)
    _save(fig)
    print("done: f2 length", len(f2))


if __name__ == "__main__":
    run()
