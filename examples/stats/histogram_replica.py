"""Histograms of chebfuns and point data.

Faithful replica of stats/Histogram.m by Nick Trefethen
(December 2012): a piecewise-constant histogram chebfun of a wiggly
function via cumsum differences, and a histogram of Dirac deltas
placed at random points.

Original: https://www.chebfun.org/examples/stats/Histogram.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Histogram_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _hist_chebfun_data(f, edges):
    """Bin integrals of f: data[k] = int_{e_k}^{e_{k+1}} f."""
    fsum = f.cumsum()
    return np.array([float(fsum(b)) - float(fsum(a))
                     for a, b in zip(edges[:-1], edges[1:])])


def _plot_hist(ax, edges, data, color='r', lw=2):
    for (a, b), v in zip(zip(edges[:-1], edges[1:]), data):
        ax.plot([a, b], [v, v], color, lw=lw)
        ax.plot([a, a], [0, v], color, lw=0.8, alpha=0.5)
        ax.plot([b, b], [0, v], color, lw=0.8, alpha=0.5)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    f = cj.chebfun(
        lambda x: x / 3 + jnp.cos(2 * x) + 0.5 * jnp.sin(x**2)
        + 0.2 * jnp.sin(27 * x), domain=(0.0, 10.0))
    edges = np.arange(0, 11, dtype=float)
    data = _hist_chebfun_data(f, edges)

    xs = np.linspace(0, 10, 2000)
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.plot(xs, np.asarray(f(xs)), lw=1)
    _plot_hist(ax, edges, data)
    ax.grid(True)
    _save(fig)

    # histogram of 50 random points as Dirac deltas
    rs = np.random.RandomState(5489)
    npts = 50
    xpts = 5 + rs.randn(npts)
    xpts = xpts[(xpts > 0) & (xpts < 10)]
    counts = np.array([np.sum((xpts >= a) & (xpts < b))
                       for a, b in zip(edges[:-1], edges[1:])],
                      dtype=float)

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    for xp in xpts:
        ax.plot([xp, xp], [0, 1], 'b', lw=0.8)
        ax.plot(xp, 1, '^b', ms=5)
    _plot_hist(ax, edges, counts)
    ax.grid(True)
    _save(fig)

    # finer bins (MATLAB: edges = 0:.5:10)
    edges2 = np.arange(0, 10.1, 0.5)
    counts2 = np.array([np.sum((xpts >= a) & (xpts < b))
                        for a, b in zip(edges2[:-1], edges2[1:])],
                       dtype=float)
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.plot(xpts, np.zeros_like(xpts), '.k', ms=6)
    _plot_hist(ax, edges2, counts2)
    ax.set_ylim(-1, counts2.max() + 1)
    ax.grid(True)
    _save(fig)
    print("total mass:", counts.sum(), counts2.sum())


if __name__ == "__main__":
    run()
