"""Marching squares for bivariate rootfinding.

Translation of roots/MarchingSquares.m by Alex Townsend
(March 2013): common zeros of pairs of chebfun2 objects located by
marching squares, including the Trott curve and the critical points
of a bivariate function.

Original: https://www.chebfun.org/examples/roots/MarchingSquares.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import hashlib
import json
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt

# uses-numpy: Matplotlib rendering boundary, artist recording and root markers
import numpy as np
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, curve_plot_data
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.environ.get('MARCHINGSQUARES_FIGURES', os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots'))
_ARTISTS = os.environ.get('MARCHINGSQUARES_ARTISTS')

FIG = [0]


def _plot_curves(ax, curves, color):
    for c in curves:
        data = curve_plot_data(c)
        ax.plot(np.asarray(data["xLine"]), np.asarray(data["yLine"]), color=color, lw=1.6)


def _source_tick_label(value, _position):
    """Format fixed source ticks without Matplotlib's trailing .0."""
    return format(value, ".15g")


def _save(fig, close=True):
    FIG[0] += 1
    fig.set_facecolor("white")
    ax = fig.axes[0]
    # MATLAB chebsite 2016 figFormats.m uses this normalized axes rectangle.
    ax.set_position([0.13, 0.11, 0.775, 0.815])
    ax.xaxis.set_major_formatter(FuncFormatter(_source_tick_label))
    ax.yaxis.set_major_formatter(FuncFormatter(_source_tick_label))
    artists = []
    for line in ax.lines:
        x = np.asarray(line.get_xdata())
        y = np.asarray(line.get_ydata())
        artists.append({
            "label": line.get_label(), "color": line.get_color(),
            "linestyle": line.get_linestyle(), "linewidth": line.get_linewidth(),
            "marker": line.get_marker(), "markersize": line.get_markersize(),
            "x_shape": list(x.shape), "y_shape": list(y.shape),
            "x_data": x.tolist(), "y_data": y.tolist(),
            "x_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "y_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        })
    _savefig(fig, os.path.join(_IMG, f"MarchingSquares_{FIG[0]:02d}.png"), dpi=72.009)
    if _ARTISTS:
        record = {
            "figure_index": FIG[0], "source_case": [1, 1, 2, 3][FIG[0]-1],
            "canvas_px": [600, 270], "export_dpi": 72.009,
            "axes_position": list(ax.get_position().bounds),
            "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
            "xticks": np.asarray(ax.get_xticks()).tolist(),
            "yticks": np.asarray(ax.get_yticks()).tolist(),
            "xticklabels": [t.get_text() for t in ax.get_xticklabels()],
            "yticklabels": [t.get_text() for t in ax.get_yticklabels()],
            "source_linewidth_pt": 1.6, "source_marker_size_pt": 20,
            "matplotlib_marker_size_adapter": 10,
            "curve_sampling_rule": "source_chebtech_prolong_per_fun",
            "artists": artists,
        }
        with open(_ARTISTS, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    if close:
        plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    d = (-4.0, 4.0, -4.0, 4.0)
    f = cj.chebfun2(
        lambda x, y: 2 * y * jnp.cos(y**2) * jnp.cos(2 * x)
        - jnp.cos(y), domain=d)
    g = cj.chebfun2(
        lambda x, y: 2 * jnp.sin(y**2) * jnp.sin(2 * x)
        - jnp.sin(x), domain=d)
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    _plot_curves(ax, g.roots(), (1.0, 0.0, 0.0))
    _plot_curves(ax, f.roots(), (0.0, 1.0, 0.0))
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-4.0, 4.0)
    ax.set_xticks((-3, -2, -1, 0, 1, 2, 3))
    ax.set_yticks((-4, -3, -2, -1, 0, 1, 2, 3, 4))
    _save(fig, close=False)
    r = np.atleast_2d(np.asarray(f.roots(g, method="ms")))
    ax.plot(r[:, 0], r[:, 1], '.k', ms=10)                  # hold on
    _save(fig)

    trott = cj.chebfun2(
        lambda x, y: 144 * (x**4 + y**4) - 225 * (x**2 + y**2)
        + 350 * x**2 * y**2 + 81)
    g = cj.chebfun2(lambda x, y: y - x**6)
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    _plot_curves(ax, trott.roots(), 'b')
    _plot_curves(ax, g.roots(), 'r')
    r = np.atleast_2d(np.asarray(trott.roots(g, method="ms")))
    ax.plot(r[:, 0], r[:, 1], 'k.', ms=10)
    # The MATLAB source uses axis equal while retaining the wide axes box.
    # datalim adjusts the data limits instead of shrinking the axes box.
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks((-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2))
    ax.set_yticks((-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8))
    ax.set_aspect("equal", adjustable="datalim")
    _save(fig)

    f = cj.chebfun2(
        lambda x, y: (x**2 - y**3 + 1 / 8) * jnp.sin(10 * x * y))
    fx = f.diff(dim=2)   # d/dx
    fy = f.diff(dim=1)   # d/dy
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    _plot_curves(ax, fx.roots(), 'b')
    _plot_curves(ax, fy.roots(), 'r')
    r = np.atleast_2d(np.asarray(fx.roots(fy, method="ms")))
    ax.plot(r[:, 0], r[:, 1], 'k.', ms=10)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks((-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8))
    ax.set_yticks((-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1))
    _save(fig)


if __name__ == "__main__":
    run()
