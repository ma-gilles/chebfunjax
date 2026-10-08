"""Visualizing conformal maps: source Chebfun transformations and five figures.

Translation of complex/ConformalVis.m by Nick Trefethen (December2016).
Original: https://www.chebfun.org/examples/complex/ConformalVis.html
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
import numpy as np
from matplotlib.ticker import StrMethodFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import chebfunjax as cj
from chebfunjax.plotting import PARULA, chebfun_style, matlab_axis_equal, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.scribble import scribble

chebfun_style()
_IMG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "images", "complex"
)
_GRID_REPORT = None


def g(z):
    w = (z + 1) * (jnp.pi / 2)
    return (w.sinh() if isinstance(w, cj.Chebfun) else jnp.sinh(w)) / jnp.sinh(jnp.pi / 2)


def h(w):
    return (w - 1) / (w + 1)


def f(z):
    return h(g(z))


def _figure():
    fig, ax = plt.subplots(figsize=(600 / 72.009, 253 / 72.009))
    ax.set_position([0.13, 0.11, 0.775, 0.815])
    ax.tick_params(labelsize=12)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    return fig, ax


def _save(fig, index):
    _savefig(fig, os.path.join(_IMG, f"ConformalVis_{index:02d}.png"), size=(600, 253), dpi=72.009)
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    s = cj.chebfun(lambda s: s)
    square = (-1j + s).join(1 + 1j * s).join(1j - s).join(-1 - 1j * s)
    E = 4 * ((square + 1) / 2).real() - 1 + 1j * square.imag()
    scales = jnp.arange(1, 10, dtype=jnp.float64) / 10
    Z = [E] + [square * r for r in scales]
    fig, ax = _figure()
    matlab_plot(Z, ax=ax, linewidth=1.2)
    ax.set_xlim(-1.5, 3.5)
    matlab_axis_equal(ax)
    ax.set_xticks([-1, 0, 1, 2, 3])
    ax.set_yticks([-1, 0, 1])
    _save(fig, 1)
    G = [g(z) for z in Z]
    fig, ax = _figure()
    matlab_plot(G, ax=ax, linewidth=1.2)
    ax.axis([-4, 6, -5, 5])
    ax.set_box_aspect(1)
    ax.set_xticks([-2, 0, 2, 4, 6])
    ax.set_yticks([-4, -2, 0, 2, 4])
    _save(fig, 2)
    F = [h(w) for w in G]
    fig, ax = _figure()
    matlab_plot(F, ax=ax, linewidth=1.2)
    ax.margins(x=0, y=0)
    ax.set_xlim(-2, 2)
    matlab_axis_equal(ax)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    _save(fig, 3)
    fig, ax = _figure()
    matlab_plot(F, ax=ax, linewidth=0.5)
    ax.margins(x=0, y=0)
    ax.set_xlim(-2, 2)
    matlab_axis_equal(ax)
    # MATLAB changes the preceding colored lines to.5, then holds these
    # axes while adding the mapped source scribble strokes at1.2.
    for text in [0.7j + scribble(" conformal"), -0.9j + scribble(" mapping")]:
        matlab_plot(f(text), "k", ax=ax, linewidth=1.2)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    _save(fig, 4)
    # The source final contour is explicitly a140²array, not aChebfun2 fit.
    x = jnp.linspace(-5, 3, 140)
    y = jnp.linspace(-4, 4, 140)
    xx, yy = jnp.meshgrid(x, y)
    zz = xx + 1j * yy
    values = jnp.log10(jnp.abs(f(zz)))
    levels = jnp.linspace(-0.7, 0.7, 29)
    fig, ax = _figure()
    ax.set_position([0.295, 0.11, 0.345, 0.815])
    ax.set_box_aspect(1)
    contours = ax.contour(
        np.asarray(x),
        np.asarray(y),
        np.asarray(values),
        levels=np.asarray(levels),
        cmap=PARULA,
        linewidths=0.5 * 100 / 72.009,
    )
    cax = fig.add_axes([0.66, 0.11, 0.032, 0.815])
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=contours.norm, cmap=PARULA), cax=cax)
    bar.set_ticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
    cax.tick_params(labelsize=12)
    cax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.axis([-5, 3, -4, 4])
    _save(fig, 5)
    if _GRID_REPORT is not None:
        host = np.asarray(values)
        record = {
            "grid_shape": list(host.shape),
            "grid_sha256": hashlib.sha256(host.tobytes()).hexdigest(),
            "all_finite": bool(jnp.all(jnp.isfinite(values))),
            "levels": np.asarray(levels).tolist(),
            "contour_segment_counts": [len(v) for v in contours.allsegs],
            "source_math": "public Chebfun g/h for curves; JAX140²literalgrid for finalsourcecontour",
            "palette": "existing approximatePARULA; exacthistoricalpalette unqualified",
        }
        with open(_GRID_REPORT, "w") as file:
            json.dump(record, file, indent=2)


if __name__ == "__main__":
    run()
