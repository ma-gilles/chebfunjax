"""Rouche's theorem: source Chebfun curves and all seven figures.

Translation of complex/RoucheTheorem.m by Anthony Austin (November 2012,
revised October2020). Original: https://www.chebfun.org/examples/complex/RoucheTheorem.html
Copyright by The University of Oxford and The Chebfun Developers.
"""

import matplotlib

matplotlib.use("Agg")
import json
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from matplotlib.ticker import StrMethodFormatter

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, matlab_axis_equal, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_IMG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "images", "complex"
)

_ROOTS_REPORT = None


def _figure():
    fig, ax = plt.subplots(figsize=(600 / 72.009, 253 / 72.009))
    ax.set_position([0.13, 0.14, 0.775, 0.765])
    ax.tick_params(labelsize=12)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    return fig, ax


def _save(fig, index):
    _savefig(fig, os.path.join(_IMG, f"RoucheTheorem_{index:02d}.png"), size=(600, 253), dpi=72.009)
    plt.close(fig)


def _norms(f, g, index, upper):
    fig, ax = _figure()
    matlab_plot([f.abs(), (f - g).abs()], ax=ax, linewidth=1.4)
    ax.set_title("|f| (above) and |f - g| (below) on the unit circle", fontsize=14)
    ax.set_xlabel("t", fontsize=14)
    ax.axis([0, 2 * np.pi, 0, upper])
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    _save(fig, index)


def _complex(f, g, index):
    fig, ax = _figure()
    matlab_plot([f, g], ax=ax, linewidth=1.4)
    ax.set_title("Images of the unit circle under f and g", fontsize=14)
    ax.set_xlabel("Re", fontsize=14)
    ax.set_ylabel("Im", fontsize=14)
    ax.margins(x=0, y=0)
    ax.grid(True, linewidth=0.5)
    matlab_axis_equal(ax)
    if index == 6:
        ax.set_xticks(list(range(-40, 41, 10)))
    _save(fig, index)


def _ratio(f, g, index):
    fig, ax = _figure()
    matlab_plot(g / f, ax=ax, linewidth=1.4)
    ax.set_title("Image of the unit circle under g/f", fontsize=14)
    ax.set_xlabel("Re", fontsize=14)
    ax.set_ylabel("Im", fontsize=14)
    ax.axis([0, 1.5, -0.5, 0.5])
    ax.grid(True, linewidth=0.5)
    matlab_axis_equal(ax)
    ax.set_xticks([0, 0.5, 1, 1.5, 2], ["0", "0.5", "1", "1.5", "2"])
    ax.set_yticks([-0.5, 0, 0.5], ["-0.5", "0", "0.5"])
    _save(fig, index)


def run():
    os.makedirs(_IMG, exist_ok=True)
    t = cj.chebfun(lambda t: t, domain=[0, 2 * np.pi])
    z = (1j * t).exp()
    f = z
    g = z.sin()
    _norms(f, g, 1, 1.1)
    _complex(f, g, 2)
    _ratio(f, g, 3)
    f = 15 * z**3
    g = z**7 - 2 * z**5 + 15 * z**3 - z + 1
    _norms(f, g, 4, 16)
    p = jnp.asarray([1.0, 0, -2, 0, 15, 0, -1, 1], dtype=jnp.float64)
    r = jnp.roots(p, strip_zeros=False)
    fig, ax = _figure()
    matlab_plot(z, ax=ax, linewidth=1.4)
    # Numeric polynomial roots are a JAX calculation; only Matplotlib gets
    # the explicit host arrays used to draw the seven source markers.
    ax.plot(
        np.asarray(r.real),
        np.asarray(r.imag),
        "o",
        markersize=7 * 100 / 72.009,
        markerfacecolor="none",
    )
    ax.set_title("Roots of g", fontsize=14)
    ax.set_xlabel("Re", fontsize=14)
    ax.set_ylabel("Im", fontsize=14)
    ax.axis([-1.5, 1.5, -1.5, 1.5])
    ax.grid(True, linewidth=0.5)
    matlab_axis_equal(ax)
    _save(fig, 5)
    _complex(f, g, 6)
    _ratio(f, g, 7)
    record = {
        "root_count": len(r),
        "roots_inside_unit_circle": int(jnp.sum(jnp.abs(r) < 1)),
        "roots": np.asarray(jnp.stack([r.real, r.imag], axis=1)).tolist(),
        "polynomial_residual_max": float(jnp.max(jnp.abs(jnp.polyval(p, r)))),
        "source_figures": 7,
        "new_numerical_math": "public Chebfun JAX arithmetic and jnp.roots",
    }
    if _ROOTS_REPORT is not None:
        with open(_ROOTS_REPORT, "w") as file:
            json.dump(record, file, indent=2)


if __name__ == "__main__":
    run()
