"""Absolute value approximations by rationals.

Translation of approx/AbsoluteValue.m by Nick Trefethen (May
2011): Newton's method for r^2 = x^2 generates type (2^k, 2^k)
rational approximations to |x|, with maximum error exactly 2^-k after
k steps but much faster convergence away from x = 0.

Original: https://www.chebfun.org/examples/approx/AbsoluteValue.html
Copyright by The University of Oxford and The Chebfun Developers.
"""

import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, "..", "..", "docs", "images", "approx")


def _newton_grid(x, r, fname, label):
    """Six Newton steps in a 3x2 subplot grid; returns final r."""
    fig = plt.figure(figsize=(6, 2.7))
    # Framing measured from the cached publication, not data.
    lefts = (0.1292, 0.5692)
    bottoms = (0.7204, 0.4204, 0.1204)
    for k in range(6):
        ax = fig.add_axes((lefts[k % 2], bottoms[k // 2], 0.335, 0.1852))
        matlab_plot(r, ax=ax, linewidth=1.6)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.2, 1.2)
        ax.grid(True, color="#d9d9d9", linewidth=0.5)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticklabels(["-1", "-0.5", "0", "0.5", "1"])
        ax.set_yticklabels(["0", "0.5", "1"])
        ax.tick_params(labelsize=7, length=2, pad=1.5)
        err = float((r - x.abs()).norm(jnp.inf))
        n = len(r)
        # Renderer point sizes follow the cached publication glyph bounds,
        # separately from the native figure metadata fontsize=12 points.
        ax.set_title(f"error={err:4.1e}   {label}{n}", fontsize=7.5, fontweight="bold", pad=2.5)
        r = (r**2 + x**2) / (2 * r)
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(_IMG, fname))
    plt.close(fig)
    return r


def _error_plot(x, r, fname):
    fig, ax = plt.subplots(figsize=(6, 2.7))
    # Construct the source error Chebfun before requesting its plotData.
    error = (r - x.abs()).abs()
    matlab_plot(error, ax=ax, linewidth=1.6)
    ax.set_yscale("log", nonpositive="mask")
    ax.set_position((0.13, 0.11, 0.775, 0.815))
    ax.set_xlim(-1, 1)
    ax.set_ylim(1e-18, 10)
    ax.grid(True, color="#d9d9d9", linewidth=0.5)
    ticks = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([format(value, "g") for value in ticks])
    ax.set_yticks([1e-15, 1e-10, 1e-5, 1])
    ax.tick_params(labelsize=8)
    ax.set_xlabel("x", fontsize=9, labelpad=-0.5)
    ax.set_title("Error", fontsize=9, fontweight="bold", pad=2.5)
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(_IMG, fname))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Newton iteration for r^2 = x^2, no breakpoint
    x = cj.chebfun("x")
    r = cj.chebfun("1")
    _newton_grid(x, r, "AbsoluteValue_01.png", "len=")

    # With a breakpoint at x = 0 the lengths stay modest
    x = cj.chebfun("x", domain=[-1.0, 0.0, 1.0])
    r = cj.chebfun("1", domain=[-1.0, 0.0, 1.0])
    r = _newton_grid(x, r, "AbsoluteValue_02.png", "length = ")

    # Error after six steps
    _error_plot(x, r, "AbsoluteValue_03.png")

    # Six more steps
    r = _newton_grid(x, r, "AbsoluteValue_04.png", "length = ")

    # Error after twelve steps
    _error_plot(x, r, "AbsoluteValue_05.png")


if __name__ == "__main__":
    run()
