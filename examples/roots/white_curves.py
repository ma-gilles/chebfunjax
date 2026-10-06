"""The white curves of Ortiz and Rivlin.

Translation of roots/WhiteCurves.m by Stefan Guettel
(November 2011): the white curves visible in plots of superimposed
Chebyshev polynomials T_1..T_30, given by T_{n-m}(x) = T_2(y), and
the analogous weighted picture for Legendre polynomials.

Original: https://www.chebfun.org/examples/roots/WhiteCurves.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import chebpoly, legpoly
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, "..", "..", "docs", "images", "roots")
FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    # Display adapter measured from all four cached chebfun.org PNGs:
    # 600x420 canvas, axes frame x=78..543/y=30..374, .2 ticks.
    # Explicit settings translate the reference graphics into Matplotlib;
    # they are not additional mathematical operations from the MATLAB cell.
    for ax in fig.axes:
        ax.set_position((78 / 600, 46 / 420, 465 / 600, 344 / 420))
        ticks = np.arange(-10, 11, 2) / 10
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontfamily("Arial")
        for line in ax.lines:
            line.set_linewidth(0.5)
            if line.get_marker() == ".":
                # Reference isolated dots occupy about 3x3 pixels; preserve
                # endpoint glyphs extending across the axes frame.
                line.set_markersize(4)
                line.set_markeredgewidth(0)
                line.set_clip_on(False)
    _savefig(
        fig, os.path.join(_IMG, f"WhiteCurves_{FIG[0]:02d}.png"),
        size=(600, 420),
    )


def _roots_against_level(f, level):
    """Use the public Chebfun root finder for f(x) = level."""
    return np.asarray((f - level).roots())


def run():
    os.makedirs(_IMG, exist_ok=True)

    # plot(chebpoly(1:30), 'b-'), hold on
    T = chebpoly(np.arange(1, 31))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    matlab_plot(T, "b-", ax=ax)
    ax.axis([-1, 1, -1, 1])
    _save(fig)

    # Chebfun roots of T_j(x) - T_2(y), overlaid on the first panel.
    T2 = chebpoly(2)
    for j in range(1, 5):
        Tj = chebpoly(j)
        for y in np.linspace(-1, 1, 200):
            xroots = _roots_against_level(Tj, T2(y))
            ax.plot(xroots, np.full(xroots.shape, y), "r.")
    ax.axis([-1, 1, -1, 1])
    _save(fig)
    plt.close(fig)

    # plot(L.*q): form the weighted Legendre Chebfuns, then use matlab_plot.
    # The source bounded-real-Singfun plotData path now exists in matlab_plot;
    # this page’s particular product representation and rendered values remain
    # unqualified until the example is run and its figures are compared.
    x = cj.chebfun("x")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for j in range(1, 31):
        L = legpoly(j)
        q = (np.pi * j / 2) ** 0.5 * (1 - x**2) ** 0.25
        matlab_plot(L * q, color=(0.6, 0.4, 0), ax=ax)
    ax.axis([-1, 1, -1, 1])
    _save(fig)
    plt.close(fig)

    # Chebfun roots of L_j(x) - L_2(y).
    L2 = legpoly(2)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for j in range(1, 5):
        Lj = legpoly(j)
        for y in np.linspace(-1, 1, 200):
            xroots = _roots_against_level(Lj, L2(y))
            if xroots.size:
                ax.plot(xroots, np.full(xroots.shape, y), "r.")
    ax.axis([-1, 1, -1, 1])
    _save(fig)
    plt.close(fig)


if __name__ == "__main__":
    run()
