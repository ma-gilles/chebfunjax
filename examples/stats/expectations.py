"""Mean, median, mode of probability distributions.

Translation of stats/Expectations.m by Mark Richardson (May 2011): moments of an exponential density and the
mean, median and mode of a polynomial density, all as chebfun
computations.

Original: https://www.chebfun.org/examples/stats/Expectations.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')

FIG = [0]


def _plot(f, dom, ylim, ylab):
    FIG[0] += 1
    xs = np.linspace(*dom, 700)
    fig, ax = plt.subplots(figsize=(5.2, 1.8), dpi=100)
    ax.plot(xs, np.asarray(f(xs)), lw=1.6)
    ax.grid(True)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MultipleLocator(5 if dom[1] == 40 else .5))
    ax.set_xlabel("x", labelpad=0)
    ax.set_ylabel(ylab, rotation=0)
    ax.yaxis.set_label_coords(-.06, .5)
    ax.yaxis.set_major_locator(MultipleLocator(.5 if dom[1] == 40 and ylim[1] > 2 else .05 if ylim[1] <= .31 else .1 if ylim[1] <= .61 else .2))
    fig.set_facecolor("white")
    fig.subplots_adjust(left=.13, bottom=.17, right=.905, top=.925)
    ax.tick_params(labelsize=7.5, top=True, right=True)
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    ax.grid(True, alpha=.3, linewidth=.5)
    _savefig(fig, os.path.join(
        _IMG, f"Expectations_{FIG[0]:02d}.png"), size=(520, 180))
    plt.close(fig)
    return fig


def run():
    os.makedirs(_IMG, exist_ok=True)

    x = cj.chebfun(lambda t: t, domain=(0.0, 40.0))
    f = 2 * (-2 * x).exp()
    _plot(f, (0, 40), (-0.2, 2.2), "f(x)")
    print("ans =")
    print(f"   {float(f.sum()):.15f}")

    xf = x * f
    _plot(xf, (0, 40), (-0.05, 0.4), "x f(x)")
    print("ans =")
    print(f"   {float(xf.sum()):.15f}")

    xxf = x**2 * f
    _plot(xxf, (0, 40), (-0.03, 0.31), "$x^2$ f(x)")
    print("ans =")
    print(f"   {float(xxf.sum()):.15f}")

    x = cj.chebfun(lambda t: t, domain=(0.0, 3.0))
    g = 4 * x * (9 - x**2) / 81
    _plot(g, (0, 3), (-0.01, 0.61), "g(x)")
    mean = float((x * g).sum())
    print("mean =")
    print(f"   {mean:.15f}")

    G = g.cumsum()
    _plot(G, (0, 3), (0, 1), "G(x)")
    median = float(np.asarray((G - 0.5).roots())[0])
    print("median =")
    print(f"   {median:.15f}")
    print("median_exact =")
    print(f"   {np.sqrt(9 - 9 * np.sqrt(2) / 2):.15f}")

    mode, gmax = g.max()
    mode = float(mode)
    print("mode =")
    print(f"   {mode:.15f}")
    print("mode_exact =")
    print(f"   {np.sqrt(3):.15f}")

    FIG[0] += 1
    xs = np.linspace(0, 3, 500)
    fig, ax = plt.subplots(figsize=(5.2, 1.8), dpi=100)
    ax.plot(xs, np.asarray(g(xs)), lw=1.6)
    ax.grid(True)
    for pos, col, lab, tx in ((mean, 'r', 'mean', 0.2),
                              (median, 'm', 'median', 1.2),
                              (mode, 'k', 'mode', 2.2)):
        ax.plot([pos, pos], [0, float(g(pos))], '-' + col, lw=1.6)
        ax.text(tx, 0.55, f"{lab} = {pos:1.2f}", color=col, fontsize=7.5)
    ax.set_ylim(-0.01, 0.61)
    ax.xaxis.set_major_locator(MultipleLocator(.5))
    ax.set_xlabel("x", labelpad=0)
    ax.set_ylabel("g(x)", rotation=0)
    ax.yaxis.set_label_coords(-.06, .5)
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    fig.set_facecolor("white")
    fig.subplots_adjust(left=.13, bottom=.17, right=.905, top=.925)
    ax.tick_params(labelsize=7.5, top=True, right=True)
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    ax.grid(True, alpha=.3, linewidth=.5)
    _savefig(fig, os.path.join(
        _IMG, f"Expectations_{FIG[0]:02d}.png"), size=(520, 180))
    plt.close(fig)


if __name__ == "__main__":
    run()
