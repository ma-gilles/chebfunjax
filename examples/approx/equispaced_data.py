"""Chebfuns from equispaced data.

Translation of approx/EquispacedData.m by Nick Trefethen (April
2015): constructing a chebfun from equispaced samples via Floater-Hormann
rational approximation ('equi'), versus the catastrophic polynomial interpolant,
plus truncation, loosened tolerance, and noisy data.

Original: https://www.chebfun.org/examples/approx/EquispacedData.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.plotting import chebfun_style, matlab_plot, plotcoeffs
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

PURPLE = (0.8, 0, 1)
FIGURE_SIZE = (598, 273)


def _historical_axes(ax, title=None, fontsize=10):
    # The cached page uses the older connected coefficient curves and larger
    # renderer fonts, rather than the pinned library default dot markers.
    ax.tick_params(labelsize=18, direction="in", top=True, right=True, length=3)
    ax.grid(True, color="0.8", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.6)
    if title is not None:
        ax.set_title(title, fontsize=fontsize * 4 / 3, fontweight="bold", pad=4)



def _dataplot(f, grid, data, title, fname, fontsize=10):
    fig, ax = plt.subplots(figsize=(598 / 72.009, 273 / 72.009))
    matlab_plot(f, "b", ax=ax, linewidth=1)
    ax.plot(grid, data, '.k', ms=8)
    _historical_axes(ax, title, fontsize)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    fig.set_facecolor("white")
    ax.set_position([0.13, 0.11, 0.775, 0.815])
    _savefig(fig, os.path.join(_IMG, fname), size=FIGURE_SIZE, dpi=72.009)
    plt.close(fig)


def _coeffplot(f, title, fname):
    fig, ax = plt.subplots(figsize=(598 / 72.009, 273 / 72.009))
    plotcoeffs(f, ax=ax, color=PURPLE, source=True, fmt="-", linewidth=1)
    ax.axis([0, 100, 1e-16, 10])
    _historical_axes(ax, title)
    ax.set_yticks([1e-10, 1])
    ax.minorticks_off()
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    fig.set_facecolor("white")
    ax.set_position([0.13, 0.185, 0.775, 0.735])
    _savefig(fig, os.path.join(_IMG, fname), size=FIGURE_SIZE, dpi=72.009)
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    ff = lambda x: np.exp(x) * np.cos(10 * x) * np.tanh(4 * x)  # noqa: E731
    grid = np.linspace(-1, 1, 40)
    data = ff(grid)
    f = cj.chebfun(jnp.asarray(data), equi=True)
    _dataplot(f, grid, data,
              "chebfun constructed from 40 equispaced data values",
              "EquispacedData_01.png")

    fexact = cj.chebfun(
        lambda x: jnp.exp(x) * jnp.cos(10 * x) * jnp.tanh(4 * x))
    print("error =")
    print(f"     {float((f - fexact).norm(np.inf)):.15e}")

    # The polynomial interpolant through the same data: Runge disaster
    runge = Chebfun.interp1(jnp.asarray(grid), jnp.asarray(data))
    fig, ax = plt.subplots(figsize=(598 / 72.009, 273 / 72.009))
    matlab_plot(runge, 'r', ax=ax, linewidth=1)
    _historical_axes(ax)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.plot(grid, data, '.k', ms=8)
    fig.set_facecolor("white")
    ax.set_position([0.13, 0.11, 0.775, 0.815])
    _savefig(fig, os.path.join(_IMG, "EquispacedData_02.png"), size=FIGURE_SIZE, dpi=72.009)
    plt.close(fig)

    print("f =")
    print(repr(f))
    _coeffplot(f, "Chebyshev coefficients", "EquispacedData_03.png")

    # Interpolate f in 51 Chebyshev points (degree 50)
    f50 = cj.chebfun(f, n=51)
    print("error50 =")
    print(f"     {float((f50 - fexact).norm(np.inf)):.15e}")
    _coeffplot(f50, "Chebyshev coefficients up to degree 50",
               "EquispacedData_04.png")

    # Loosened tolerance
    floose = cj.chebfun(jnp.asarray(data), equi=True, eps=1e-6)
    print("errorloose =")
    print(f"     {float((floose - fexact).norm(np.inf)):.15e}")
    _coeffplot(floose, "Chebyshev coefficients with loosened tolerance",
               "EquispacedData_05.png")

    # NumPy normals do not match the MATLAB rng(0) normal stream.
    # These two figures remain illustrative until source draws are captured.
    rs = np.random.RandomState(5489)
    noisy = data + 1e-1 * rs.standard_normal(data.shape)
    for ep, lab, fn in ((1e-2, "1e-2", "EquispacedData_06.png"),
                        (3e-2, "3e-2", "EquispacedData_07.png")):
        fn_ = cj.chebfun(jnp.asarray(noisy), equi=True, eps=ep)
        _dataplot(fn_, grid, noisy,
                  f"noisy data with 'equi', eps = {lab}: "
                  f"length(f) = {len(fn_)}", fn, fontsize=12)


if __name__ == "__main__":
    run()
