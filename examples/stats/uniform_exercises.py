"""Probability exercises: uniform distributions.

Translation of stats/UniformExercises.m by Jie Gao and Nick
Trefethen (June 2013): quartiles of a uniform density; recovering
(a, b) from mean and variance via chebfun2 roots and via 1D
substitution; and wheel-of-fortune conditional probabilities.

Original: https://www.chebfun.org/examples/stats/UniformExercises.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"UniformExercises_{FIG[0]:02d}.png"))
    plt.close(fig)


def _show(name, v):
    """MATLAB ``format long`` display of a real scalar/vector/matrix."""
    print(f"{name} =")
    for row in np.atleast_2d(np.asarray(v, dtype=float)):
        print("".join(f"{x:20.15f}" for x in row))


def _area(ax, f, color):
    """area(f): fill under a chebfun, with MATLAB's black edge."""
    a, b = f.domain.support
    xs = np.linspace(a, b, 200)
    ys = np.asarray(f(jnp.asarray(xs)))
    ax.fill_between(xs, ys, color=color, edgecolor="k", linewidth=0.5)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    LW = 2

    f = cj.chebfun(lambda x: 1 / (2 - 1) + 0 * x, domain=(1.0, 2.0))
    fint = f.cumsum()
    mu_x = cj.chebfun(lambda x: x * f(x), domain=(1.0, 2.0)).sum()
    _show("mu_x", mu_x)
    a = float((1 - fint - 1 / 4).roots()[0])
    _show("a", a)
    _show("z", a - float(mu_x))
    fig, ax = plt.subplots()
    _area(ax, f.restrict(a, 2.0), (0.3, 0.2, 0.5))
    ax.set_xlim(1, 2)
    ax.set_ylim(0, 2)
    matlab_plot(f, "k", ax=ax, lw=LW)
    ax.grid(True)
    _save(fig)

    mean = 1
    var = 4 / 3
    f = cj.chebfun2(lambda a, b: (a + b) / 2 - mean, domain=(-5, 5, -5, 5))
    g = cj.chebfun2(lambda a, b: (b - a) ** 2 / 12 - var,
                    domain=(-5, 5, -5, 5))
    r = np.asarray(f.roots(g))
    _show("r", r)
    a = float(np.min(r[1, :]))
    b = float(np.max(r[1, :]))
    _show("a", a)
    _show("b", b)
    f = cj.chebfun(lambda x: 0 * x + 1 / (b - a), domain=(a, b))
    fint = f.cumsum()
    _show("p", fint(0.0))
    fig, ax = plt.subplots()
    _area(ax, f.restrict(a, 0.0), (0.75, 0.3, 0.2))
    ax.set_xlim(a, b)
    ax.set_ylim(0, 0.5)
    matlab_plot(f, "k", ax=ax, lw=1.6)
    ax.grid(True)
    _save(fig)

    def b(a):
        return 2 - a
    print("b = ")
    print("    @(a)2-a")
    g = cj.chebfun(lambda a: (b(a) - a) ** 2 / 12 - 4 / 3, domain=(-5.0, 5.0))
    aa = np.asarray(g.roots())
    _show("aa", aa[:, None])
    a = float(aa[0])
    b = float(aa[1])
    _show("a", a)
    _show("b", b)
    fig, ax = plt.subplots()
    matlab_plot(g, ax=ax, lw=2)
    ax.plot([-5, 5], [0, 0], "-k")
    ax.autoscale()
    ax.plot(aa, np.asarray(g(jnp.asarray(a))) + 0 * aa, "r.", ms=20)
    ax.grid(True)
    _save(fig)

    meanab = cj.chebfun2(lambda a, b: (a + b) / 2, domain=(-5, 5, -5, 5))
    varab = cj.chebfun2(lambda a, b: (b - a) ** 2 / 12, domain=(-5, 5, -5, 5))
    ab = np.asarray((meanab - 1).roots(varab - 4 / 3))
    _show("ab", ab)
    a = float(ab[0, 1])
    b = float(ab[0, 0])
    _show("a", a)
    _show("b", b)

    # wheel of fortune: uniform on [0, 360]
    f = cj.chebfun(lambda x: 1 / (360 - 0) + 0 * x, domain=(0.0, 360.0))
    fig, ax = plt.subplots()
    for lo, hi, col in [(0, 5, (1, 0, 0)), (5, 20, (0, 1, 1)),
                        (20, 55, (1, 1, 0)), (55, 105, (0, 1, 0)),
                        (105, 170, (1, 1, 1)), (170, 250, (0, 0, 1)),
                        (250, 360, (0, 0, 0))]:
        _area(ax, f.restrict(float(lo), float(hi)), col)
    matlab_plot(f, "k", ax=ax, lw=LW)
    ax.grid(True)
    _save(fig)
    fint = f.cumsum()
    _show("p1", fint(5.0 + 15))
    _show("p1_exact", (5 + 15) / 360)
    fig, ax = plt.subplots()
    matlab_plot(f, "k", ax=ax, lw=LW)
    ax.grid(True)
    _area(ax, f.restrict(0.0, 20.0), (0.7, 0, 0.6))
    ax.set_xlim(0, 360)
    _save(fig)

    pnb = 1 - float(fint(80.0))
    _show("pnb", pnb)
    pnyb = 1 - float(fint(35.0)) - float(fint(110.0))
    _show("pnyb", pnyb)
    pn = pnyb - float(fint(80.0))
    _show("pn", pn)
    p2 = pn / pnb
    _show("p2", p2)
    _show("p2_exact", (1 - (35 + 110 + 80) / 360) / (1 - 80 / 360))

    g = f / (280 / 360)
    g = g * cj.chebfun([g, 0, g], domain=[0, 170, 250, 360])
    fig, ax = plt.subplots()
    _area(ax, g.restrict(0.0, 20.0), (0.3, 0.5, 0.2))
    _area(ax, g.restrict(55.0, 170.0), (0.3, 0.5, 0.2))
    matlab_plot(g, "b", ax=ax, lw=LW)
    ax.set_xlim(0, 360)
    ax.grid(True)
    _save(fig)


if __name__ == "__main__":
    run()
