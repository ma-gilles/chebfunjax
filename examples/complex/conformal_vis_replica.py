"""Visualizing conformal maps.

Faithful replica of complex/ConformalVis.m by Nick Trefethen
(December 2016): mapping an infinite half-strip to the unit disk as
the composition of sinh and a Mobius transformation, visualized with
a quasimatrix of concentric squares built with `join`.

Original: https://www.chebfun.org/examples/complex/ConformalVis.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.scribble import scribble

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'complex')

FIG = [0]


def _sample_pieces(cf, n=120):
    """Sample a (possibly piecewise) complex chebfun piece by piece."""
    bps = list(cf.domain.breakpoints)
    out = []
    for a, b in zip(bps[:-1], bps[1:]):
        t = np.linspace(a, b, n)
        out.append(np.asarray(cf(t)))
    return out


def _plot_cf(ax, cf, fn=None, color=None, lw=1.2):
    for zv in _sample_pieces(cf):
        v = fn(zv) if fn is not None else zv
        ax.plot(v.real, v.imag, color=color, lw=lw)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"ConformalVis_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    s = cj.chebfun(lambda x: x)
    unitsquare = ((-1j + s).join(1 + 1j * s)
                  .join(1j - s).join(-1 - 1j * s))
    E = 4 * (unitsquare.real() + 1) / 2 - 1 + 1j * unitsquare.imag()
    Z = [E] + [float(r) * unitsquare for r in np.arange(0.1, 0.95, 0.1)]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for i, cf in enumerate(Z):
        _plot_cf(ax, cf, color=f"C{i % 10}")
    ax.set_xlim(-1.5, 3.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(-1, 4))
    ax.set_yticks(range(-1, 2))
    _save(fig)

    g = lambda z: np.sinh(np.pi * (z + 1) / 2) / np.sinh(np.pi / 2)  # noqa: E731

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    for i, cf in enumerate(Z):
        _plot_cf(ax, cf, fn=g, color=f"C{i % 10}")
    ax.axis([-4, 6, -5, 5])
    ax.set_aspect("equal")
    ax.set_xticks(range(-2, 7, 2))
    ax.set_yticks(range(-4, 5, 2))
    _save(fig)

    h = lambda w: (w - 1) / (w + 1)   # noqa: E731
    f = lambda z: h(g(z))             # noqa: E731

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    for i, cf in enumerate(Z):
        _plot_cf(ax, cf, fn=f, color=f"C{i % 10}", lw=0.5)
    ax.set_xlim(-2, 2)
    ax.set_aspect("equal")
    ax.set_xticks(range(-1, 2))
    ax.set_yticks(range(-1, 2))
    _save(fig)

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    for i, cf in enumerate(Z):
        _plot_cf(ax, cf, fn=f, color=f"C{i % 10}", lw=0.5)
    for txt in (0.7j + scribble(' conformal'),
                -0.9j + scribble(' mapping')):
        _plot_cf(ax, txt, fn=f, color='k')
    ax.set_xlim(-2, 2)
    ax.set_aspect("equal")
    ax.set_xticks(range(-1, 2))
    ax.set_yticks(range(-1, 2))
    _save(fig)

    x = np.linspace(-5, 3, 140)
    y = np.linspace(-4, 4, 140)
    xx, yy = np.meshgrid(x, y)
    zz = xx + 1j * yy
    with np.errstate(all="ignore"):
        L = np.log10(np.abs(f(zz)))
    fig, ax = plt.subplots(figsize=(6.9, 6.2))
    cs = ax.contour(x, y, L, levels=np.arange(-0.7, 0.71, 0.05))
    fig.colorbar(cs, ax=ax)
    ax.set_xticks(range(-4, 5, 2))
    ax.set_yticks(range(-4, 5, 2))
    ax.axis([-5, 3, -4, 4])
    ax.set_box_aspect(1)
    _save(fig)
    print("figures:", FIG[0])


if __name__ == "__main__":
    run()
