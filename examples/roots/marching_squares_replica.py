"""Marching squares for bivariate rootfinding.

Faithful replica of roots/MarchingSquares.m by Alex Townsend
(August 2013): common zeros of pairs of chebfun2 objects located by
marching squares, including the Trott curve and the critical points
of a bivariate function.

Original: https://www.chebfun.org/examples/roots/MarchingSquares.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')

FIG = [0]


def _plot_curves(ax, curves, color):
    for c in curves:
        bps = list(c.domain.breakpoints)
        for a, b in zip(bps[:-1], bps[1:]):
            t = np.linspace(a, b, 300)
            v = np.asarray(c(t))
            ax.plot(v.real, v.imag, color=color, lw=1.4)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"MarchingSquares_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
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
    _plot_curves(ax, g.roots(), 'r')
    _plot_curves(ax, f.roots(), 'g')
    r = np.atleast_2d(np.asarray(f.roots(g, method="ms")))
    ax.plot(r[:, 0], r[:, 1], '.k', ms=10)
    ax.set_aspect("equal")
    _save(fig)
    print(f"case 1: {len(r)} common zeros")

    trott = cj.chebfun2(
        lambda x, y: 144 * (x**4 + y**4) - 225 * (x**2 + y**2)
        + 350 * x**2 * y**2 + 81)
    g = cj.chebfun2(lambda x, y: y - x**6)
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    _plot_curves(ax, trott.roots(), 'b')
    _plot_curves(ax, g.roots(), 'r')
    r = np.atleast_2d(np.asarray(trott.roots(g, method="ms")))
    ax.plot(r[:, 0], r[:, 1], 'k.', ms=10)
    ax.set_aspect("equal")
    _save(fig)
    print(f"case 2 (Trott curve): {len(r)} common zeros")

    f = cj.chebfun2(
        lambda x, y: (x**2 - y**3 + 1 / 8) * jnp.sin(10 * x * y))
    fx = f.diff(dim=2)   # d/dx
    fy = f.diff(dim=1)   # d/dy
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    _plot_curves(ax, fx.roots(), 'b')
    _plot_curves(ax, fy.roots(), 'r')
    r = np.atleast_2d(np.asarray(fx.roots(fy, method="ms")))
    ax.plot(r[:, 0], r[:, 1], 'k.', ms=10)
    ax.set_aspect("equal")
    _save(fig)
    print(f"case 3 (critical points): {len(r)} extrema")


if __name__ == "__main__":
    run()
