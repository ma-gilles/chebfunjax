"""Areas and centroids of planar regions.

Translation of geom/Area.m by Stefan Guettel (October 2011):
the area enclosed by parametrized curves via Green's theorem, and
the centroid of a region, all as chebfun integrals.

Original: https://www.chebfun.org/examples/geom/Area.html
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

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'geom')

FIG = [0]


def _save(fig, close=True):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"Area_{FIG[0]:02d}.png"))
    if close:
        plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    dom = (0.0, 2 * np.pi)
    b, m = 1, 7
    a = (m - 1) * b
    t = cj.chebfun('t', domain=dom)
    x = (a + b) * t.cos() - b * ((a + b) / b * t).cos()
    y = (a + b) * t.sin() - b * ((a + b) / b * t).sin()
    ts = np.linspace(*dom, 1200)
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.fill(np.asarray(x(ts)), np.asarray(y(ts)),
            facecolor=(0.6, 0.6, 1), edgecolor="k")
    ax.set_aspect("equal", adjustable="datalim")
    _save(fig)

    print("x =")
    print(repr(x))
    print("y =")
    print(repr(y))
    A = float((x * y.diff()).sum())
    print("A =")
    print(f"     {A:.15e}")
    print("exact =")
    print(f"     {np.pi * b**2 * (m**2 + m):.15e}")

    z = (1j * t).exp() + (1 + 1j) * (6 * t).sin()**2
    zr = z.real()
    zi = z.imag()
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    zv = np.asarray(z(ts))
    ax.fill(zv.real, zv.imag, facecolor=(0.6, 1, 0.6), edgecolor="k")
    ax.set_aspect("equal", adjustable="datalim")
    _save(fig, close=False)
    A2 = float((zr * zi.diff()).sum())
    print("ans =")
    print(f"   {A2:.15f}")
    print(f"   {np.pi:.15f}")

    c = complex(np.asarray((z.diff() * z * z.conj()).sum())) \
        / (2j * A2)
    ax.plot(c.real, c.imag, 'r+', ms=16, mew=2)
    _save(fig)


if __name__ == "__main__":
    run()
