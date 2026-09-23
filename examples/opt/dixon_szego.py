"""The six-hump camel function of Dixon and Szego.

Translation of opt/DixonSzego.m by Nick Trefethen
(September 2010, revised 2016): global minimization of the six-hump
camel function, first by nested 1D chebfun minimization, then in one
step with chebfun2 min2.

Original: https://www.chebfun.org/examples/opt/DixonSzego.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import PARULA, chebfun_style, contour
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'opt')


def f(x, y):
    return ((4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y
            + 4 * (y**2 - 1) * y**2)


def _save(fig, k):
    fig.set_facecolor("white")
    fig.set_size_inches(6.0, 2.7)
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"DixonSzego_{k:02d}.png"))


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1.25, 1.25, 100)
    xx, yy = np.meshgrid(x, y)
    ff = f(xx, yy)
    fig, ax = plt.subplots()
    cs = ax.contour(x, y, ff, 30, linewidths=1.2, cmap=PARULA)
    fig.colorbar(cs, ax=ax)
    ax.axis([-2, 2, -1.25, 1.25])
    _save(fig, 1)

    t0 = time.time()

    def fminx0(x0):
        return cj.chebfun(lambda t: f(x0, t), domain=(-1.25, 1.25)).min()[1]

    fminx = cj.chebfun(
        lambda t: jnp.asarray([fminx0(float(v)) for v in np.atleast_1d(t)]),
        domain=(-2.0, 2.0), splitting=True)
    minx, minf = fminx.min()
    print("minf =")
    print(f"{minf:20.15f}")
    print("minx =")
    print(f"{minx:20.15f}")
    miny, minf = cj.chebfun(lambda t: f(minx, t), domain=(-1.0, 3.0)).min()
    print("minf =")
    print(f"{minf:20.15f}")
    print("miny =")
    print(f"{miny:20.15f}")
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")
    ax.plot(minx, miny, '.k', ms=14)                     # hold on
    _save(fig, 2)
    plt.close(fig)

    t0 = time.time()
    F = cj.chebfun2(f, domain=(-2, 2, -1.25, 1.25))
    minf, minx = F.min2()
    print("minf =")
    print(f"{float(minf):20.15f}")
    print("minx =")
    print(f"{float(minx[0]):20.15f}{float(minx[1]):20.15f}")
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")

    fig, ax = plt.subplots()
    contour(F, ax=ax, levels=30)
    fig.colorbar(ax.collections[-1], ax=ax)
    ax.set_aspect("auto")
    ax.axis([-2, 2, -1.25, 1.25])
    ax.plot(float(minx[0]), float(minx[1]), '.k', ms=14)
    _save(fig, 3)
    plt.close(fig)


if __name__ == "__main__":
    run()
