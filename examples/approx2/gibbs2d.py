"""Gibbs phenomenon in 2D.

Translation of approx2/Gibbs2D.m: interpolating a 100x100
square-block data matrix at Chebyshev (chebfun2(A)) and uniform/
periodic (chebfun2(A,'periodic')) grids exhibits the 2D Gibbs
overshoot; a triangular block shows the same with full matrix rank.

Original: https://www.chebfun.org/examples/approx2/Gibbs2D.html
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

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2')
FIG = [0]


def _surf(fn, zlim=(-.2, 1.5), n=240, dom=(-1, 1)):
    FIG[0] += 1
    g = np.linspace(dom[0], dom[1], n)
    X, Y = np.meshgrid(g, g)
    Z = fn(X, Y)
    fig, ax = plt.subplots(figsize=(7.2, 5.4),
                           subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, cmap="viridis", rstride=1, cstride=1,
                    linewidth=0)
    ax.set_zlim(*zlim)
    ax.view_init(50, -20)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"Gibbs2D_{FIG[0]:02d}.png"))
    plt.close(fig)


def _contour(fn, n=400):
    FIG[0] += 1
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)
    Z = fn(X, Y)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    cs = ax.contour(X, Y, Z, 10)
    ax.set_xlim(-.6, .6)
    ax.set_ylim(-.6, .6)
    ax.set_aspect("equal")
    fig.colorbar(cs.collections[0] if hasattr(cs, 'collections')
                 else cs, ax=ax)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"Gibbs2D_{FIG[0]:02d}.png"))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    A = np.zeros((100, 100))
    A[39:61, 39:61] = 1
    p = Chebfun2.from_values(A)
    _surf(lambda X, Y: np.asarray(p(X, Y)))
    _contour(lambda X, Y: np.asarray(p(X, Y)))

    m2, _ = p.max2()
    print("ans =")
    print(f"   {float(m2):.15f}")

    a = np.zeros(100)
    a[39:61] = 1
    p1 = chebfun(a)
    _, m1 = p1.max()
    print("ans =")
    print(f"   {float(m1):.15f}")

    # Zoom near a corner of the block (MATLAB plot(p{0,.5,0,.5})).
    pzoom = p.restrict((0, .5, 0, .5))
    _surf(lambda X, Y: np.asarray(pzoom(X, Y)), dom=(0, .5))

    mn, _ = p.min2()
    print("ans =")
    print(f"  {float(mn):.15f}")

    # Periodic interpolant of the same data.
    t = Chebfun2.from_values(A, trig=True)
    _surf(lambda X, Y: np.asarray(t(X, Y)))
    _contour(lambda X, Y: np.asarray(t(X, Y)))
    mt, _ = t.max2()
    mnt, _ = t.min2()
    print("ans =")
    print(f"   {float(mt):.15f}")
    print("ans =")
    print(f"  {float(mnt):.15f}")

    # Triangular block: same Gibbs, full matrix rank.
    A2 = np.tril(A)
    p2 = Chebfun2.from_values(A2)
    p2z = p2.restrict((-.5, .5, -.5, .5))
    _surf(lambda X, Y: np.asarray(p2z(X, Y)), dom=(-.5, .5))
    m2b, _ = p2.max2()
    mnb, _ = p2.min2()
    print("ans =")
    print(f"   {float(m2b):.15f}")
    print("ans =")
    print(f"  {float(mnb):.15f}")
    _contour(lambda X, Y: np.asarray(p2(X, Y)))

    # Ranks: block data is rank 1; triangular block is full rank.
    for r in (p.rank, t.rank, p2.rank, np.linalg.matrix_rank(A2)):
        print("ans =")
        print(f"{r:6d}")
    # spy(A2), axis([36 65 36 65])
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    ii, jj = np.nonzero(A2)
    ax.plot(jj + 1, ii + 1, '.', color="C0", ms=5)
    ax.set_xlim(36, 65)
    ax.set_ylim(65, 36)
    ax.set_aspect("equal")
    ax.set_xlabel(f"nz = {len(ii)}")
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"Gibbs2D_{FIG[0]:02d}.png"))
    plt.close(fig)


if __name__ == "__main__":
    run()
