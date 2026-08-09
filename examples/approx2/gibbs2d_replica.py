"""Gibbs phenomenon in 2D.

Faithful replica of approx2/Gibbs2D.m: interpolating a 100x100
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

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2')
FIG = [0]


def _surf(fn, zlim=(-.2, 1.5), n=240):
    FIG[0] += 1
    g = np.linspace(-1, 1, n)
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
    fig.savefig(os.path.join(_IMG, f"Gibbs2D_repl_{FIG[0]:02d}.png"),
                dpi=140, bbox_inches="tight")
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
    fig.savefig(os.path.join(_IMG, f"Gibbs2D_repl_{FIG[0]:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def _trig_interp2(A):
    """2D trig interpolant of uniform-grid values on [-1, 1)."""
    n = A.shape[0]
    C = np.fft.fft2(A) / A.size
    k = np.fft.fftfreq(n, d=1.0 / n)

    def ev(x, y):
        Ex = np.exp(1j * np.pi * np.outer(np.asarray(x).ravel() + 1, k))
        Ey = np.exp(1j * np.pi * np.outer(np.asarray(y).ravel() + 1, k))
        return np.real(np.einsum("pk,kl,pl->p", Ey, C, Ex)
                       ).reshape(np.shape(x))
    return ev


def _extreme(fn, kind, n=1600):
    from scipy.optimize import minimize
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)
    Z = fn(X, Y)
    if kind == "max":
        i = np.unravel_index(np.argmax(Z), Z.shape)
    else:
        i = np.unravel_index(np.argmin(Z), Z.shape)
    sgn = -1.0 if kind == "max" else 1.0

    def obj(v):
        z = fn(np.array([[v[0]]]), np.array([[v[1]]]))
        return sgn * float(np.ravel(z)[0])

    res = minimize(obj, [X[i], Y[i]], method="Nelder-Mead",
                   options={"xatol": 1e-13, "fatol": 1e-14,
                            "maxiter": 4000})
    return sgn * float(res.fun)


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
    FIG[0] += 1
    gz = np.linspace(0, .5, 240)
    XZ, YZ = np.meshgrid(gz, gz)
    ZZ = np.asarray(p(XZ, YZ))
    fig, ax = plt.subplots(figsize=(7.2, 5.4),
                           subplot_kw={"projection": "3d"})
    ax.plot_surface(XZ, YZ, ZZ, cmap="viridis", rstride=1, cstride=1,
                    linewidth=0)
    ax.view_init(50, -20)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Gibbs2D_repl_{FIG[0]:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    mn, _ = p.min2()
    print("ans =")
    print(f"  {float(mn):.15f}")

    # Periodic interpolant of the same data.
    t = _trig_interp2(A)
    _surf(t)
    _contour(t)
    print("ans =")
    print(f"   {_extreme(t, 'max'):.15f}")
    print("ans =")
    print(f"  {_extreme(t, 'min'):.15f}")

    # Triangular block: same Gibbs, full matrix rank.
    A2 = np.tril(A)
    p2 = Chebfun2.from_values(A2)
    _surf(lambda X, Y: np.asarray(p2(X, Y)))
    m2b, _ = p2.max2()
    mnb, _ = p2.min2()
    print("ans =")
    print(f"   {float(m2b):.15f}")
    print("ans =")
    print(f"  {float(mnb):.15f}")
    _contour(lambda X, Y: np.asarray(p2(X, Y)))

    # Ranks: block data is rank 1; triangular block is full rank.
    print("ans =")
    print(f"     {p.rank}")
    print("ans =")
    print(f"     {p2.rank}")
    print("ans =")
    print(f"    {np.linalg.matrix_rank(A2)}")


if __name__ == "__main__":
    run()
