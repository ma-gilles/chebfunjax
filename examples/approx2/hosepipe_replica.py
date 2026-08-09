"""Combining Chebyshev and trigonometric.

Faithful replica of approx2/Hosepipe.m (Trefethen, 2019): mixed
Chebyshev/trig chebfun2 representations via the 'trigy' flag -- a
corrugated hosepipe surface (nonperiodic in x, periodic in phi), the
display of the three coordinate chebfun2 objects, mixed plotcoeffs,
and a function on an annulus.

Original: https://www.chebfun.org/examples/approx2/Hosepipe.html
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


def _save(fig, k):
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Hosepipe_repl_{k:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def _display(name, F):
    xa, xb, ya, yb = F.domain
    cv = [float(F(np.array([x]), np.array([y]))[0])
          for (x, y) in [(xa, ya), (xb, ya), (xa, yb), (xb, yb)]]
    g = np.linspace(-1, 1, 101)
    X, Y = np.meshgrid(g, g)
    vs = float(np.max(np.abs(np.asarray(F(X, Y)))))
    print(f"{name} =")
    print("   chebfun2 object  (trig in y)")
    print("       domain                 rank       corner values")
    print(f"[{xa:4.0f},{xb:4.0f}] x [{ya:4.0f},{yb:4.0f}]"
          f"     {int(F.rank):4d}     "
          f"[{cv[0]:.2g} {cv[1]:.2g} {cv[2]:.2g} {cv[3]:.2g}]")
    print(f"vertical scale = {vs:.2g}")


def _plotcoeffs(F, k):
    """Mixed plotcoeffs: Chebyshev row coeffs and Fourier col coeffs."""
    rows = F.approx.rows
    cols = F.approx.cols
    rc = np.zeros(max(int(r.coeffs.shape[0]) for r in rows))
    for r in rows:
        a = np.abs(np.asarray(r.coeffs)).ravel()
        rc[:a.shape[0]] = np.maximum(rc[:a.shape[0]], a)
    nmax = max(int(c.coeffs.shape[0]) for c in cols)
    cc = np.zeros(nmax)
    ks = None
    for c in cols:
        a = np.abs(np.asarray(c.coeffs)).ravel()
        n = a.shape[0]
        kk = np.arange(-(n // 2), n - n // 2)
        if ks is None or n == nmax:
            ks = np.arange(-(nmax // 2), nmax - nmax // 2)
        pad = np.zeros(nmax)
        off = (nmax // 2) - (n // 2)
        pad[off:off + n] = a
        cc = np.maximum(cc, pad)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax1.semilogy(np.arange(rc.shape[0]), np.maximum(rc, 1e-20), '.')
    ax1.set_title("Chebyshev coefficients (rows, x)")
    ax1.set_xlabel("degree")
    ax1.grid(True)
    ax2.semilogy(ks, np.maximum(cc, 1e-20), '.')
    ax2.set_title("Fourier coefficients (columns, y)")
    ax2.set_xlabel("wavenumber")
    ax2.grid(True)
    _save(fig, k)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    r = chebfun(lambda x: .5 + .04 * np.cos(40 * x))

    def rx(x):
        return .5 + .04 * np.cos(40 * x)

    F = Chebfun2.from_function(lambda x, ph: 2 * x, trigy=True)
    G = Chebfun2.from_function(
        lambda x, ph: rx(x) * np.cos(np.pi * ph), trigy=True)
    H = Chebfun2.from_function(
        lambda x, ph: rx(x) * np.sin(np.pi * ph), trigy=True)

    # Hosepipe surface.
    xg = np.linspace(-1, 1, 400)
    pg = np.linspace(-1, 1, 200)
    X, P = np.meshgrid(xg, pg)
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(np.asarray(F(X, P)), np.asarray(G(X, P)),
                    np.asarray(H(X, P)), cmap="viridis",
                    rstride=1, cstride=2, linewidth=0)
    ax.set_box_aspect((2, 1, 1))
    ax.axis("off")
    _save(fig, 1)

    _display("F", F)
    _display("G", G)
    _display("H", H)

    _plotcoeffs(G, 2)

    # Annulus: f analytic in 1/2 <= |z| <= 3/2.
    def f(z):
        return (1 + 4 / z**3)**-1 * (z**3 + .1)**-1

    def Fa(rr, tt):
        return np.abs(f(rr * np.exp(1j * tt)))

    Fc = Chebfun2.from_function(Fa, domain=(.5, 1.5, -np.pi, np.pi),
                                trigy=True)
    print("Fc rank:", int(Fc.rank), "length:", Fc.length())

    rg = np.linspace(.5, 1.5, 200)
    tg = np.linspace(-np.pi, np.pi, 400)
    R, T = np.meshgrid(rg, tg)
    Z = np.asarray(Fc(R, T))
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    pc = ax.pcolormesh(R, T, Z, cmap="viridis", shading="auto")
    fig.colorbar(pc, ax=ax)
    ax.set_xlabel("r")
    ax.set_ylabel("t")
    _save(fig, 3)

    _plotcoeffs(Fc, 4)


if __name__ == "__main__":
    run()
