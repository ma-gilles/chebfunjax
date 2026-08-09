"""Padua points in Chebfun2.

Faithful replica of approx2/PaduaPoints.m (Hale & Townsend, 2014):
the Padua grid for n = 8, its characterization via a Lissajous curve
and via every other point of an (n+1)x(n+2) Chebyshev tensor grid,
interpolation from Padua samples ('padua' flag), and the spy plot of
the total-degree-n bivariate Chebyshev coefficients.

Original: https://www.chebfun.org/examples/approx2/PaduaPoints.html
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

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.padua import paduapts
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.quadrature import chebpts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2')


def _save(fig, k):
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"PaduaPoints_repl_{k:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    n = 8
    x = np.asarray(paduapts(n))

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.plot(x[:, 0], x[:, 1], 'ok', markerfacecolor='k', ms=6)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("Padua points", fontsize=14)
    _save(fig, 1)

    # Lissajous curve L(t) = -cos((n+1)t) - 1i cos(nt), t in [0, pi].
    t = np.linspace(0, np.pi, 2000)
    L = -np.cos((n + 1) * t) - 1j * np.cos(n * t)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.plot(x[:, 0], x[:, 1], 'ok', markerfacecolor='k', ms=6)
    ax.plot(np.real(L), np.imag(L), 'b', lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("Padua points", fontsize=14)
    _save(fig, 2)

    # Every other point of an (n+1) x (n+2) Chebyshev tensor grid.
    x1 = np.asarray(chebpts(n + 1))
    x2 = np.asarray(chebpts(n + 2))
    X, Y = np.meshgrid(x1, x2)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.plot(x[:, 0], x[:, 1], 'ok', markerfacecolor='k', ms=6)
    ax.plot(np.real(L), np.imag(L), 'b', lw=1.2)
    ax.plot(X.ravel(), Y.ravel(), 'or', markerfacecolor='none', ms=6)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("Padua points", fontsize=14)
    _save(fig, 3)

    # Chebfun2 from samples on the Padua grid.
    def f(xx, yy):
        return np.cos(np.exp(2 * xx + yy)) * np.sin(yy)

    fx = f(x[:, 0], x[:, 1])
    F = Chebfun2.from_padua(fx)
    g = np.linspace(-1, 1, 240)
    GX, GY = np.meshgrid(g, g)
    Z = np.asarray(F(GX, GY))
    fig, ax = plt.subplots(figsize=(7.2, 5.6),
                           subplot_kw={"projection": "3d"})
    ax.plot_surface(GX, GY, Z, cmap="viridis", rstride=1, cstride=1,
                    linewidth=0)
    Fx = np.asarray(F(x[:, 0], x[:, 1]))
    ax.plot(x[:, 0], x[:, 1], Fx, 'ok', markerfacecolor='k', ms=5)
    ax.view_init(30, -37.5)
    _save(fig, 4)

    err = float(np.max(np.abs(Fx - fx)))
    print(f"max interpolation error on the Padua grid: {err:.2e}")

    # Spy plot of the bivariate Chebyshev coefficients (total degree n).
    C = np.array(F.chebcoeffs2())
    C[np.abs(C) < 1e-10] = 0
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ii, jj = np.nonzero(C)
    ax.plot(jj, ii, 'sb', ms=8)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("Spy plot of bivariate Chebyshev coefficients",
                 fontsize=14)
    _save(fig, 5)
    print(f"coefficient matrix shape: {C.shape}; "
          f"nonzeros on/below anti-diagonal only: "
          f"{bool(np.all(ii + jj <= n))}")


if __name__ == "__main__":
    run()
