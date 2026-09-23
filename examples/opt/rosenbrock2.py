"""Rosenbrock revisited with chebfun2.

Translation of opt/Rosenbrock2.m by Alex Townsend (March 2013):
2D global minimization in a single chebfun2 call, plus all critical
points from the roots of the gradient.

Original: https://www.chebfun.org/examples/opt/Rosenbrock2.html
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
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'opt')

FIG = [0]


def _save(fig, close=True):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"Rosenbrock2_{FIG[0]:02d}.png"))
    if close:
        plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    F = cj.chebfun2(lambda x, y: (1 - x)**2 + 100 * (y - x**2)**2,
                    domain=(-1.5, 1.5, -1, 3))
    v, loc = F.min2()
    print("minf =")
    print(f"     {float(v):.15e}")
    print("minx =")
    print(f"   {float(loc[0]):.15f}   {float(loc[1]):.15f}")

    x = np.linspace(-1.5, 1.5, 150)
    y = np.linspace(-1, 3, 150)
    X, Y = np.meshgrid(x, y)
    Z = np.asarray(F(jnp.asarray(X), jnp.asarray(Y)))
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    cs = ax.contour(X, Y, Z, levels=np.arange(10, 301, 10))
    fig.colorbar(cs, ax=ax)
    ax.plot(float(loc[0]), float(loc[1]), '.k', ms=14)
    _save(fig)

    t0 = time.time()
    G = cj.chebfun2(lambda x, y: jnp.exp(x - 2 * x**2 - y**2)
                    * jnp.sin(6 * (x + y + x * y**2)))
    v2, loc2 = G.min2()
    print("minf =")
    print(f"  {float(v2):.15f}")
    print("minx =")
    print(f"   {float(loc2[0]):.15f}   {float(loc2[1]):.15f}")
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")

    Gx = G.diff(dim=2)
    Gy = G.diff(dim=1)
    tp = np.atleast_2d(np.asarray(Gx.roots(Gy, method="ms")))
    # tp = roots(grad(F)): the eight stationary points of the chebfun.org
    # page (Chebfun 7574c77's resultant rootfinder reports only the first
    # five; all eight have |grad F| ~ 1e-15).
    print("tp =")
    for px, py in tp:
        print(f"  {px: .15f}  {py: .15f}")

    xs = np.linspace(-1, 1, 150)
    X2, Y2 = np.meshgrid(xs, xs)
    Z2 = np.asarray(G(jnp.asarray(X2), jnp.asarray(Y2)))
    # contour(F,30), colorbar, hold on, plot(minx(1), minx(2), '.k')
    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    cs = ax.contour(X2, Y2, Z2, 30)
    fig.colorbar(cs, ax=ax)
    ax.plot(float(loc2[0]), float(loc2[1]), '.k', ms=14)
    ax.set_aspect("equal")
    _save(fig, close=False)
    # plot(tp(:,1), tp(:,2), 'ko', MS, 12, LW, 2)
    ax.plot(tp[:, 0], tp[:, 1], 'ko', ms=7, mfc='none', mew=2)
    _save(fig, close=False)
    # g = roots(F); plot(g, 'k', LW, 2)
    tt = np.linspace(-1, 1, 400)
    for c in G.roots():
        z = np.asarray(c(jnp.asarray(tt)))
        ax.plot(np.real(z), np.imag(z), 'k', lw=2)
    _save(fig, close=False)
    # g = roots(F - .5); plot(g, 'r', LW, 2)
    for c in (G - 0.5).roots():
        z = np.asarray(c(jnp.asarray(tt)))
        ax.plot(np.real(z), np.imag(z), 'r', lw=2)
    _save(fig)


if __name__ == "__main__":
    run()
