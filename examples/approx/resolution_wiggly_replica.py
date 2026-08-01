"""Resolution of wiggly functions.

Faithful replica of approx/ResolutionWiggly.m by Nick Trefethen (June
2014): interpolation, least-squares, and best approximation of a
wiggly function by polynomials of half the resolved degree all fail
comparably — resolution is what matters.

Original: https://www.chebfun.org/examples/approx/ResolutionWiggly.html
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
from chebfunjax.utils.minimax import minimax

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

XS = np.linspace(0, 14, 4000)


def _plot(curves, title, fname, ylim=(-2.5, 2.5)):
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    for ys, color in curves:
        ax.plot(XS, ys, color, lw=1.2)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    f = cj.chebfun(lambda x: jnp.sin(x)**2 + jnp.sin(x**2),
                   domain=(0.0, 14.0))
    fv = np.asarray(f(jnp.asarray(XS)))
    _plot([(fv, 'b')], "", "ResolutionWiggly_repl_01.png")

    n_p = len(f)
    print("np =")
    print(f"   {n_p}")
    nphalf = round(n_p / 2)
    print("nphalf =")
    print(f"    {nphalf}")

    # Interpolant of half the degree
    pinterp = cj.chebfun(lambda x: f(x), domain=(0.0, 14.0), n=nphalf)
    pv = np.asarray(pinterp(jnp.asarray(XS)))
    _plot([(fv, 'b'), (pv, 'r')],
          "f and interpolant of half the degree",
          "ResolutionWiggly_repl_02.png")
    _plot([(fv - pv, 'k')],
          "error of interpolant of half the degree",
          "ResolutionWiggly_repl_03.png")

    # Least-squares approximant of half the degree
    pleastsq = f.polyfit(nphalf - 1)
    lv = np.asarray(pleastsq(jnp.asarray(XS)))
    _plot([(fv, 'b'), (lv, 'r')],
          "f and least-squares approximant of half the degree",
          "ResolutionWiggly_repl_04.png")
    _plot([(fv - lv, 'k')],
          "error of least-squares approximant of half the degree",
          "ResolutionWiggly_repl_05.png")

    # Best approximant of half the degree
    res = minimax(lambda x: jnp.sin(x)**2 + jnp.sin(x**2), nphalf - 1,
                  domain=(0.0, 14.0), max_iter=100)
    pbest = cj.chebfun(jnp.asarray(res.coeffs), coeffs=True,
                       domain=(0.0, 14.0))
    bv = np.asarray(pbest(jnp.asarray(XS)))
    _plot([(fv, 'b'), (bv, 'r')],
          "f and best approximant of half the degree",
          "ResolutionWiggly_repl_06.png")
    _plot([(fv - bv, 'k')],
          "error of best approximant of half the degree",
          "ResolutionWiggly_repl_07.png")

    print(f"interp err   = {np.max(np.abs(fv - pv)):.4f}")
    print(f"leastsq err  = {np.max(np.abs(fv - lv)):.4f}")
    print(f"best err     = {np.max(np.abs(fv - bv)):.4f}")


if __name__ == "__main__":
    run()
