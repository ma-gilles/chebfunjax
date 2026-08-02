"""Bivariate rootfinding for a fluid mechanics problem.

Faithful replica of roots/Subramanian.m by Nick Trefethen
(December 2015): common zeros of two cubic bivariate polynomials
arising in a fluid mechanics stability problem of Subramanian et al.

Original: https://www.chebfun.org/examples/roots/Subramanian.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')

DOM = (-0.3, 0.3, -0.15, 0.15)
Q = 1
NU = 0.1
FIG = [0]


def _plot_cf(ax, cf, color):
    bps = list(cf.domain.breakpoints)
    for a, b in zip(bps[:-1], bps[1:]):
        t = np.linspace(a, b, 200)
        v = np.asarray(cf(t))
        ax.plot(v.real, v.imag, color=color, lw=1.2)


def _case(mu, q):
    FIG[0] += 1
    p = cj.chebfun2(
        lambda z, w: mu * z + 2 * Q * w**2 + 4 * Q * w * z
        - 6 * w**3 - 42 * w**2 * z - 18 * w * z**2 - 27 * z**3,
        domain=DOM)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for c in p.roots():
        _plot_cf(ax, c, 'b')
    for c in q.roots():
        _plot_cf(ax, c, 'r')
    ax.grid(True)
    r = np.atleast_2d(np.asarray(p.roots(q)))
    r = r[np.lexsort((r[:, 1], r[:, 0]))]
    print("r =")
    for row in r:
        print(f"  {row[0]:>18.15f}  {row[1]:>18.15f}")
    ax.plot(r[:, 0], r[:, 1], '.k', ms=9)
    ax.set_aspect("equal")
    ax.set_xlabel("z")
    ax.set_ylabel("w")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Subramanian_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    t0 = time.time()

    q = cj.chebfun2(
        lambda z, w: NU * w + 4 * Q * w * z + 2 * Q * z**2
        - 27 * w**3 - 18 * w**2 * z - 42 * w * z**2 - 6 * z**3,
        domain=DOM)

    _case(0.1, q)
    _case(-0.1, q)

    print("Time_for_this_example =")
    print(f"   {time.time() - t0:.15f}")


if __name__ == "__main__":
    run()
