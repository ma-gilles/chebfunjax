"""Roots of a secular equation with poles.

Faithful replica of roots/SecularRoots.m by Nick Trefethen
(November 2010): roots of the secular equation
f(x) = 1 + sum_k 1/(k - x), whose chebfun representation carries
simple poles at x = 1, 2, 3, 4 as SingFun pieces; roots(f) includes
the sign flips through the poles, roots(f, 'nojump') gives only the
genuine roots.

Original: https://www.chebfun.org/examples/roots/SecularRoots.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')


def run():
    os.makedirs(_IMG, exist_ok=True)

    x = cj.chebfun(lambda t: t, domain=(-5.0, 10.0))
    f = 1 + 1 / (1 - x) + 1 / (2 - x) + 1 / (3 - x) + 1 / (4 - x)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bps = [float(v) for v in f.domain.breakpoints]
    for a, b in zip(bps[:-1], bps[1:]):
        w = b - a
        t = np.linspace(a + 1e-6 * w, b - 1e-6 * w, 600)
        v = np.asarray(f(t))
        v = np.where(np.abs(v) > 30, np.nan, v)
        ax.plot(t, v, 'b', lw=2)
    ax.set_ylim(-15, 15)
    ax.grid(True)

    r = np.asarray(f.roots())
    print("r =")
    for v in r:
        print(f"   {v:.15f}")

    rn = np.asarray(f.roots(nojump=True))
    print("r =")
    for v in rn:
        print(f"   {v:.15f}")

    ax.plot(rn, np.asarray(f(rn)), '.r', ms=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "SecularRoots_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
