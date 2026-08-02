"""Complex roots near the real axis.

Faithful replica of roots/RootsNearAxis.m by Nick Trefethen
(October 2011): complex roots of a chebfun in the region where it has
some accuracy, via roots(f, 'complex'), with the Chebfun ellipse from
plotregion.

Original: https://www.chebfun.org/examples/roots/RootsNearAxis.html
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
from chebfunjax.plotting import chebfun_style, plotregion

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"RootsNearAxis_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")


def run():
    os.makedirs(_IMG, exist_ok=True)

    f = cj.chebfun(lambda x: 3 + jnp.sin(x) + jnp.sin(jnp.pi * x),
                   domain=(0.0, 30.0))
    xs = np.linspace(0, 30, 1200)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xs, np.asarray(f(xs)), 'b', lw=1.2)
    _save(fig)
    plt.close(fig)

    r0 = np.asarray(f.roots())
    print("ans =")
    if len(r0) == 0:
        print("  0x1 empty double column vector")

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    plotregion(f, ax=ax, title="")
    ax.grid(True)
    ax.set_xlim(-5, 35)
    ax.set_aspect("equal")
    ax.plot([0, 30], [0, 0], 'k')
    _save(fig)

    r = np.asarray(f.roots(complex_roots=True))
    ax.plot(r.real, r.imag, '.r', ms=9)
    _save(fig)
    plt.close(fig)

    print("number_of_roots =")
    print(f"    {len(r)}")
    print("degree =")
    print(f"    {len(f) - 1}")

    ra = np.asarray(f.roots(all_roots=True))
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.plot(ra.real, ra.imag, 'or', ms=5, mfc='none')
    ax.set_aspect("equal")
    _save(fig)
    plt.close(fig)


if __name__ == "__main__":
    run()
