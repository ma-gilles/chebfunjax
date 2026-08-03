"""The area between two circles.

Faithful replica of geom/TwoCircles.m by Nick Trefethen
(October 2011): the lens-shaped area between arcs of two circles,
via splitting, roots, and restricted integrals.

Original: https://www.chebfun.org/examples/geom/TwoCircles.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'geom')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    big = cj.chebfun(lambda x: jnp.sqrt(4 - (x - 1)**2),
                     splitting=True)
    little = cj.chebfun(lambda x: 2 - jnp.sqrt(1 - (x + 1)**2),
                        domain=(-1.0, 0.0), splitting=True)

    diffc = big.restrict(-1.0, 0.0) - little
    x = np.asarray(diffc.roots())
    x1, x2 = float(x[0]), float(x[1])
    print("x1 =")
    print(f"  {x1:.15f}")
    print("x2 =")
    print(f"  {x2:.15f}")

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.plot([-1, 1, 1, -1, -1], [0, 0, 2, 2, 0], 'k')
    ts = np.linspace(x1, x2, 300)
    ax.fill(np.concatenate([ts, ts[::-1]]),
            np.concatenate([np.asarray(little(ts)),
                            np.asarray(big(ts[::-1]))]), 'r')
    xs = np.linspace(-1, 1, 500)
    ax.plot(xs, np.asarray(big(xs)), 'k', lw=1.2)
    xs2 = np.linspace(-1, 0, 300)
    ax.plot(xs2, np.asarray(little(xs2)), 'k', lw=1.2)
    ax.axis([-1, 1, 0, 2])
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([0, 1, 2])
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "TwoCircles_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    area = float((big.restrict(x1, x2)
                  - little.restrict(x1, x2)).sum())
    print("area =")
    print(f"   {area:.15f}")
    exact = (np.arccos(5 * np.sqrt(2) / 8)
             + 4 * np.arccos(11 * np.sqrt(2) / 16)
             - np.sqrt(7) / 2)
    print("exact =")
    print(f"   {exact:.15f}")


if __name__ == "__main__":
    run()
