"""Mercury-Earth conjunctions via determinants.

Faithful replica of linalg/MercuryEarthConjunctions.m by Nikhil
Chaudhary (June 2014): conjunctions of Mercury and Earth occur when
the determinant of the matrix of their position vectors vanishes —
found as the roots of a chebfun of the determinant over 600 days.

Original: https://www.chebfun.org/examples/linalg/MercuryEarthConjunctions.html
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

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')


def _det(t):
    t = jnp.asarray(t)
    a11 = -11.9084 + 57.9117 * jnp.cos(2 * jnp.pi * t / 87.97)
    a12 = 56.6741 * jnp.sin(2 * jnp.pi * t / 87.97)
    a21 = -2.4987 + 149.6041 * jnp.cos(2 * jnp.pi * t / 365.25)
    a22 = 149.5832 * jnp.sin(2 * jnp.pi * t / 365.25)
    return a11 * a22 - a12 * a21


def run():
    os.makedirs(_IMG, exist_ok=True)

    f = cj.chebfun(_det, domain=(0.0, 600.0))
    z = np.asarray(f.roots())
    print("first conjunction times (days):")
    for v in z[:10]:
        print(f"   {v:.6f}")

    xs = np.linspace(0, 600, 2000)
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.plot(xs, np.asarray(f(xs)), lw=1.6)
    ax.grid(True)
    ax.plot(z[:10], np.zeros(min(10, len(z))), '.r', ms=12)
    ax.set_xlabel("Time (days)")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, "MercuryEarthConjunctions_repl_01.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)

    # orbits with the first few conjunction sight-lines
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    tt = np.linspace(0, 365.25, 500)
    mx = -11.9084 + 57.9117 * np.cos(2 * np.pi * tt / 87.97)
    my = 56.6741 * np.sin(2 * np.pi * tt / 87.97)
    ex = -2.4987 + 149.6041 * np.cos(2 * np.pi * tt / 365.25)
    ey = 149.5832 * np.sin(2 * np.pi * tt / 365.25)
    ax.plot(mx, my, lw=1.2, label="Mercury")
    ax.plot(ex, ey, lw=1.2, label="Earth")
    for v in z[:6]:
        pmx = -11.9084 + 57.9117 * np.cos(2 * np.pi * v / 87.97)
        pmy = 56.6741 * np.sin(2 * np.pi * v / 87.97)
        pex = -2.4987 + 149.6041 * np.cos(2 * np.pi * v / 365.25)
        pey = 149.5832 * np.sin(2 * np.pi * v / 365.25)
        ax.plot([0, pex], [0, pey], 'k--', lw=0.7)
        ax.plot([pmx, pex], [pmy, pey], 'r.', ms=8)
    ax.plot(0, 0, 'o', color="orange", ms=10)
    ax.set_aspect("equal")
    ax.legend()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, "MercuryEarthConjunctions_repl_02.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
