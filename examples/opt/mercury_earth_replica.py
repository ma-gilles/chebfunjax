"""Closest approach of Mercury and Earth.

Faithful replica of opt/MercuryEarth.m by Tonatiuh Sanchez-Vizuet
(June 2014): the minimum distance between Mercury and Earth over
1000 days, from a chebfun of the inter-planet distance.

Original: https://www.chebfun.org/examples/opt/MercuryEarth.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'opt')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    def xm(t):
        return -11.9084 + 57.9117 * jnp.cos(2 * jnp.pi * t / 87.97)

    def ym(t):
        return 56.6741 * jnp.sin(2 * jnp.pi * t / 87.97)

    def xe(t):
        return -2.4987 + 149.6041 * jnp.cos(2 * jnp.pi * t / 365.25)

    def ye(t):
        return 149.5832 * jnp.sin(2 * jnp.pi * t / 365.25)

    f = cj.chebfun(
        lambda t: jnp.sqrt((ym(t) - ye(t))**2 + (xm(t) - xe(t))**2),
        domain=(0.0, 1000.0))
    mintime, minval = f.min()
    mintime, minval = float(mintime), float(minval)
    print(f"min distance = {minval:.6f} at t = {mintime:.6f} days")

    ts = np.linspace(0, 1000, 2500)
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.plot(ts, np.asarray(f(ts)), lw=1.2)
    ax.set_xlabel("Time (days)")
    ax.plot(mintime, minval, '.r', ms=14)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "MercuryEarth_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    tt = jnp.asarray(np.linspace(0, 365.25, 800))
    ax.plot(np.asarray(xm(tt)), np.asarray(ym(tt)), lw=1.2)
    ax.plot(np.asarray(xe(tt)), np.asarray(ye(tt)), lw=1.2)
    tm = jnp.asarray(mintime)
    ax.plot(float(xm(tm)), float(ym(tm)), '.r', ms=14)
    ax.plot(float(xe(tm)), float(ye(tm)), '.r', ms=14)
    ax.plot([float(xm(tm)), float(xe(tm))],
            [float(ym(tm)), float(ye(tm))], 'k--', lw=0.8)
    ax.set_title("Mercury and Earth Orbits", fontsize=12)
    ax.set_aspect("equal")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "MercuryEarth_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
