"""The tiger's tail.

Faithful replica of roots/Tiger.m by Nick Trefethen (November 2013):
the equation f(x) = round(f(x)) solved by chebfun rootfinding, giving
the stripes of a tiger's tail.

Original: https://www.chebfun.org/examples/roots/Tiger.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')

ORANGE = (1, 0.5, 0.25)
FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Tiger_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    f = cj.chebfun(
        lambda x: 2 * jnp.exp(0.5 * x)
        * (jnp.sin(5 * x) + jnp.sin(101 * x)),
        domain=(-2.0, 1.0))
    roundf = f.round()
    r = np.asarray((f - roundf).roots(nojump=True))

    xs = np.linspace(-2, 1, 4000)
    fv = np.asarray(f(xs))
    rndv = np.asarray(roundf(xs))
    fr = np.asarray(f(r))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(xs, fv, color=ORANGE, lw=1.0)
    ax.set_ylim(-8, 6)
    ax.plot(r, fr, '.k', ms=5)
    _save(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(xs, fv, color=ORANGE, lw=1.0)
    ax.set_ylim(-8, 6)
    ax.plot(xs, rndv, 'k', lw=1.0)
    _save(fig)

    print("number_of_roots =")
    print(f"   {len(r)}")

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(xs, fv, color=ORANGE, lw=1.0)
    ax.plot(xs, rndv, 'k', lw=1.0)
    ax.plot(r, fr, '.k', ms=5)
    ax.set_ylim(-8, 6)
    _save(fig)


if __name__ == "__main__":
    run()
