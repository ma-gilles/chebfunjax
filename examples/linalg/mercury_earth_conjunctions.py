"""Mercury-Earth conjunctions via determinants.

Translation of linalg/MercuryEarthConjunctions.m by Nikhil
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

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')


def M(t):
    return jnp.array([[-11.9084 + 57.9117 * jnp.cos(2 * jnp.pi * t / 87.97),
                       56.6741 * jnp.sin(2 * jnp.pi * t / 87.97)],
                      [-2.4987 + 149.6041 * jnp.cos(2 * jnp.pi * t / 365.25),
                       149.5832 * jnp.sin(2 * jnp.pi * t / 365.25)]])


def run():
    os.makedirs(_IMG, exist_ok=True)

    # chebfun(@(t) det(M(t)), [0 600], 'vectorize')
    f = cj.chebfun(jax.vmap(lambda t: jnp.linalg.det(M(t))),
                   domain=(0.0, 600.0))

    z = np.asarray(f.roots())

    fig, ax = plt.subplots()
    matlab_plot(f, ax=ax, linewidth=1.6)
    ax.grid(True)
    ax.plot(z[:10], np.zeros(min(10, len(z))), '.r', ms=20 * 0.6)
    ax.set_xlabel("Time (days)")
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, "MercuryEarthConjunctions_01.png"))
    plt.close(fig)


if __name__ == "__main__":
    run()
