"""Phase portraits of singularities.

Faithful replica of complex/Singularities.m by Nick Trefethen (May
2017): phase portraits of a removable singularity, poles of orders
1-4, and an essential singularity, via chebfun2 representations of
'smashed' complex functions.

Original: https://www.chebfun.org/examples/complex/Singularities.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'complex')


def _phase_plot(F, dom, fname, n=500):
    xa, xb, ya, yb = dom
    xs = np.linspace(xa, xb, n)
    ys = np.linspace(ya, yb, n)
    X, Y = np.meshgrid(xs, ys)
    with np.errstate(all="ignore"):
        V = np.asarray(F(X + 1j * Y))
    H = (np.angle(V) + np.pi) / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.imshow(plt.cm.hsv(H), origin="lower",
              extent=(xa, xb, ya, yb), aspect="equal")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def smash(v):
    with np.errstate(all="ignore"):
        g = v / (1 + np.abs(v) ** 2)
    return np.where(np.isnan(g), 0.0, g)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Removable singularity: sin(z)/z (representable as a smooth
    # chebfun2 of the two real variables)
    f = lambda z: np.sin(z + 1e-14) / (z + 1e-14)  # noqa: E731
    rm = cj.chebfun2(lambda x, y: jnp.asarray(
        f(np.asarray(x) + 1j * np.asarray(y))),
        domain=tuple(1.5 * np.pi * np.array([-1, 1, -1, 1])))
    print("removable chebfun2 rank:", rm.rank)
    _phase_plot(f, 1.5 * np.pi * np.array([-1, 1, -1, 1]),
                "Singularities_repl_01.png")

    # Poles of orders 1, 2, 3, 4 at 1, i, -1, -i (smashed)
    g = lambda z: ((z - 1) ** -1 * (z - 1j) ** -2  # noqa: E731
                   * (z + 1) ** -3 * (z + 1j) ** -4)
    _phase_plot(lambda z: smash(g(z)), (-2, 2, -2, 2),
                "Singularities_repl_02.png")

    # Essential singularity exp(-1/z^0.9) at the origin
    h1 = lambda z: np.exp(-1.0 / (z ** 0.9))  # noqa: E731
    _phase_plot(h1, (0, 0.5, -0.25, 0.25),
                "Singularities_repl_03.png")


if __name__ == "__main__":
    run()
