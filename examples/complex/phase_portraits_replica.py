"""Phase portraits with chebfun2.

Faithful replica of complex/PhasePortraits.m by Alex Townsend (March
2013): phase portraits of analytic functions represented as complex
chebfun2 objects.

Original: https://www.chebfun.org/examples/complex/PhasePortraits.html
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

FIG = [0]


def portrait(f2, dom, title):
    FIG[0] += 1
    xa, xb, ya, yb = dom
    xs = np.linspace(xa, xb, 480)
    ys = np.linspace(ya, yb, 480)
    X, Y = np.meshgrid(xs, ys)
    V = np.asarray(f2(jnp.asarray(X), jnp.asarray(Y)))
    H = (np.angle(V) + np.pi) / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    ax.imshow(plt.cm.hsv(H), origin="lower", extent=(xa, xb, ya, yb),
              aspect="equal")
    ax.set_title(title, fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG,
                             f"PhasePortraits_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    d = np.pi * np.array([-1, 1, -1, 1])

    f = cj.chebfun2(lambda z: jnp.sin(z), domain=tuple(d))
    portrait(f, d, "Phase portrait for sin(z)")

    f = cj.chebfun2(lambda z: jnp.cos(z**2), domain=tuple(d))
    portrait(f, d, "cos(z^2)")

    g = cj.chebfun2(
        lambda z: sum(z**k for k in range(10)), domain=tuple(d / 2))
    portrait(g, d / 2, "Nearly the ten roots of unity")

    f = cj.chebfun2(lambda z: jnp.sin(z) - jnp.sinh(z),
                    domain=tuple(2 * d))
    portrait(f, 2 * d, "Phase portrait plot for sin(z)-sinh(z)")
    print("portraits:", FIG[0])


if __name__ == "__main__":
    run()
