"""Hyperfunctions.

Faithful replica of complex/Hyperfuns.m by Nick Trefethen (June 2013):
the delta and Heaviside functions realized as limits of differences of
analytic functions evaluated above and below the real axis.

Original: https://www.chebfun.org/examples/complex/Hyperfuns.html
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

F = lambda z: -1.0 / (2j * np.pi * z)                    # noqa: E731
G = lambda z: -1.0 / (2j * np.pi) * np.log(-z + 0j)      # noqa: E731

XS = np.linspace(-1, 1, 4000)


def run():
    os.makedirs(_IMG, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for ep in np.arange(0.1, 0.0005, -0.01):
        v = np.real(F(XS + 1j * ep) - F(XS - 1j * ep))
        ax.plot(XS, v, color=(0, 0.7, 0), lw=1.0)
    ax.set_title("Delta function", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Hyperfuns_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for ep in np.arange(0.1, 0.0005, -0.01):
        v = np.real(G(XS + 1j * ep) - G(XS - 1j * ep))
        ax.plot(XS, v, color=(0.2, 0, 0.7), lw=1.0)
    ax.set_title("Heaviside function", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Hyperfuns_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # sanity: the delta hyperfun integrates to 1 for each epsilon
    v = cj.chebfun(lambda x: jnp.real(
        -1.0 / (2j * jnp.pi * (x + 0.01j))
        + 1.0 / (2j * jnp.pi * (x - 0.01j))))
    print(f"integral of hyperfun delta (ep=0.01): {float(v.sum()):.6f}")


if __name__ == "__main__":
    run()
