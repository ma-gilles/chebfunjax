"""Chebfun gallery functions.

Faithful replica of approx/Galleries.m by Hrothgar (November 2014):
a tour of cheb.gallery and cheb.gallerytrig showpieces.

Original: https://www.chebfun.org/examples/approx/Galleries.html
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

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.gallery import gallery
from chebfunjax.utils.gallerytrig import gallerytrig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def _vals(f, n=4000):
    a, b = float(f.domain.a), float(f.domain.b)
    xs = np.linspace(a, b, n)
    return xs, np.asarray(f(jnp.asarray(xs)))


def run():
    os.makedirs(_IMG, exist_ok=True)

    # The rose
    f = gallery("rose")
    _, v = _vals(f, 6000)
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.fill(np.real(v), np.imag(v), color=(1, 0, 0))
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Galleries_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Airy function
    xs, v = _vals(gallery("airy"))
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.plot(xs, v, lw=1.2)
    ax.set_ylim(-0.6, 0.6)
    ax.set_title("Airy function", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Galleries_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Zigzag: polynomial of degree 10,000
    xs, v = _vals(gallery("zigzag"), 8000)
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.plot(xs, v, 'm', lw=1.0)
    ax.set_ylim(-0.13, 0.09)
    ax.set_title("polynomial of degree 10,000", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Galleries_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # The motto
    _, v = _vals(gallery("motto"), 8000)
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    ax.plot(np.real(v), np.imag(v), 'k', lw=1.6)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Galleries_repl_04.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Weierstrass (trig)
    xs, v = _vals(gallerytrig("weierstrass"), 8000)
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.plot(xs, v, lw=1.0, color=(0, 0.6, 0))
    ax.set_title("Weierstrass function", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Galleries_repl_05.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Tsunami (trig)
    xs, v = _vals(gallerytrig("tsunami"), 8000)
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.plot(xs, np.real(v), color=(0.8, 0.5, 0), lw=1.0)
    ax.set_ylim(-0.2, 0.2)
    ax.set_title("tsunami", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Galleries_repl_06.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("galleries plotted: rose, airy, zigzag, motto, weierstrass, "
          "tsunami")


if __name__ == "__main__":
    run()
