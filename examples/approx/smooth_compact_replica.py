"""A smooth function with compact support.

Faithful replica of approx/SmoothCompact.m by Nick Trefethen (May
2012): iterated convolution of box functions (an up-function relative)
gives a C^2 bump of compact support; shifted copies sum to a partition
of unity.

Original: https://www.chebfun.org/examples/approx/SmoothCompact.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def _plot_pieces(ax, g, color, lw=1.6, pad=0.0):
    brk = [float(b) for b in g.domain.breakpoints]
    for lo, hi in zip(brk[:-1], brk[1:]):
        xs = np.linspace(lo + 1e-12, hi - 1e-12, 400)
        ax.plot(xs, np.asarray(g(jnp.asarray(xs))), color, lw=lw)


def run():
    os.makedirs(_IMG, exist_ok=True)

    p = lambda h: cj.chebfun(lambda x: (1.0 / h) + 0 * x,  # noqa: E731
                             domain=(-h / 2, h / 2))
    f = p(1.0)
    for k in range(3, 6):
        f = f.conv(p(2.0**-k))

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    _plot_pieces(ax, f, 'b')
    ax.axis([-1, 1, -0.2, 1.2])
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "SmoothCompact_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("ans =")
    print(f"     {float(f.sum()):.0f}")

    a = float(f.domain.a)
    b = float(f.domain.b)

    def embed(g, lo, hi):
        # chebfun({0, g, 0}, [-1 lo hi 2]) — g on [lo,hi], zero outside
        def ev(x):
            inside = (x >= lo) & (x <= hi)
            xc = jnp.clip(x, lo, hi)
            return jnp.where(inside, g(xc), 0.0)
        return ev

    f1 = embed(f, a, b)
    fsh = f.new_domain((a + 1.0, b + 1.0))
    f2 = embed(fsh, a + 1.0, b + 1.0)

    xs = np.linspace(-1, 2, 3000)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(xs, np.asarray(f1(jnp.asarray(xs))), 'b', lw=1.6)
    ax.plot(xs, np.asarray(f2(jnp.asarray(xs))), 'g', lw=1.6)
    ax.axis([-1, 2, -0.2, 1.2])
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "SmoothCompact_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    gv = (np.asarray(f1(jnp.asarray(xs)))
          + np.asarray(f2(jnp.asarray(xs))))
    ax.plot(xs, gv, 'm', lw=1.6)
    ax.axis([-1, 2, -0.2, 1.2])
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "SmoothCompact_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
