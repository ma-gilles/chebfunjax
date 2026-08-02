"""The doublelength flag.

Faithful replica of cheb/DoublelengthFlag.m by Nick Trefethen (July
2019): constructing chebfuns at twice the normally selected length to
inspect the rounding plateau.

Original: https://www.chebfun.org/examples/cheb/DoublelengthFlag.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'cheb')


def _pair_plot(f, f2, fname, dots=True, trig=False):
    fig, ax = plt.subplots(figsize=(8.8, 4.4))

    def coeffs_idx(g):
        c = np.abs(np.asarray(g.coeffs)) + 1e-30
        if trig:
            n = len(c)
            k = np.arange(n) - n // 2
            return np.abs(k), c
        return np.arange(len(c)), c

    k2, c2 = coeffs_idx(f2)
    k1, c1 = coeffs_idx(f)
    if dots:
        ax.semilogy(k2, c2, '.', ms=7)
        ax.semilogy(k1, c1, 'or', ms=5, mfc='none')
    else:
        ax.semilogy(k2, c2, lw=1)
        ax.semilogy(k1, c1, 'r', lw=1)
    ax.grid(True)
    ax.set_xlabel("degree" if not trig else "wave number")
    ax.set_ylabel("magnitude of coefficient")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # doublelength emulated by fixed-length construction at 2*len
    f = cj.chebfun(lambda x: jnp.exp(x))
    f2 = cj.chebfun(lambda x: jnp.exp(x), n=2 * len(f))
    _pair_plot(f, f2, "DoublelengthFlag_repl_01.png")
    print(f"exp: len {len(f)} -> {len(f2)}")

    g = cj.chebfun(lambda x: jnp.sin(x) + jnp.sin(x**2),
                   domain=(0.0, 10.0))
    g2 = cj.chebfun(lambda x: jnp.sin(x) + jnp.sin(x**2),
                    domain=(0.0, 10.0), n=2 * len(g))
    _pair_plot(g, g2, "DoublelengthFlag_repl_02.png", dots=False)
    print(f"wiggly: len {len(g)} -> {len(g2)}")

    ff = lambda t: 1.0 / (2 - jnp.cos(17 * (t - 1)))  # noqa: E731
    h = cj.chebfun(ff, domain=(-np.pi, np.pi), trig=True)
    h2 = cj.chebfun(ff, domain=(-np.pi, np.pi), trig=True,
                    n=2 * len(h) + 1)
    _pair_plot(h, h2, "DoublelengthFlag_repl_03.png", trig=True)
    print(f"trig: len {len(h)} -> {len(h2)}")


if __name__ == "__main__":
    run()
