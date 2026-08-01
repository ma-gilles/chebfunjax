"""Symbolic and numerical integration.

Faithful replica of quad/SymbolicNumeric.m by Nick Trefethen: chebfun
integrates functions numerically to machine precision whether or not a
symbolic antiderivative exists.

Original: https://www.chebfun.org/examples/quad/SymbolicNumeric.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'quad')


def _plot(g, color, title, stem):
    xs = np.linspace(-1, 1, 900)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(xs, np.asarray(g(jnp.asarray(xs))), color=color, lw=2.2)
    ax.grid(True)
    ax.set_title(title, fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, stem + ".png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    f = cj.chebfun(lambda x: jnp.log(2 + x) ** 3 * jnp.log(3 + x)
                   * x ** 3)
    fi = f.cumsum()
    print("fi =")
    print(repr(fi))
    _plot(fi, (0, 0.7, 0), "symbolically integrable",
          "SymbolicNumeric_repl_01")
    print("ans =")
    print(f"   {float(f.sum()):.15f}")
    d = (float(np.asarray(fi(jnp.asarray([1.0])))[0])
         - float(np.asarray(fi(jnp.asarray([-1.0])))[0]))
    print("ans =")
    print(f"   {d:.15f}")

    g = cj.chebfun(lambda x: jnp.log(2 + x) ** 3
                   * jnp.log(3 + x) ** 2 * x ** 3)
    gi = g.cumsum()
    print("gi =")
    print(repr(gi))
    _plot(gi, (0.7, 0, 0.7), "not symbolically integrable",
          "SymbolicNumeric_repl_02")
    print("ans =")
    print(f"   {float(g.sum()):.15f}")
    return True


if __name__ == "__main__":
    run()
