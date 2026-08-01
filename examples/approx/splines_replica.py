"""Splines.

Faithful replica of approx/Splines.m by Nick Trefethen (February
2013): a cubic spline interpolant as a piecewise chebfun, its
derivatives with jumps at the knots, and edge detection recovering the
knots via splitting-on construction.

Original: https://www.chebfun.org/examples/approx/Splines.html
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
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_IMG, exist_ok=True)

    # A smooth function on [0, 10]
    f = cj.chebfun(lambda x: jnp.sin(x + 0.25 * x**2),
                   domain=(0.0, 10.0))
    xs = np.linspace(0, 10, 2000)
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.plot(xs, np.asarray(f(jnp.asarray(xs))), lw=1.4)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.2, 1.2)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Splines_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Spline through samples at the integers
    nodes = np.arange(0, 11, dtype=np.float64)
    vals = np.asarray(f(jnp.asarray(nodes)))
    s = Chebfun.spline(jnp.asarray(nodes), jnp.asarray(vals))
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.plot(xs, np.asarray(f(jnp.asarray(xs))), lw=1.4)
    ax.plot(xs, np.asarray(s(jnp.asarray(xs))), 'r', lw=1.4)
    ax.plot(nodes, vals, '.r', ms=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.2, 1.2)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Splines_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Derivatives 1-3: jumps in the third derivative at the knots
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 7.5))
    for d in range(1, 4):
        sd = s.diff(d)
        ax = axes[d - 1]
        for lo, hi in zip(nodes[:-1], nodes[1:]):
            xx = np.linspace(lo + 1e-9, hi - 1e-9, 120)
            ax.plot(xx, np.asarray(sd(jnp.asarray(xx))), 'b', lw=1.1)
        ax.text(6.5, 0.8 * 2**(d + 1), f"derivative {d}", fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Splines_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Splitting-on reconstruction: edge detection finds the knots
    s2 = cj.chebfun(lambda x: s(x), domain=(0.0, 10.0), splitting=True)
    err = float((s - s2).norm(np.inf))
    print("ans =")
    print(f"     {err:.15e}")
    ends = sorted(float(b) for b in s2.domain.breakpoints)
    print("ans =")
    for e in ends:
        print(f"  {e:.15f}")


if __name__ == "__main__":
    run()
