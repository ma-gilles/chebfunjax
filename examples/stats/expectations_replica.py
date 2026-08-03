"""Mean, median, mode of probability distributions.

Faithful replica of stats/Expectations.m by Jie Gao and Nick
Trefethen (June 2013): moments of an exponential density and the
mean, median and mode of a polynomial density, all as chebfun
computations.

Original: https://www.chebfun.org/examples/stats/Expectations.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')

FIG = [0]


def _plot(f, dom, ylim, ylab):
    FIG[0] += 1
    xs = np.linspace(*dom, 700)
    fig, ax = plt.subplots(figsize=(9.0, 3.3))
    ax.plot(xs, np.asarray(f(xs)), lw=1.6)
    ax.grid(True)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel(ylab, rotation=0)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Expectations_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def run():
    os.makedirs(_IMG, exist_ok=True)

    x = cj.chebfun(lambda t: t, domain=(0.0, 40.0))
    f = cj.chebfun(lambda t: 2 * jnp.exp(-2 * t), domain=(0.0, 40.0))
    _plot(f, (0, 40), (-0.2, 2.2), "f(x)")
    print("ans =")
    print(f"   {float(f.sum()):.15f}")

    xf = x * f
    _plot(xf, (0, 40), (-0.05, 0.4), "x f(x)")
    print("ans =")
    print(f"   {float(xf.sum()):.15f}")

    xxf = x**2 * f
    _plot(xxf, (0, 40), (-0.03, 0.31), "x^2 f(x)")
    print("ans =")
    print(f"   {float(xxf.sum()):.15f}")

    x = cj.chebfun(lambda t: t, domain=(0.0, 3.0))
    g = 4 * x * (9 - x**2) * (1 / 81)
    _plot(g, (0, 3), (-0.01, 0.61), "g(x)")
    mean = float((x * g).sum())
    print("mean =")
    print(f"   {mean:.15f}")

    G = g.cumsum()
    _plot(G, (0, 3), (0, 1.05), "G(x)")
    median = float(np.asarray((G - 0.5).roots())[0])
    print("median =")
    print(f"   {median:.15f}")
    print("median_exact =")
    print(f"   {np.sqrt(9 - 9 * np.sqrt(2) / 2):.15f}")

    mode, gmax = g.max()
    mode = float(mode)
    print("mode =")
    print(f"   {mode:.15f}")
    print("mode_exact =")
    print(f"   {np.sqrt(3):.15f}")

    FIG[0] += 1
    xs = np.linspace(0, 3, 500)
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.plot(xs, np.asarray(g(xs)), lw=1.6)
    ax.grid(True)
    for pos, col, lab, tx in ((mean, 'r', 'mean', 0.2),
                              (median, 'm', 'median', 1.2),
                              (mode, 'k', 'mode', 2.2)):
        ax.plot([pos, pos], [0, float(g(pos))], '-' + col, lw=1.6)
        ax.text(tx, 0.55, f"{lab} = {pos:1.2f}", color=col)
    ax.set_ylim(-0.01, 0.61)
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)", rotation=0)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Expectations_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
