"""Discrete and continuous least squares.

Faithful replica of stats/LeastSquares.m by Alex Townsend
(March 2013): polynomial least-squares fitting of noisy discrete
data, and continuous least-squares fitting of a piecewise function.

Original: https://www.chebfun.org/examples/stats/LeastSquares.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    npts = 100
    x = np.linspace(-1, 1, npts)
    rs = np.random.RandomState(5489)
    y = 1.0 / (1 + 25 * x**2) + 1e-1 * rs.randn(npts)
    c = np.polynomial.chebyshev.chebfit(x, y, 10)
    xs = np.linspace(-1, 1, 600)
    fit = np.polynomial.chebyshev.chebval(xs, c)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(x, y, 'xk', ms=7)
    ax.plot(xs, fit, 'r', lw=1.6)
    ax.set_title("Discrete polynomial least-squares fit",
                 fontsize=12)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "LeastSquares_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    f = cj.chebfun(
        lambda t: jnp.abs(t + 0.2) - 0.5 * jnp.sign(t - 0.5),
        splitting=True)
    r = f.polyfit(10)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    bps = [float(v) for v in f.domain.breakpoints]
    for a, b in zip(bps[:-1], bps[1:]):
        t = np.linspace(a, b, 200)
        ax.plot(t, np.asarray(f(t)), 'k', lw=1.4)
    ax.plot(xs, np.asarray(r(xs)), 'r', lw=1.6)
    ax.set_title("Continuous polynomial least-squares fit",
                 fontsize=12)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "LeastSquares_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
