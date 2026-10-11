"""Hermite quadrature.

Translation of quad/HermiteQuad.m by Nick Trefethen and Andre
Weideman: Gauss-Hermite quadrature of exp(-x^2) cos(x) versus the
exact sqrt(pi) e^{-1/4}, compared with a simple trapezoidal rule, and
the observation that most Gauss-Hermite nodes lie in the negligible
tail.

Original: https://www.chebfun.org/examples/quad/HermiteQuad.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import StrMethodFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.quadrature import hermpts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'quad')


def run():
    os.makedirs(_IMG, exist_ok=True)
    ff = np.cos
    g = cj.chebfun(lambda x: jnp.exp(-x ** 2) * jnp.cos(x),
                   domain=[-np.inf, np.inf])
    print("ans =")
    print(f"   {float(g.sum()):.15f}")
    exact = np.sqrt(np.pi) * np.exp(-0.25)
    print("exact =")
    print(f"   {exact:.15f}")
    print("ans =")
    print(f"   {float(g.restrict(-6.0, 6.0).sum()):.15f}")

    print("    n        error")
    s = w = None
    for n in range(1, 13):
        s, w = (np.asarray(v) for v in hermpts(n))
        print(f"{n:3d} {float(w @ ff(s)) - exact:19.15f}")

    xs = np.linspace(-8, 8, 1200)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(xs, np.asarray(g(jnp.asarray(xs))), lw=2)
    ax.plot(s, np.asarray(g(jnp.asarray(s))), linestyle="none", color="red",
            marker=MarkerStyle(".").scaled(2/3), ms=16, markeredgewidth=0)
    fig.set_facecolor("white")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-.2, 1)
    ax.set_xticks(np.arange(-8, 9, 2))
    ax.set_yticks(np.arange(-.2, 1.01, .2))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.tick_params(top=True, right=True)
    _savefig(fig, os.path.join(_IMG, "HermiteQuad_01.png"), size=(600, 270), layout="matlab")
    plt.close(fig)

    print("    n        error")
    for n in range(3, 25, 3):
        h = (-1 + np.sqrt(8 * np.pi * n)) / (2 * n)
        d = (n - 1) * h / 2
        sg = jnp.linspace(-d, d, n)
        In = float(jnp.sum(h * g(sg)))              # trapezoidal sum
        print(f"{n:3d} {In - exact:19.15f}")

    # Native default fast Hermite method at all three source sizes.
    for n in (1000, 10000, 100000):
        t0 = time.perf_counter()
        s, w = hermpts(n)
        s.block_until_ready()
        w.block_until_ready()
        print(f"Elapsed time is {time.perf_counter() - t0:.6f} seconds.")

    n = 10000
    s = np.asarray(hermpts(n)[0])
    tail_points = s[np.exp(-s ** 2) < np.finfo(float).eps]
    print("ratio =")
    print(f"    {len(tail_points) / n:.4f}")
    return True


if __name__ == "__main__":
    run()
