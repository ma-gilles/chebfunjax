"""Bernstein polynomials.

Faithful replica of approx/BernsteinPolys.m by Nick Trefethen (May
2012): Bernstein's random-walk proof of the Weierstrass Approximation
Theorem in action — slow, monotone, Gibbs-free convergence that takes
no advantage of smoothness.

Original: https://www.chebfun.org/examples/approx/BernsteinPolys.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, Domain
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.quadrature import chebpts_ab

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def Bn(f, n):
    """Degree-n Bernstein polynomial of f on [0,1], via interpolation
    at Chebyshev points (as in the MATLAB example)."""
    x = np.asarray(chebpts_ab(n + 1, 0.0, 1.0))
    data = np.zeros_like(x)
    for k in range(n + 1):
        fk = float(f(jnp.asarray(k / n)))
        data = data + fk * comb(n, k, exact=False) * x**k * (1 - x)**(n - k)
    return Chebfun.from_values(jnp.asarray(data), domain=Domain((0.0, 1.0)))


def _plot_pair(f, B, title, fname):
    xs = np.linspace(0, 1, 1500)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(xs, np.asarray(f(jnp.asarray(xs))), lw=1.6)
    ax.plot(xs, np.asarray(B(jnp.asarray(xs))), 'r', lw=1.6)
    ax.set_title(title, fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # A continuous function on [0,1]
    s = cj.chebfun(lambda t: t, domain=(0.0, 1.0))
    f = (s - 0.3).abs().minimum(2.0 * (s - 0.7).abs())
    f = s + (1.0 - 5.0 * f).maximum(0.0)

    xs = np.linspace(0, 1, 1500)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(xs, np.asarray(f(jnp.asarray(xs))), lw=1.6)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "BernsteinPolys_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Slow convergence as n increases
    for i, n in enumerate([25, 50, 100]):
        _plot_pair(f, Bn(f, n), f"n = {n}",
                   f"BernsteinPolys_repl_{i+2:02d}.png")

    # A far smoother function: convergence no better
    f2 = s + ((-50 * (s - 0.3)**2).exp() + (-200 * (s - 0.7)**2).exp())
    for i, n in enumerate([25, 50, 100]):
        _plot_pair(f2, Bn(f2, n), f"n = {n}",
                   f"BernsteinPolys_repl_{i+5:02d}.png")

    # Chebyshev interpolation nails the smooth function at n = 100:
    print("ans =")
    print(f"    {len(f2)}")


if __name__ == "__main__":
    run()
