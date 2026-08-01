"""AAA approximation of a spline.

Faithful replica of approx/AAASpline.m by Nick Trefethen (February
2018): the poles of a AAA rational approximant to a cubic spline
cluster in beautiful arcs at the knots, where the spline has jumps in
its third derivative.

Original: https://www.chebfun.org/examples/approx/AAASpline.html
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

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.aaa import aaa

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_IMG, exist_ok=True)

    nodes = np.arange(0, 11, dtype=np.float64)
    data = np.sin(nodes + nodes**2 / 4)
    s = Chebfun.spline(jnp.asarray(nodes), jnp.asarray(data))
    X = np.linspace(0, 10, 1000)
    sX = np.asarray(s(jnp.asarray(X)))
    r, poles, *_ = aaa(jnp.asarray(sX), jnp.asarray(X),
                       mmax=200, tol=1e-10)
    p = np.asarray(poles)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(p.real, p.imag, '.r', ms=12)
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_xlim(1.5, 8.5)
    ax.set_ylim(-2, 2)
    ax.set_title(f"poles of AAA approximant, {len(p)} in total",
                 fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "AAASpline_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(p.real, p.imag, '.r', ms=10)
    ax.set_xlim(3.9, 4.1)
    ax.set_ylim(-0.8, 0.8)
    ax.grid(True)
    ax.set_title("poles near x=4", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "AAASpline_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    xs = np.linspace(0, 10, 2000)
    ax.plot(xs, np.asarray(s(jnp.asarray(xs))), lw=1.4)
    ax.plot(nodes, data, '.k', ms=12)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True)
    ax.set_title("the spline being approximated", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "AAASpline_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    err = float(np.linalg.norm(sX - np.real(np.asarray(r(X)))))
    print("error =")
    print(f"     {err:.15e}")


if __name__ == "__main__":
    run()
