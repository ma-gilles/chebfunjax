"""Local complexity of a function.

Faithful replica of approx/Local.m by Nick Trefethen (February 2013):
scanning the local complexity of a function by measuring the chebfun
length of short-window restrictions at loosened tolerance.

Original: https://www.chebfun.org/examples/approx/Local.html
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
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def scan(f, dom, ep, d, fname):
    a, b = dom
    xs = np.linspace(a, b, 3000)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.0))
    axes[0].plot(xs, np.asarray(f(jnp.asarray(xs))), lw=1.4)
    axes[0].set_title("f", fontsize=14)
    np_ = round((b - a) / d)
    xx = np.linspace(a + d, b - d, np_ - 1)
    ll = np.zeros(len(xx))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j, xj in enumerate(xx):
            w = cj.chebfun(lambda t: f(t),
                           domain=(xj - 0.999999 * d, xj + 0.999999 * d),
                           eps=ep)
            ll[j] = len(w)
    axes[1].plot(xx, ll, '.-k', lw=1.2, ms=6)
    axes[1].set_xlim(a, b)
    axes[1].set_title("Local complexity of f", fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{fname}: max local length {int(np.max(ll))}, "
          f"min {int(np.min(ll))}")


def run():
    os.makedirs(_IMG, exist_ok=True)

    f = cj.chebfun(lambda x: jnp.sin(x / (1.02 + jnp.cos(5 * x))))
    scan(f, (-1.0, 1.0), 1e-6, 0.04, "Local_repl_01.png")

    # Solution of an oscillatory ODE: 0.01 u'' + x cos(x) u = 1,
    # u(+-10) = 0
    N = Chebop(lambda x, u: 0.01 * u.diff(2) + (x * x.cos()) * u,
               domain=(-10.0, 10.0), bc=0.0)
    u = N.solve(1.0)
    scan(u, (-10.0, 10.0), 1e-6, 0.2, "Local_repl_02.png")

    # Zoom into [8, 10]
    scan(u, (8.0, 10.0), 1e-6, 0.2, "Local_repl_03.png")


if __name__ == "__main__":
    run()
