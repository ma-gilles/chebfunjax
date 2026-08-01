"""Hermite quadrature.

Faithful replica of quad/HermiteQuad.m by Nick Trefethen and Andre
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

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
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
    ax.plot(s, np.asarray(g(jnp.asarray(s))), ".r", ms=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "HermiteQuad_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("    n        error")
    gnp = lambda x: np.exp(-x ** 2) * np.cos(x)
    for n in range(3, 25, 3):
        h = (-1 + np.sqrt(8 * np.pi * n)) / (2 * n)
        d = (n - 1) * h / 2
        sg = np.linspace(-d, d, n)
        print(f"{n:3d} {float(np.sum(h * gnp(sg))) - exact:19.15f}")

    # MATLAB computes hermpts(100000) in ~0.05 s via asymptotic methods;
    # our hermpts is Golub-Welsch (O(n^2) eigenvalue) -- the asymptotic
    # fast path is a ledgered feature gap.  The tail-fraction phenomenon
    # is scale-invariant, shown here at n = 2000.
    n = 2000
    s, _ = (np.asarray(v) for v in hermpts(n))
    tail = s[np.exp(-s ** 2) < np.finfo(float).eps]
    print("ratio =")
    print(f"    {len(tail) / n:.4f}")
    return True


if __name__ == "__main__":
    run()
