"""The resultant method for bivariate rootfinding.

Faithful replica of roots/ResultantMethod.m by Alex Townsend
(March 2013): common zeros of chebfun2 pairs via the hidden-variable
Bezout resultant method of Nakatsukasa, Noferini & Townsend,
including a degenerate case where marching squares fails and the
resultant method succeeds.

Original: https://www.chebfun.org/examples/roots/ResultantMethod.html
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')

FIG = [0]


def _plot_case(f, g, r, axis_lim=None, title=None):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    for c in f.roots():
        bps = list(c.domain.breakpoints)
        for a, b in zip(bps[:-1], bps[1:]):
            t = np.linspace(a, b, 300)
            v = np.asarray(c(t))
            ax.plot(v.real, v.imag, 'r', lw=1.4)
    for c in g.roots():
        bps = list(c.domain.breakpoints)
        for a, b in zip(bps[:-1], bps[1:]):
            t = np.linspace(a, b, 300)
            v = np.asarray(c(t))
            ax.plot(v.real, v.imag, 'b', lw=1.4)
    if r is not None and len(r):
        ax.plot(r[:, 0], r[:, 1], 'k.', ms=12)
    if axis_lim is not None:
        ax.axis(axis_lim)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=13)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"ResultantMethod_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    f = cj.chebfun2(lambda x, y: jnp.cos(7 * x**2 * y + y))
    g = cj.chebfun2(lambda x, y: jnp.cos(7 * x * y))
    t0 = time.time()
    r = np.atleast_2d(np.asarray(f.roots(g, method="resultant")))
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    _plot_case(f, g, r)
    print(f"[{len(r)} common zeros]")

    w = 10
    f = cj.chebfun2(lambda x, y: jnp.sin(w * x - y / w) + y)
    g = cj.chebfun2(lambda x, y: jnp.cos(w * y - x / w) - x)
    t0 = time.time()
    r = np.atleast_2d(np.asarray(f.roots(g, method="resultant")))
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    _plot_case(f, g, r, axis_lim=[-1, 1, -1, 1])
    resid = max(
        np.linalg.norm(np.asarray(f(r[:, 0], r[:, 1]))),
        np.linalg.norm(np.asarray(g(r[:, 0], r[:, 1]))))
    print("ans =")
    print(f"     {resid:.15e}")

    rect = (-3.45, 3.45, -4.0, 3.0)
    f = cj.chebfun2(
        lambda x, y: 2 * y * jnp.cos(y**2) * jnp.cos(2 * x)
        - jnp.cos(y), domain=rect)
    g = cj.chebfun2(
        lambda x, y: 2 * jnp.sin(y**2) * jnp.sin(2 * x)
        - jnp.sin(x), domain=rect)
    t0 = time.time()
    r = np.atleast_2d(np.asarray(f.roots(g, method="resultant")))
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    _plot_case(f, g, r, axis_lim=list(rect))

    d = (-0.2, 0.2, -1.0, 1.0)
    f = cj.chebfun2(lambda x, y: (y - 5 * x) * (y + 5 * x),
                    domain=d)
    g = cj.chebfun2(lambda x, y: 0.01 * y - x + 0.0001, domain=d)
    r_ms = np.asarray(f.roots(g, method="ms"))
    print("r =")
    if r_ms.size == 0:
        print("     []")
    else:
        r_ms = np.atleast_2d(r_ms)
        for row in r_ms:
            print(f"   {row[0]:.6f} {row[1]:.6f}")
    _plot_case(f, g, np.atleast_2d(r_ms) if r_ms.size else None,
               axis_lim=list(d),
               title="Nearly parallel curves (MATLAB's marching squares misses these)")

    r = np.atleast_2d(np.asarray(f.roots(g, method="resultant")))
    _plot_case(f, g, r, axis_lim=list(d),
               title="Resultant method confirms both solutions")
    print(f"[resultant finds {len(r)} solutions]")


if __name__ == "__main__":
    run()
