"""Random polynomials and their roots in [-1,1].

Faithful replica of roots/RandomPolys.m by Nick Trefethen
(July 2017): degree-n polynomials with random normalized-Legendre
coefficients have, on average, a fraction 1/sqrt(3) = 0.5774 of
their roots in [-1,1] as n -> infinity.

randn draws are not bit-reproducible between MATLAB and numpy; the
statistics reproduce, individual draws do not.

Original: https://www.chebfun.org/examples/roots/RandomPolys.html
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
from chebfunjax.utils.transforms import leg2cheb

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"RandomPolys_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    rs = np.random.RandomState(5489)  # rng('default')

    n = 30
    cleg = rs.randn(n + 1)
    ccheb = np.asarray(leg2cheb(jnp.asarray(cleg), normalize=True))
    p = cj.Chebfun.from_coeffs(jnp.asarray(ccheb))
    xs = np.linspace(-1.1, 1.1, 1000)

    rr = np.asarray(p.roots())
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    xs_in = np.linspace(-1, 1, 1000)
    ax.plot(xs_in, np.asarray(p(xs_in)), 'b', lw=1.2)
    ax.axis([-1.1, 1.1, -n, n])
    ax.grid(True)
    ax.plot(rr, np.asarray(p(rr)), '.r', ms=9)
    ratio = len(rr) / n
    ax.set_title(f"fraction of roots in [-1,1]: {ratio:g}",
                 fontsize=11)
    _save(fig)

    r = np.asarray(p.roots(all_roots=True))
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot([-1, 1], [0, 0], 'k')
    ax.grid(True)
    ax.plot(r.real, r.imag, '.r', ms=9)
    ax.set_xlim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(-2, 3))
    _save(fig)

    n = 1000
    data = []
    for _k in range(10):
        cleg = rs.randn(n + 1)
        ccheb = np.asarray(leg2cheb(jnp.asarray(cleg),
                                    normalize=True))
        p = cj.Chebfun.from_coeffs(jnp.asarray(ccheb))
        rr = np.asarray(p.roots())
        ratio = len(rr) / n
        data.append(ratio)
        print(f"fraction of roots in [-1,1]: {ratio:g}")
    print("ans =")
    print(f"   {np.mean(data):.15f}")
    print(f"(theoretical limit 1/sqrt(3) = {1/np.sqrt(3):.15f})")


if __name__ == "__main__":
    run()
