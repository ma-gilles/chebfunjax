"""Halphen's constant for approximation of exp(x).

Faithful replica of approx/Halphen.m by Nick Trefethen (May 2011): the
"one-ninth" constant 9.28903... governing type (n,n) rational
approximation of e^x on (-inf,0], computed as a root of an Eisenstein
series via chebfun rootfinding.

Original: https://www.chebfun.org/examples/approx/Halphen.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_IMG, exist_ok=True)

    halphen_const = 9.289025491920818918755449435951
    print("halphen_const =")
    print(f"   {halphen_const:.15f}")

    n = np.arange(0, 11)
    err = np.array([.5, .0668, 7.36e-3, 7.99e-4, 8.65e-5, 9.35e-6,
                    1.01e-6, 1.09e-7, 1.17e-8, 1.26e-9, 1.36e-10])
    model = 2 * halphen_const**(-n - 0.5)
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.semilogy(n, model, '-b', lw=1.2)
    ax.semilogy(n, err, '.k', ms=14)
    ax.grid(True)
    ax.set_xlabel("n")
    ax.set_ylabel("error")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Halphen_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Eisenstein-series computation of the constant
    s = cj.chebfun(lambda t: t, domain=(1.0 / 12, 1.0 / 6))
    f = 0.0 * s
    k = 0
    normsk = 999.0
    while normsk > 1e-16:
        k += 1
        sk = s**k
        f = f + k * sk / (1 - (-1.0)**k * sk)
        normsk = float(sk.norm(np.inf))

    roots = np.atleast_1d(np.asarray((f - 1.0 / 8).roots()))
    h = 1.0 / float(roots[0])
    print("h =")
    print(f"   {h:.13f}")

    xs = np.linspace(1.0 / 12, 1.0 / 6, 800)
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.plot(1.0 / xs, np.asarray(f(jnp.asarray(xs))), lw=1.3)
    ax.plot([h], [1.0 / 8], '.r', ms=24)
    ax.grid(True)
    ax.set_title("Halphen's constant", fontsize=12)
    ax.text(h, 0.135, f"{h:16.13f}")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Halphen_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
