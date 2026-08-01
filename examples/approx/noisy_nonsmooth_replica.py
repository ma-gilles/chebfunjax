"""Chebfuns of noisy functions with discontinuities.

Faithful replica of approx/NoisyNonsmooth.m by Nick Trefethen (June
2015): edge detection still works in the presence of 1e-8 noise, for a
jump function and for an eigenvalue max with kinks.

Original: https://www.chebfun.org/examples/approx/NoisyNonsmooth.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def _coeffplot(f, title, fname):
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    for piece in f.funs:
        c = np.abs(np.asarray(piece.coeffs)) + 1e-30
        ax.semilogy(np.arange(len(c)), c, '.-', lw=1, ms=8)
    ax.grid(True)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("degree of Chebyshev polynomial")
    ax.set_ylabel("magnitude of coefficient")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    rs = np.random.RandomState(5489)

    def ff(x):
        arr = np.asarray(x)
        noise = 1e-8 * rs.standard_normal(arr.shape)
        return jnp.asarray(np.sign(arr - 0.1) / 2 + np.cos(4 * arr)
                           + noise)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = cj.chebfun(ff, splitting=True, eps=1e-8)
    xs = np.linspace(-1, 1, 3000)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(xs, np.asarray(f(jnp.asarray(xs))), 'm', lw=1.6)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "NoisyNonsmooth_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    _coeffplot(f, "Chebyshev coefficients of the two pieces",
               "NoisyNonsmooth_repl_02.png")
    print("ans =")
    print("  " + "   ".join(f"{float(b):.15f}"
                            for b in f.domain.breakpoints))

    # Eigenvalue max of a matrix pencil: kinks at eigenvalue crossings
    A = np.array([[1, 2, 0], [0, 2, 1], [1, 0, 2]], dtype=float)
    B = np.array([[1, 1, 0], [1, -1, 1], [-1, 1, 1]], dtype=float)
    print("A ="); print(A.astype(int))
    print("B ="); print(B.astype(int))

    def gg(t):
        arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
        out = [np.max(np.abs(np.linalg.eigvals(tv * A + (1 - tv) * B)))
               + 1e-8 * rs.standard_normal()
               for tv in arr.ravel()]
        return jnp.asarray(out, dtype=jnp.float64).reshape(arr.shape)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = cj.chebfun(gg, domain=(0.0, 1.0), splitting=True, eps=1e-8)
    ts = np.linspace(0, 1, 2000)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(ts, np.asarray(g(jnp.asarray(ts))), 'm', lw=1.6)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "NoisyNonsmooth_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("ans =")
    for b in g.domain.breakpoints:
        print(f"   {float(b):.15f}")
    _coeffplot(g, "Chebyshev coefficients", "NoisyNonsmooth_repl_04.png")


if __name__ == "__main__":
    run()
