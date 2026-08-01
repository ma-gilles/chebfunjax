"""Approximation of a log-singular function of Pushnitski.

Faithful replica of approx/Pushnitski.m by Nick Trefethen (May 2020):
polynomial, rational-Remez, and CF approximation of
-heaviside(x)/log(x) on [-0.1, 0.1] — a function with a logarithmic
flat spot at the origin.

Original: https://www.chebfun.org/examples/approx/Pushnitski.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.cfpade import cf
from chebfunjax.utils.minimax import minimax

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

XS = np.linspace(-0.1, 0.1, 4000)


def fop(x):
    ax = jnp.where(x > 0, x, 1e-300)
    return jnp.where(x > 0, -1.0 / jnp.log(ax), 0.0)


def run():
    os.makedirs(_IMG, exist_ok=True)

    f = cj.chebfun(fop, domain=[-0.1, 0.0, 0.1])
    fv = np.asarray(f(jnp.asarray(XS)))
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(XS, fv, 'k', lw=2)
    ax.set_ylim(-0.2, 0.5)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Pushnitski_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Chebyshev coefficients of the length-1000 interpolant decay slowly
    f1000 = cj.chebfun(fop, domain=(-0.1, 0.1), n=1000)
    c = np.abs(np.asarray(f1000.coeffs)) + 1e-18
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.loglog(np.arange(1, len(c) + 1), c, '.', ms=4)
    ax.set_xlim(1, 500)
    ax.grid(True)
    ax.set_xlabel("degree of Chebyshev polynomial")
    ax.set_ylabel("magnitude of coefficient")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Pushnitski_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Polynomial minimax of degrees 4, 8, 12, 16
    t0 = time.time()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2))
    for m in range(1, 5):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimax(fop, 4 * m, domain=(-0.1, 0.1),
                          breakpoints=[0.0])
        p = cj.chebfun(jnp.asarray(res.coeffs), coeffs=True,
                       domain=(-0.1, 0.1))
        ev = fv - np.asarray(p(jnp.asarray(XS)))
        ax = axes[(m - 1) // 2, (m - 1) % 2]
        ax.plot(XS, ev, lw=1.1)
        ax.grid(True)
        ax.set_title(f"degree {4*m}", fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Pushnitski_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")

    # Rational minimax of types (0,0)..(3,3)
    t0 = time.time()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2))
    for m in range(1, 5):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r = minimax(fop, m - 1, rational=True, denom=m - 1,
                            domain=(-0.1, 0.1))
                ev = fv - np.asarray(r.r(XS))
            except Exception:
                ev = np.full_like(XS, np.nan)
        ax = axes[(m - 1) // 2, (m - 1) % 2]
        ax.plot(XS, ev, lw=1.1)
        ax.grid(True)
        ax.set_title(f"type ({m-1},{m-1})", fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Pushnitski_repl_04.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")

    # CF approximation of the same types
    t0 = time.time()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2))
    for m in range(1, 5):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                p, q, rh, s = cf(f1000, m - 1, m - 1, 4000)
                ev = fv - np.asarray(rh(jnp.asarray(XS)))
            except Exception:
                ev = np.full_like(XS, np.nan)
        ax = axes[(m - 1) // 2, (m - 1) % 2]
        ax.plot(XS, ev, lw=1.1)
        ax.grid(True)
        ax.set_title(f"type ({m-1},{m-1})", fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Pushnitski_repl_05.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")


if __name__ == "__main__":
    run()
