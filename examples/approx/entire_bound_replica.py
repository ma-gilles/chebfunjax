"""Bernstein-ellipse bounds for entire functions.

Faithful replica of approx/EntireBound.m by Nick Trefethen (April
2017): interpolation errors of exp(x) and cos(100x) against families
of Bernstein-ellipse bounds, whose lower envelope tracks the actual
convergence.

Original: https://www.chebfun.org/examples/approx/EntireBound.html
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

XS = np.linspace(-1, 1, 4001)


def _study(ff, title, rhos, M_of_rho, fname, ms=12):
    fexact = cj.chebfun(ff)
    fv = np.asarray(fexact(jnp.asarray(XS)))
    nmax = len(fexact) - 2
    nvec = np.arange(0, nmax + 1)
    errvec = []
    for n in nvec:
        fn = cj.chebfun(ff, n=int(n) + 1)
        errvec.append(np.max(np.abs(np.asarray(fn(jnp.asarray(XS)))
                                    - fv)))
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.semilogy(nvec, errvec, '.', ms=ms)
    ax.set_xlabel("degree n")
    ax.set_ylabel("error")
    ax.set_title(title, fontsize=12)
    for rho in rhos:
        M = M_of_rho(rho)
        bound = 4 * M * rho**(-nvec.astype(float)) / (rho - 1)
        ax.semilogy(nvec, bound, '-k', lw=1)
        ax.text(1.01 * nmax, bound[-1], rf"$\rho$={rho:g}")
    ax.axis([0, nmax, 1e-16, 1e3])
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{title}: nmax = {nmax}, final err = {errvec[-1]:.2e}")


def run():
    os.makedirs(_IMG, exist_ok=True)
    _study(lambda x: jnp.exp(x), "exp(x)", [2, 4, 8, 16, 32],
           lambda rho: np.exp((rho + 1 / rho) / 2),
           "EntireBound_repl_01.png", ms=18)
    _study(lambda x: jnp.cos(100 * x), "cos(100x)", [1.5, 2, 3, 3.5],
           lambda rho: np.cosh(100 * (rho - 1 / rho) / 2),
           "EntireBound_repl_02.png", ms=8)


if __name__ == "__main__":
    run()
