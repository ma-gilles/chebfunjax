"""Quadrature convergence rates for differentiable functions.

Faithful replica of quad/QuadratureConvergence.m by Nick Trefethen:
for f = |x - 0.3|, both Gauss and Clenshaw-Curtis quadrature converge
at the rate O(n^-2), illustrating that for functions of finite
smoothness the two rules converge at similar rates.

Original: https://www.chebfun.org/examples/quad/QuadratureConvergence.html
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
from chebfunjax.utils.quadrature import chebpts, chebweights, legpts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'quad')


def run():
    os.makedirs(_IMG, exist_ok=True)
    fnp = lambda x: np.abs(np.asarray(x) - 0.3)
    fc = cj.chebfun(lambda x: jnp.abs(x - 0.3), n=100000)
    c = np.abs(np.asarray(fc.funs[0].coeffs))
    k = np.arange(1, len(c) + 1)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    pos = c > 0
    ax.loglog(k[pos], c[pos], ".", ms=2)
    nn = np.round(2.0 ** np.arange(1, 16.1, 0.5))
    ax.loglog(nn, 0.01 * nn ** -2.0, "--k", lw=2)
    ax.text(4e2, 0.5e-9, r"$n^{-2}$", fontsize=18)
    ax.set_xlim(1, 1e5)
    ax.set_ylim(1e-12, 1)
    ax.set_xlabel("n", fontsize=12)
    ax.set_ylabel("Chebyshev coefficient", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "QuadratureConvergence_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # exact integral of |x - 0.3| on [-1, 1]
    exact = float(cj.chebfun(lambda x: jnp.abs(x - 0.3),
                             splitting=True).sum())
    errg, errc = [], []
    nn = np.round(2.0 ** np.arange(1, 16.1, 0.5)).astype(int)
    for n in nn:
        s, w = (np.asarray(v) for v in legpts(int(n)))
        errg.append(abs(float(w @ fnp(s)) - exact))
        s = np.asarray(chebpts(int(n)))
        w = np.asarray(chebweights(int(n)))
        errc.append(abs(float(w @ fnp(s)) - exact))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.loglog(nn, np.maximum(errg, 1e-18), ".-b", ms=8, label="Gauss")
    ax.loglog(nn, np.maximum(errc, 1e-18), ".-r", ms=8,
              label="Clenshaw-Curtis")
    ax.loglog(nn, 0.1 * nn ** -2.0, "--k", lw=2)
    ax.text(2e3, 1e-8, r"$n^{-2}$", fontsize=18)
    ax.grid(True)
    ax.set_xlabel("n", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.legend(loc="lower left")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "QuadratureConvergence_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("exact =", f"{exact:.15f}")
    return True


if __name__ == "__main__":
    run()
