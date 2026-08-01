"""Accuracy of Chebyshev coefficients via aliasing.

Faithful replica of approx/AliasingCoefficients.m by Yuji Nakatsukasa
(April 2016): the coefficients of a low-degree Chebyshev interpolant
err by aliased tails of the full expansion — high accuracy in c_0,
very high accuracy in c_n, geometric growth between.

Original: https://www.chebfun.org/examples/approx/AliasingCoefficients.html
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
from chebfunjax.utils.quadrature import chebpts2

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

EPS = float(np.finfo(np.float64).eps)
GREEN = (0.0, 0.7, 0.0)


def _coeff_plot(f, p, fname, xmax=None):
    fc = np.abs(np.asarray(f.coeffs)) + EPS
    pc = np.abs(np.asarray(p.coeffs)) + EPS
    err = np.abs(np.asarray(p.coeffs)
                 - np.asarray(f.coeffs)[:len(np.asarray(p.coeffs))]) + EPS
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.semilogy(np.arange(len(fc)), fc, '.', color=GREEN, ms=10, label='f')
    ax.semilogy(np.arange(len(pc)), pc, '.b', ms=10, label='p')
    ax.semilogy(np.arange(len(err)), err, '.r', ms=10, label='f-p')
    if xmax is not None:
        ax.set_xlim(0, xmax)
    ax.set_xlabel('degree of Chebyshev polynomial', fontsize=12)
    ax.set_ylabel('magnitude of coefficient', fontsize=12)
    ax.legend(fontsize=16)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Section 1: analytic function
    fori = lambda x: jnp.log(jnp.sin(10 * x) + 2)  # noqa: E731
    f = cj.chebfun(fori)
    n_p = round(len(f) / 3)
    p = cj.chebfun(fori, n=n_p)
    print(f"length(f) = {len(f)}, interpolant on {n_p} points")
    _coeff_plot(f, p, "AliasingCoefficients_repl_01.png")

    # Non-analytic function: twice differentiable but not analytic
    fori2 = lambda x: jnp.abs((x - 0.5) ** 3)  # noqa: E731
    f2 = cj.chebfun(fori2)
    n_p2 = round(len(f2) / 6)
    p2 = cj.chebfun(fori2, n=n_p2)
    print(f"length(f2) = {len(f2)}, interpolant on {n_p2} points")
    _coeff_plot(f2, p2, "AliasingCoefficients_repl_02.png",
                xmax=len(f2) / 2)

    # Section 2: two dimensions
    p2d = cj.chebfun2(lambda x, y: jnp.sin(x + y) + jnp.cos(x - y))
    pc = np.asarray(p2d.chebcoeffs2())

    xg, yg = chebpts2(6)
    vals = np.asarray(p2d(jnp.asarray(xg), jnp.asarray(yg)))
    pt = cj.chebfun2(jnp.asarray(vals))
    ptc = np.asarray(pt.chebcoeffs2())

    D = ptc - pc[:ptc.shape[0], :ptc.shape[1]]
    print("ans =")
    for row in np.real(D):
        print("".join(f"{v:13.4e}" for v in row))


if __name__ == "__main__":
    run()
