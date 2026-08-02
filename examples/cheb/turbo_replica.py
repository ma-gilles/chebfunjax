"""Turbocharged Chebyshev coefficients.

Faithful replica of cheb/Turbo.m by Nick Trefethen (January 2019):
the 'turbo' flag doubles the coefficient count via complex-contour
evaluation, dramatically improving derivatives and near-interval
complex evaluation.

Original: https://www.chebfun.org/examples/cheb/Turbo.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import iv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'cheb')


def _disp(v):
    print("ans =")
    if isinstance(v, complex):
        print(f"      {v.real:.15e} + {v.imag:.15e}i" if v.imag >= 0
              else f"      {v.real:.15e} - {abs(v.imag):.15e}i")
    else:
        print(f"   {v:.15f}")


def run():
    os.makedirs(_IMG, exist_ok=True)

    f = cj.chebfun(lambda x: jnp.exp(x))
    ft = cj.chebfun(lambda x: jnp.exp(x), turbo=True)

    c = np.abs(np.asarray(f.coeffs)) + 1e-30
    ct = np.abs(np.asarray(ft.coeffs)) + 1e-30
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.semilogy(np.arange(len(ct)), ct, 'or', ms=8, mfc='none')
    ax.semilogy(np.arange(len(c)), c, '.k', ms=12)
    ax.grid(True)
    ax.set_title("Ordinary and turbocharged Cheb coeffs of exp(x)",
                 fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Turbo_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Tenth derivative at 0 (exact value 1)
    _disp(float(f.diff(10)(jnp.asarray(0.0))))
    _disp(float(ft.diff(10)(jnp.asarray(0.0))))

    # Coefficient accuracy vs exact Bessel values
    n = len(f)
    cex = 2 * iv(np.arange(0, 4 * n), 1.0)
    cex[0] = cex[0] / 2
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.semilogy(np.arange(len(ct)), ct, 'or', ms=8, mfc='none')
    ax.semilogy(np.arange(len(c)), c, '.k', ms=12)
    ax.semilogy(np.arange(n),
                np.abs(np.asarray(f.coeffs) - cex[:n]) + 1e-30,
                '.-k', lw=1, ms=3)
    ax.semilogy(np.arange(len(ct)),
                np.abs(np.asarray(ft.coeffs) - cex[:len(ct)]) + 1e-30,
                '.-r', lw=1, ms=3)
    ax.grid(True)
    ax.set_title("Lines added to show accuracy", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Turbo_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Near-interval complex evaluation
    ff = lambda x: (jnp.exp(x) * (1 + 100 * x**2)  # noqa: E731
                    / (1 + 25 * x**2))
    g = cj.chebfun(ff)
    gt = cj.chebfun(ff, turbo=True)
    exact = complex(np.exp(0.1j) * (1 + 100 * (0.1j) ** 2)
                    / (1 + 25 * (0.1j) ** 2))
    v1 = complex(np.asarray(g(jnp.asarray(0.1j))))
    v2 = complex(np.asarray(gt(jnp.asarray(0.1j))))
    _disp(v1 - exact)
    _disp(v2 - exact)


if __name__ == "__main__":
    run()
