"""Chebyshev coefficients.

Faithful replica of cheb/ChebyshevCoeffs.m by Nick Trefethen
(September 2010): chebcoeffs of polynomials and smooth functions,
coefficient decay plots, and the truncated Chebyshev series of
sign(x).

Original: https://www.chebfun.org/examples/cheb/ChebyshevCoeffs.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'cheb')


def _print_coeffs(a, label=None):
    if label:
        print(label)
    print("a =")
    for v in np.asarray(a):
        print(f"  {v:.15f}" if v < 0 else f"   {v:.15f}")


def _coeffplot(f, title, fname, ylim):
    c = np.abs(np.asarray(f.coeffs)) + 1e-30
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.semilogy(np.arange(len(c)), c, '.', ms=6)
    ax.grid(True)
    ax.set_xlabel("degree n")
    ax.set_ylabel(r"$|a_n|$")
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    x = cj.chebfun(lambda t: t)

    p = 99 * x**2 + x**3
    _print_coeffs(p.coeffs, "Cheb coeffs of 99x^2 + x^3:")

    e = x.exp()
    _print_coeffs(e.coeffs, "Cheb coeffs of exp(x):")

    _coeffplot(e, "Chebyshev coefficients of exp(x)",
               "ChebyshevCoeffs_repl_01.png", (1e-17, 1e1))

    g = x.exp() / (1 + 10000 * x**2)
    _coeffplot(g, "Chebyshev coefficients of exp(x)/(1+10000x^2)",
               "ChebyshevCoeffs_repl_02.png", (1e-18, 1))

    # sign(x): the exact Chebyshev series has a_k = 4/(pi k) (-1)^((k-1)/2)
    # for odd k; truncate to 10 terms (MATLAB chebfun(f,'trunc',10)).
    a = np.zeros(10)
    for k in range(1, 10, 2):
        a[k] = 4 / (np.pi * k) * (-1.0) ** ((k - 1) // 2)
    _print_coeffs(a)
    ptr = cj.chebfun(jnp.asarray(a), coeffs=True)
    pin = cj.chebfun(lambda t: jnp.sign(t), n=10,
                     domain=(-1.0, 1.0))
    xs = np.linspace(-1, 1, 3000)
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    sgn = np.sign(xs)
    ax.plot(xs[xs < 0], sgn[xs < 0], 'k', lw=1.4)
    ax.plot(xs[xs > 0], sgn[xs > 0], 'k', lw=1.4)
    ax.plot(xs, np.asarray(ptr(jnp.asarray(xs))), 'm', lw=1.4)
    ax.plot(xs, np.asarray(pin(jnp.asarray(xs))), 'C0', lw=1.4)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("sign(x), truncated Chebyshev series, and interpolant",
                 fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ChebyshevCoeffs_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
