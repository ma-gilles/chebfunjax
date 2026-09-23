"""Chebyshev coefficients.

Translation of cheb/ChebyshevCoeffs.m by Nick Trefethen
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
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'cheb')


def _print_coeffs(a, label=None):
    if label:
        print(label)
    print("a =")
    strs = [f"{v:.15f}" for v in np.asarray(a)]
    w = max(20, max(len(t) for t in strs) + 2)   # MATLAB format long
    for t in strs:
        print(t.rjust(w))


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
    _savefig(fig, os.path.join(_IMG, fname))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    x = cj.chebfun(lambda t: t)

    p = 99 * x**2 + x**3
    _print_coeffs(p.coeffs, "Cheb coeffs of 99x^2 + x^3:")

    e = x.exp()
    _print_coeffs(e.coeffs, "Cheb coeffs of exp(x):")

    _coeffplot(e, "Chebyshev coefficients of exp(x)",
               "ChebyshevCoeffs_01.png", (1e-17, 1e1))

    g = x.exp() / (1 + 10000 * x**2)
    _coeffplot(g, "Chebyshev coefficients of exp(x)/(1+10000x^2)",
               "ChebyshevCoeffs_02.png", (1e-18, 1))

    # sign(x) and its truncated Chebyshev series, chebfun(f,'trunc',10)
    f = x.sign()
    # MATLAB turns splitting on for 'trunc' (chebfun.m parseInputs);
    # chebfunjax does not, so pass it explicitly.
    ptr = cj.chebfun(f, trunc=10, splitting=True)
    _print_coeffs(ptr.coeffs)
    pin = cj.chebfun(f, n=10)
    xs = np.linspace(-1, 1, 3000)
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    sgn = np.sign(xs)
    ax.plot(xs[xs < 0], sgn[xs < 0], 'k', lw=1.4)
    ax.plot(xs[xs > 0], sgn[xs > 0], 'k', lw=1.4)
    ax.plot([0, 0], [-1, 1], 'k', lw=1.4)          # 'jumpline','-'
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("sign(x)", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "ChebyshevCoeffs_03.png"))
    # hold on, plot(p,'m')
    ax.plot(xs, np.asarray(ptr(jnp.asarray(xs))), 'm', lw=1.4)
    ax.set_title("sign(x) and truncated Chebyshev series", fontsize=12)
    _savefig(fig, os.path.join(_IMG, "ChebyshevCoeffs_04.png"))
    # plot(pinterp)
    ax.plot(xs, np.asarray(pin(jnp.asarray(xs))), 'C0', lw=1.4)
    ax.set_title("Same, also with Chebyshev interpolant", fontsize=12)
    _savefig(fig, os.path.join(_IMG, "ChebyshevCoeffs_05.png"))
    plt.close(fig)


if __name__ == "__main__":
    run()
