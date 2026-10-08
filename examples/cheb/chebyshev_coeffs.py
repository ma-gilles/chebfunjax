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

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, matlab_plot, plotcoeffs
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
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    plotcoeffs(f, ax=ax, source=True)
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
    fig, ax = plt.subplots(figsize=(600 / 72.009, 253 / 72.009))
    # Reference page axes occupy 13%..90.5% horizontally and 11%..91%
    # vertically; preserve this box instead of tight_layout reflow.
    ax.set_position([0.13, 0.11, 0.775, 0.8])
    matlab_plot(f, 'k', ax=ax, jumpline={'linestyle': '-', 'color': 'k'}, linewidth=1.6)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1], ['-1', '-0.5', '0', '0.5', '1'])
    ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5],
                  ['-1.5', '-1', '-0.5', '0', '0.5', '1', '1.5'])
    ax.tick_params(labelsize=12)
    ax.set_title("sign(x)", fontsize=14)
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(_IMG, "ChebyshevCoeffs_03.png"), size=(600, 253), dpi=72.009)
    # Source hold-on overlay retains the manual y-limits and original black line.
    matlab_plot(ptr, 'm', ax=ax, linewidth=1.6)
    ax.set_title("sign(x) and truncated Chebyshev series", fontsize=14)
    _savefig(fig, os.path.join(_IMG, "ChebyshevCoeffs_04.png"), size=(600, 253), dpi=72.009)
    matlab_plot(pin, ax=ax, linewidth=1.6)
    ax.set_title("Same, also with Chebyshev interpolant", fontsize=14)
    _savefig(fig, os.path.join(_IMG, "ChebyshevCoeffs_05.png"), size=(600, 253), dpi=72.009)
    plt.close(fig)


if __name__ == "__main__":
    run()
