"""Chebfuns from equispaced data.

Faithful replica of approx/EquispacedData.m by Nick Trefethen (June
2015): constructing a chebfun from equispaced samples via Gregory-type
extension ('equi'), versus the catastrophic polynomial interpolant,
plus truncation, loosened tolerance, and noisy data.

Original: https://www.chebfun.org/examples/approx/EquispacedData.html
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
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

PURPLE = (0.8, 0, 1)
XS = np.linspace(-1, 1, 3000)


def _dataplot(f, grid, data, title, fname, color='C0'):
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(XS, np.asarray(f(jnp.asarray(XS))), color, lw=1)
    ax.plot(grid, data, '.k', ms=8)
    ax.set_title(title, fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _coeffplot(f, title, fname):
    c = np.abs(np.asarray(f.coeffs)) + 1e-30
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.semilogy(np.arange(len(c)), c, '.', color=PURPLE, ms=6)
    ax.axis([0, 100, 1e-16, 10])
    ax.grid(True)
    ax.set_title(title, fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    ff = lambda x: np.exp(x) * np.cos(10 * x) * np.tanh(4 * x)  # noqa: E731
    grid = np.linspace(-1, 1, 40)
    data = ff(grid)
    f = cj.chebfun(jnp.asarray(data), equi=True)
    _dataplot(f, grid, data,
              "chebfun constructed from 40 equispaced data values",
              "EquispacedData_repl_01.png")

    fexact = cj.chebfun(
        lambda x: jnp.exp(x) * jnp.cos(10 * x) * jnp.tanh(4 * x))
    print("error =")
    print(f"     {float((f - fexact).norm(np.inf)):.15e}")

    # The polynomial interpolant through the same data: Runge disaster
    runge = Chebfun.interp1(jnp.asarray(grid), jnp.asarray(data))
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(XS, np.asarray(runge(jnp.asarray(XS))), 'r', lw=1)
    ax.plot(grid, data, '.k', ms=8)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "EquispacedData_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("f =")
    print(repr(f))
    _coeffplot(f, "Chebyshev coefficients", "EquispacedData_repl_03.png")

    # Truncate to degree 50
    c = np.asarray(f.coeffs)[:51]
    f50 = cj.chebfun(jnp.asarray(c), coeffs=True)
    print("error50 =")
    print(f"     {float((f50 - fexact).norm(np.inf)):.15e}")
    _coeffplot(f50, "Chebyshev coefficients up to degree 50",
               "EquispacedData_repl_04.png")

    # Loosened tolerance
    floose = cj.chebfun(jnp.asarray(data), equi=True, eps=1e-6)
    print("errorloose =")
    print(f"     {float((floose - fexact).norm(np.inf)):.15e}")
    _coeffplot(floose, "Chebyshev coefficients with loosened tolerance",
               "EquispacedData_repl_05.png")

    # Noisy data (MATLAB randn is not reproducible outside MATLAB; the
    # phenomenon, not the digits, is what replicates here)
    rs = np.random.RandomState(5489)
    noisy = data + 1e-1 * rs.standard_normal(data.shape)
    for ep, fn in ((1e-2, "EquispacedData_repl_06.png"),
                   (3e-2, "EquispacedData_repl_07.png")):
        fn_ = cj.chebfun(jnp.asarray(noisy), equi=True, eps=ep)
        _dataplot(fn_, grid, noisy,
                  f"noisy data with 'equi', eps = {ep:g}: "
                  f"length(f) = {len(fn_)}", fn)
        print(f"eps={ep:g}: length {len(fn_)}")


if __name__ == "__main__":
    run()
