"""Odd and even best approximations.

Translation of approx/OddEven.m by Mohsin Javed and Nick
Trefethen (March 2015): best approximation is nonlinear — the sum of
best approximations to the even and odd parts of f is not the best
approximation to f.

Original: https://www.chebfun.org/examples/approx/OddEven.html
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
from chebfunjax.utils.minimax import minimax

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

GREEN = (0.0, 0.7, 0.0)
AX = (-1, 1, -1.2, 1.2)
XS = np.linspace(-1, 1, 2000)


def _p_cheb(res):
    """The minimax polynomial as a chebfun."""
    return cj.chebfun(jnp.asarray(res.coeffs), coeffs=True)


def _v(f):
    return np.asarray(f(jnp.asarray(XS)))


def _single(f, p, title, fname, color=None):
    """One full-size axes: plot(f,'b',p,'r') or plot(p-f,CO,green)."""
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    if color is None:
        ax.plot(XS, _v(f), 'b', lw=1.3)
        ax.plot(XS, _v(p), 'r', lw=1.3)
    else:
        ax.plot(XS, _v(p) - _v(f), color=color, lw=1.3)
    ax.axis(AX)
    ax.grid(True)
    ax.set_title(title, fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, fname))
    plt.close(fig)


def _pair_plot(f, p, err, title1, fname):
    """subplot(2,2,1) and subplot(2,2,2): the lower half stays empty."""
    fig = plt.figure(figsize=(8.8, 4.0))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    fv, pv = _v(f), _v(p)
    ax1.plot(XS, fv, 'b', lw=1.3)
    ax1.plot(XS, pv, 'r', lw=1.3)
    ax1.axis(AX)
    ax1.grid(True)
    ax1.set_title(title1, fontsize=11)
    ax2.plot(XS, pv - fv, color=GREEN, lw=1.3)
    ax2.grid(True)
    ax2.axis(AX)
    ax2.set_title(f"error = {err:.5g}", fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, fname))
    plt.close(fig)


def _parts(fh, n, first):
    """Best approximations of f, its even part, its odd part and their sum."""
    f = cj.chebfun(fh)
    res = minimax(fh, n)
    p = _p_cheb(res)
    if first:
        _single(f, p, "f and its best approximation", "OddEven_01.png")
        _single(f, p, f"error = {res.err:.5g}", "OddEven_02.png",
                color=GREEN)
        k = 3
    else:
        _pair_plot(f, p, res.err, "f and its best approximation",
                   "OddEven_06.png")
        k = 7
    feh = lambda x: (fh(x) + fh(-x)) / 2  # noqa: E731
    feven = cj.chebfun(feh)
    res_e = minimax(feh, n)
    peven = _p_cheb(res_e)
    _pair_plot(feven, peven, res_e.err, "Approximation of the even part",
               f"OddEven_{k:02d}.png")
    foh = lambda x: (fh(x) - fh(-x)) / 2  # noqa: E731
    fodd = cj.chebfun(foh)
    res_o = minimax(foh, n)
    podd = _p_cheb(res_o)
    _pair_plot(fodd, podd, res_o.err, "Approximation of the odd part",
               f"OddEven_{k + 1:02d}.png")
    psum = peven + podd
    errsum = float((f - psum).norm(np.inf))
    _pair_plot(f, psum, errsum, "combined", f"OddEven_{k + 2:02d}.png")


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Degree-0 example: a Gaussian
    _parts(lambda x: jnp.exp(-150 * (x - 0.5) ** 2), 0, True)

    # Degree-1 example: a bactrian camel
    _parts(lambda x: (jnp.exp(-300 * (x - 0.25) ** 2)
                      + jnp.exp(-300 * (x - 0.75) ** 2)), 1, False)


if __name__ == "__main__":
    run()
