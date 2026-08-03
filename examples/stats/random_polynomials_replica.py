"""Random polynomials and random walks.

Faithful replica of stats/RandomPolynomials.m by Nick Trefethen
(April 2017): scaled Legendre and Foster-Habermann polynomials as
orthonormal bases for random polynomials whose limits are white
noise and Brownian paths.

randn draws are not bit-reproducible vs MATLAB; the pictures are our
own draws from the same families.

Original: https://www.chebfun.org/examples/stats/RandomPolynomials.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.randnfun import randnfun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')

FIG = [0]
XS = np.linspace(0, 1, 3000)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"RandomPolynomials_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


# Shifted Legendre values on [0,1]: P_k(2x-1), all degrees at once
_LMAX = 520
_LV = np.polynomial.legendre.legvander(2 * XS - 1, _LMAX)


def leg01_vals(n):
    return _LV[:, n]


def scaled_legendre_vals(n):
    return leg01_vals(n) * np.sqrt(2 * n + 1)


def foster_vals(n):
    return (leg01_vals(n) - leg01_vals(n - 2)) / np.sqrt(8 * n - 4)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    r = randnfun(0.01, domain=(0.0, 1.0), key=jax.random.PRNGKey(1))
    b = r.cumsum()
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4))
    axes[0].plot(XS, np.asarray(r(XS)), 'k', lw=0.6)
    axes[0].grid(True)
    axes[0].set_title("smooth random function")
    axes[1].plot(XS, np.asarray(b(XS)), lw=1.0)
    axes[1].grid(True)
    axes[1].set_title("smooth random walk")
    _save(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(XS, foster_vals(50), lw=1.2)
    ax.grid(True)
    ax.set_title("Foster-Habermann polynomial, degree 50")
    _save(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for n in range(2, 7):
        ax.plot(XS, foster_vals(n), lw=1.2)
    ax.set_ylim(-0.5, 0.5)
    ax.grid(True)
    ax.set_title("Foster-Habermann polynomials, degrees 2-6")
    _save(fig)

    for i, n in enumerate([20, 100, 500]):
        rs = np.random.RandomState(3)
        coef = rs.randn(n + 1)
        scl = np.sqrt(2 * np.arange(n + 1) + 1)
        poly_vals = _LV[:, :n + 1] @ (coef * scl)
        rs = np.random.RandomState(3)
        c0 = rs.randn()
        cw = rs.randn(n)
        ks = np.arange(2, n + 2)
        Fmat = ((_LV[:, ks] - _LV[:, ks - 2])
                / np.sqrt(8 * ks - 4)[None, :])
        walk_vals = c0 * XS + Fmat @ cw
        fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4))
        axes[0].plot(XS, poly_vals, 'k', lw=0.6)
        axes[0].set_ylim(-100, 100)
        axes[0].grid(True)
        axes[0].set_title(f"random polynomial of degree {n}")
        axes[1].plot(XS, walk_vals, lw=1.0)
        axes[1].set_ylim(-0.75, 0.75)
        axes[1].grid(True)
        axes[1].set_title("polynomial random walk")
        _save(fig)

    s_vals = XS - XS**2
    for k in range(2, 21):
        s_vals = s_vals - foster_vals(k) ** 2
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(XS, s_vals, lw=1.4)
    ax.grid(True)
    ax.set_title("Nearly-semicircular variance profile")
    _save(fig)


if __name__ == "__main__":
    run()
