"""Smoothies: nowhere-analytic functions.

Faithful replica of stats/Smoothies.m by Nick Trefethen (May 2020):
random functions that are C-infinity but nowhere analytic, generated
by random Fourier series with root-exponentially decaying
coefficients — the 'smoothie' command.

randn draws are not bit-reproducible vs MATLAB; the smoothies are our
own draws from the same distribution.

Original: https://www.chebfun.org/examples/stats/Smoothies.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.random import smoothie

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Smoothies_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _chebfun_from_coeffs(c, trig=False):
    c = jnp.asarray(np.asarray(c).ravel())
    if trig:
        from chebfunjax.chebfun1d.chebfun import Chebfun, Domain, _Piece
        from chebfunjax.tech.trigtech import Trigtech
        tech = Trigtech(coeffs=c)
        return Chebfun(funs=[_Piece(tech=tech,
                                    interval=(-1.0, 1.0))],
                       domain=Domain((-1.0, 1.0)))
    return cj.Chebfun.from_coeffs(c)


def _plotcoeffs(f, trig=False):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    cc = np.abs(np.asarray(f.funs[0].tech.coeffs)).ravel()
    cc = np.maximum(cc, 1e-40)
    if trig:
        n = len(cc)
        ks = np.arange(-(n // 2), n - n // 2)
        order = np.argsort(np.abs(ks))
        ax.semilogy(np.abs(ks)[order], cc[order], '.k', ms=3)
        ax.set_xlabel("Wave number")
    else:
        ax.semilogy(np.arange(len(cc)), cc, '.k', ms=3)
        ax.set_xlabel("Degree of Chebyshev polynomial")
    ax.set_ylabel("Magnitude of coefficient")
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Smoothies_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    key = jax.random.PRNGKey(1)
    f = _chebfun_from_coeffs(smoothie(key=key))
    xs = np.linspace(-1, 1, 2000)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xs, np.asarray(f(xs)), lw=1.0)
    ax.set_ylim(-4, 4)
    ax.grid(True)
    _save(fig)
    _plotcoeffs(f)

    ftrig = _chebfun_from_coeffs(
        smoothie(key=jax.random.PRNGKey(2), trig=True), trig=True)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xs, np.asarray(ftrig(xs)).real, lw=1.0)
    ax.grid(True)
    _save(fig)
    _plotcoeffs(ftrig, trig=True)

    fre = _chebfun_from_coeffs(smoothie(key=jax.random.PRNGKey(3)))
    fim = _chebfun_from_coeffs(smoothie(key=jax.random.PRNGKey(4)))
    zc = (np.asarray(fre(xs)) + 1j * np.asarray(fim(xs))) / np.sqrt(2)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.plot(zc.real, zc.imag, 'm', lw=1.0)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal")
    ax.grid(True)
    _save(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0))
    d1 = f.diff()
    d2 = f.diff(2)
    axes[0].plot(xs, np.asarray(d1(xs)), lw=1.0)
    axes[0].set_ylim(-80, 80)
    axes[0].grid(True)
    axes[1].plot(xs, np.asarray(d2(xs)), lw=1.0)
    axes[1].set_ylim(-8000, 8000)
    axes[1].grid(True)
    _save(fig)


if __name__ == "__main__":
    run()
