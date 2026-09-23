"""Rational minimax approximation of the Fermi-Dirac function.

Translation of approx/FermiDirac.m by Nick Trefethen (June 2019):
type (n,n) minimax approximation of the transplanted Fermi-Dirac
function 1/(1+exp(x-L)), with pole plots.

Original: https://www.chebfun.org/examples/approx/FermiDirac.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.minimax import minimax
from chebfunjax.utils.quadrature import chebpts_ab

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def make_g(L):
    return lambda st: 1.0 / (1.0 + jnp.exp((st * L + L) / (1 - st) - L))


def fermi(L, n, fname):
    t0 = time.time()
    g = make_g(L)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = minimax(g, n, rational=True, denom=n)
    err = r.err
    poles = np.asarray(r.poles)
    ss = np.concatenate([np.asarray(chebpts_ab(1000, -1.0, 0.0)),
                         np.asarray(chebpts_ab(1000, 0.0, 1.0))])
    ev = np.asarray(r.r(ss)) - np.asarray(g(jnp.asarray(ss)))
    fig = plt.figure(figsize=(9.6, 6.4))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(ss, ev, lw=1.2)
    ax.plot([-1, 1], [-err, -err], '--r', lw=1)
    ax.plot([-1, 1], [err, err], '--r', lw=1)
    ax.set_ylim(-2 * err, 2 * err)
    ax.grid(True)
    ax.set_title(f"Fermi-Dirac transplanted to [-1,1], "
                 f"L = {L}, n = {n}", fontsize=12)
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(poles.real, poles.imag, '.r', ms=8)
    ax.set_aspect("equal")
    ax.axis([-20, 20, -10, 10])
    ax.grid(True)
    ax.set_title("poles", fontsize=11)
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(poles.real, poles.imag, '.r', ms=8)
    ax.set_aspect("equal")
    ax.axis([-1, 1, -0.5, 0.5])
    ax.grid(True)
    ax.set_title("closeup", fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, fname))
    plt.close(fig)
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")


def run():
    os.makedirs(_IMG, exist_ok=True)

    L = 20
    f = lambda x: 1.0 / (1 + np.exp(np.minimum(x - L, 50)))  # noqa: E731
    xs = np.linspace(0, 80, 2000)
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.plot(xs, f(xs), lw=1.4)
    ax.grid(True)
    ax.set_ylim(-1, 2)
    ax.set_title("physical domain", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "FermiDirac_01.png"), size=(600, 270))
    plt.close(fig)

    g = make_g(L)
    ss = np.linspace(-1, 0.995, 2000)
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.plot(ss, np.asarray(g(jnp.asarray(ss))), lw=1.4)
    ax.grid(True)
    ax.set_ylim(-1, 2)
    ax.set_title("transplantation to [-1,1]", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "FermiDirac_02.png"), size=(600, 270))
    plt.close(fig)

    v1 = float(g(jnp.asarray(0.1)))
    v2 = 1.0 - float(g(jnp.asarray(-0.1)))
    print(f"   {v1:.15f}   {v2:.15f}")

    fermi(10, 10, "FermiDirac_03.png")
    fermi(100, 15, "FermiDirac_04.png")
    fermi(1000, 20, "FermiDirac_05.png")
    fermi(1000, 30, "FermiDirac_06.png")


if __name__ == "__main__":
    run()
