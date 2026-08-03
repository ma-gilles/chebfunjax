"""Boundary layer for advection-diffusion equation.

Faithful replica of ode-linear/BoundaryLayer.m by Nick Trefethen
(October 2010): the steady advection-diffusion problem

    -eps u'' - u' = 1,   u(0) = u(1) = 0

develops a boundary layer of width O(eps) at x = 0; the layer width
(where the solution crosses 1/2) is measured with roots() and shown
to scale linearly with eps.

Original: https://www.chebfun.org/examples/ode-linear/BoundaryLayer.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"BoundaryLayer_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _L(eps):
    N = Chebop(lambda x, u: -eps * u.diff(2) - u.diff(),
               domain=(0, 1))
    N.bc = "dirichlet"
    return N


def _solve(eps):
    return _L(eps).solve(1.0)


def _width(eps):
    u = _solve(eps)
    r = np.asarray((u - 0.5).roots(), dtype=float).ravel()
    return float(np.min(r))


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    t = np.linspace(0, 1, 1200)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for eps, color in ((0.1, 'b'), (0.01, 'r'), (0.001, (0, 0.8, 0))):
        u = _solve(eps)
        ax.plot(t, np.asarray(u(t)), color=color, lw=1.6,
                label=rf"$\epsilon$={eps}")
    ax.grid(True)
    ax.axis([-0.03, 1, 0, 1.03])
    ax.legend()
    ax.set_title(r"Boundary layers for three values of $\epsilon$",
                 fontsize=12)

    w = [_width(0.1), _width(0.01), _width(0.001)]
    print("w =")
    print("   " + "   ".join(f"{v:.15f}" for v in w))
    ax.plot(w, [0.5, 0.5, 0.5], '.k', ms=12)
    _save(fig)

    epsvec = np.array([0.1, 0.03, 0.01, 0.003, 0.001, 0.0003])
    wv = np.array([_width(float(e)) for e in epsvec])
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.loglog(epsvec, wv, '.-k', lw=1.6, ms=12)
    ax.grid(True)
    ax.set_xlabel(r"$\epsilon$", fontsize=12)
    ax.set_ylabel("width of boundary layer", fontsize=12)
    ax.loglog(epsvec, epsvec, '--r', lw=2)
    _save(fig)


if __name__ == "__main__":
    run()
