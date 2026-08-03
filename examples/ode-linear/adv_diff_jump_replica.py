"""Advection-diffusion equation with a jump in the advection term.

Faithful replica of ode-linear/AdvDiffJump.m by Nick Hale
(November 2014): the steady advection-diffusion problem

    0.2 u'' + b(x) u' = -1,   u(-10) = u(10) = 0,

first with constant advection b = 1, then with the discontinuous
b(x) = (x >= 0) — the coefficient jump routes the solve through the
piecewise (breakpoint) discretization.

Original: https://www.chebfun.org/examples/ode-linear/AdvDiffJump.html
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
        _IMG, f"AdvDiffJump_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t = np.linspace(-10, 10, 1600)

    N = Chebop(lambda x, u: 0.2 * u.diff(2) + u.diff(),
               domain=(-10, 10))
    N.bc = "dirichlet"
    u = N.solve(-1.0)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(t, np.asarray(u(t)), lw=2)
    ax.grid(True)
    ax.axis([-10.1, 10, 0, 20])
    _save(fig)

    N2 = Chebop(lambda x, u: 0.2 * u.diff(2) + (x >= 0) * u.diff(),
                domain=(-10, 10))
    N2.bc = "dirichlet"
    v = N2.solve(-1.0)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(t, np.asarray(v(t)), 'r', lw=2)
    ax.grid(True)
    ax.axis([-10.1, 10, 0, 75])
    _save(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(t, np.asarray(u(t)), 'b', lw=2)
    ax.plot(t, np.asarray(v(t)), '--r', lw=2)
    ax.grid(True)
    ax.axis([-10.1, 10, 0, 75])
    _save(fig)


if __name__ == "__main__":
    run()
