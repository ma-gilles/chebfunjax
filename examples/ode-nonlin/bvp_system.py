"""System of two nonlinear BVPs.

Translation of ode-nonlin/BVPSystem.m by Asgeir Birkisson and Toby
Driscoll (September 2010): a pair of coupled nonlinear ODEs solved two
ways -- with separate unknowns u and v, and with a single indexed
variable.

Original: https://www.chebfun.org/examples/ode-nonlin/BVPSystem.html
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
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"BVPSystem_{FIG[0]:02d}.png"))
    plt.close(fig)


def _problem():
    N = Chebop(lambda x, u, v: [u.diff(2) - v.sin(), v.diff(2) + u.cos()],
               domain=(-1, 1))
    N.lbc = lambda u, v: [u - 1, v.diff()]
    N.rbc = lambda u, v: [v, u.diff()]
    return N


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # --- Solution using multiple variables u and v -------------------
    (u, v), info = _problem().solvebvp([0.0, 0.0])
    nrmduvec = info["normDelta"]

    x = np.linspace(-1, 1, 2000)
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.7))
    axes[0].plot(x, np.asarray(u(x)), lw=2, label="u")
    axes[0].plot(x, np.asarray(v(x)), "--r", lw=2, label="v")
    axes[0].set_title("u and v vs. x", fontsize=8)
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xlabel("x", fontsize=8)
    axes[0].set_ylabel("u(x) and v(x)", fontsize=8)
    axes[1].semilogy(np.arange(1, len(nrmduvec) + 1), nrmduvec, "-*", lw=2)
    axes[1].set_title("Norm of update vs. iteration no.", fontsize=8)
    axes[1].grid(True)
    axes[1].set_xlabel("iteration no.", fontsize=8)
    axes[1].set_ylabel("norm of update", fontsize=8)
    _save(fig)

    # --- The same problem with one indexed (chebmatrix) variable -----
    N = Chebop(lambda x, u: [u[0].diff(2) - u[1].sin(),
                             u[1].diff(2) + u[0].cos()], domain=(-1, 1))
    N.lbc = lambda u: [u[0] - 1, u[1].diff()]
    N.rbc = lambda u: [u[1], u[0].diff()]
    u = N.solve([0.0, 0.0])
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    ax.plot(x, np.asarray(u[0](x)), lw=2, label="$u_1$")
    ax.plot(x, np.asarray(u[1](x)), "--r", lw=2, label="$u_2$")
    ax.set_title("$u_1(x)$ and $u_2(x)$ vs. x", fontsize=8)
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("$u_1(x)$ and $u_2(x)$", fontsize=8)
    _save(fig)


if __name__ == "__main__":
    run()
