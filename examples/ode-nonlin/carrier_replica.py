"""The Carrier equation.

Faithful replica of ode-nonlin/Carrier.m by Nick Trefethen and Asgeir
Birkisson (October 2010): the Carrier boundary-layer problem

    0.01 u'' + 2(1-x^2) u + u^2 = 1,   u(-1) = u(1) = 0,

which has many solutions — the one Newton finds depends on the initial
guess — together with the Newton convergence history and a variant
with a Robin condition at the right end.

Original: https://www.chebfun.org/examples/ode-nonlin/Carrier.html
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

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]
_XF = Chebfun.identity(Domain((-1.0, 1.0)))


def _op(x, u):
    return 0.01 * u.diff(2) + 2 * (1 - x**2) * u + u**2


def _accuracy(u):
    return float((_op(_XF, u) - 1).norm())


def _save(u, nrmdu):
    FIG[0] += 1
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.6))
    t = np.linspace(-1, 1, 1500)
    axs[0].plot(t, np.asarray(u(t)), lw=1.6)
    axs[0].set_title("Solution", fontsize=14)
    axs[0].grid(True)
    axs[1].semilogy(np.arange(1, len(nrmdu) + 1), nrmdu, '.-r',
                    lw=1.6, ms=12)
    axs[1].set_title("Convergence", fontsize=14)
    axs[1].set_xlim(1, max(len(nrmdu), 2))
    axs[1].grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Carrier_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    x = cj.chebfun(lambda t: t, domain=(-1, 1))

    # A simple initial guess
    N = Chebop(_op, domain=(-1, 1))
    N.bc = 0
    N.init = 2 * (x**2 - 1)
    u, info = N.solvebvp(1.0)
    _save(u, info["normDelta"])
    print("accuracy =")
    print(f"     {_accuracy(u):.15e}")

    # A wigglier initial guess finds a different solution
    N = Chebop(_op, domain=(-1, 1))
    N.bc = 0
    N.init = 2 * (x**2 - 1) * (1 - 2 / (1 + 20 * x**2))
    u, info = N.solvebvp(1.0)
    _save(u, info["normDelta"])
    print("accuracy =")
    print(f"     {_accuracy(u):.15e}")

    # Same equation, Robin condition at the right end
    N = Chebop(_op, domain=(-1, 1))
    N.init = 2 * (x**2 - 1) * (1 - 2 / (1 + 20 * x**2))
    N.lbc = 1
    N.rbc = lambda u: u.diff() + u
    u, info = N.solvebvp(1.0)
    _save(u, info["normDelta"])
    print("accuracy =")
    print(f"     {_accuracy(u):.15e}")


if __name__ == "__main__":
    run()
