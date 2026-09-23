"""The Carrier equation.

Translation of ode-nonlin/Carrier.m by Nick Trefethen and Asgeir
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
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]


def _save(u, nrmdu):
    FIG[0] += 1
    fig, axs = plt.subplots(1, 2)
    matlab_plot(u, ax=axs[0], lw=1.6)
    axs[0].set_title("Solution", fontsize=14)
    axs[1].semilogy(np.arange(1, len(nrmdu) + 1), nrmdu, '.-r',
                    lw=1.6, ms=16)
    axs[1].set_title("Convergence", fontsize=14)
    axs[1].set_xlim(1, len(nrmdu))
    axs[1].grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"Carrier_{FIG[0]:02d}.png"))
    plt.close(fig)


def _show_accuracy(N, u):
    print("accuracy =")
    print(f"     {float((N(u) - 1).norm()):.15e}")


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    N = Chebop(lambda x, u: 0.01 * u.diff(2) + 2 * (1 - x ** 2) * u + u ** 2,
               domain=(-1, 1))
    N.bc = 'dirichlet'
    x = cj.chebfun('x')
    N.init = 2 * (x ** 2 - 1)

    # cheboppref.setDefaults('display','iter'): our Chebop has no
    # Newton iteration display, so MATLAB's iteration table is not printed.
    u, info = N.solvebvp(1)
    nrmdu = info["normDelta"]
    _save(u, nrmdu)
    _show_accuracy(N, u)

    N.init = 2 * (x ** 2 - 1) * (1 - 2 / (1 + 20 * x ** 2))
    u, info = N.solvebvp(1)
    nrmdu = info["normDelta"]
    _save(u, nrmdu)
    _show_accuracy(N, u)

    N.lbc = 1
    N.rbc = lambda u: u.diff() + u
    u, info = N.solvebvp(1)
    nrmdu = info["normDelta"]
    _save(u, nrmdu)
    _show_accuracy(N, u)


if __name__ == "__main__":
    run()
