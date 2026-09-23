"""Exact solutions of nonlinear ODEs from Bender and Orszag.

Translation of ode-nonlin/ExactSolns.m by Nick Trefethen (December
2010): four nonlinear ODEs from Chapter 1 of Bender & Orszag, each
solved on [1, 2] and compared with its closed-form exact solution, with
the error printed in the figure title.

Original: https://www.chebfun.org/examples/ode-nonlin/ExactSolns.html
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

from chebfunjax import chebfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.quadrature import chebpts_ab

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]
D = (1.0, 2.0)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"ExactSolns_{FIG[0]:02d}.png"))
    plt.close(fig)


def _show(y, exact, title):
    """err = norm(y-exact,inf); plot(y,'.-'), grid on, title(...)"""
    err = float((y - exact).norm(np.inf))
    xx = np.linspace(*D, 2000)
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    ax.plot(xx, np.asarray(y(xx)), "-", lw=1)
    # '.-' marks the Chebyshev points of y
    n = len(y)
    xk = np.asarray(chebpts_ab(n, *D))
    ax.plot(xk, np.asarray(y(xk)), ".", color="C0", ms=9)
    ax.set_xlim(*D)
    ax.grid(True)
    ax.set_title(f"{title}     Error = {err:6.2e}", fontsize=10.5)
    _save(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    x = chebfun(lambda x: x, domain=D)

    # Example 1: separation of variables (I)
    N = Chebop(lambda x_, y: x_ * y.diff() - y**2 + 2 * y, domain=D)
    N.lbc = 0.0
    y = N.solve(1.0)
    _show(y, 1 - 1 / (1 + x.log()), "xy' = y^2-2y+1")

    # Example 2: separation of variables (II)
    N = Chebop(lambda x_, y: y.diff() - y.sin(), domain=D)
    N.lbc = np.pi / 2
    y = N.solve(0.0)
    _show(y, 2 * (x - 1).exp().atan(), "y' = sin(y)")

    # Example 3: order reduction
    N = Chebop(lambda x_, y: y * y.diff(2) - 2 * y.diff()**2, domain=D)
    N.lbc = 1.0
    N.rbc = 2.0
    y = N.solve(0.0)
    _show(y, 2 / (3 - x), "yy'' = 2(y')^2")

    # Example 4: an equidimensional-in-y equation
    N = Chebop(lambda x_, y: y.diff() - y / x_ - x_ / y, domain=D)
    N.lbc = 1.0
    N.init = 1 + 0 * x
    y = N.solve(0.0)
    _show(y, x * (1 + 2 * x.log()).sqrt(), "y' = y/x + x/y")


if __name__ == "__main__":
    run()
