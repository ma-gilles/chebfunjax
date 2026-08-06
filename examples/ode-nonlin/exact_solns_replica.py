"""Exact solutions of nonlinear ODEs from Bender and Orszag.

Faithful replica of ode-nonlin/ExactSolns.m by Nick Trefethen (December
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

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]
D = (1.0, 2.0)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"ExactSolns_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _show(y, exact, title):
    xx = np.linspace(*D, 2000)
    err = float(np.max(np.abs(np.asarray(y(xx)) - exact(xx))))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(xx, np.asarray(y(xx)), ".-", lw=1, markevery=100, markersize=8)
    ax.grid(True)
    ax.set_title(f"{title}     Error = {err:6.2e}", fontsize=14)
    _save(fig)
    print(f"{title:24s} Error = {err:6.2e}")


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    x = chebfun(lambda x: x, domain=D)

    # Example 1: separation of variables (I)
    N = Chebop(lambda x_, y: x_ * y.diff() - y**2 + 2 * y, domain=D)
    N.lbc = 0.0
    y = N.solve(1.0)
    _show(y, lambda t: 1 - 1 / (1 + np.log(t)), "xy' = y^2-2y+1")

    # Example 2: separation of variables (II)
    N = Chebop(lambda x_, y: y.diff() - y.sin(), domain=D)
    N.lbc = np.pi / 2
    y = N.solve(0.0)
    _show(y, lambda t: 2 * np.arctan(np.exp(t - 1)), "y' = sin(y)")

    # Example 3: order reduction
    N = Chebop(lambda x_, y: y * y.diff(2) - 2 * y.diff()**2, domain=D)
    N.lbc = 1.0
    N.rbc = 2.0
    y = N.solve(0.0)
    _show(y, lambda t: 2 / (3 - t), "yy'' = 2(y')^2")

    # Example 4: an equidimensional-in-y equation
    N = Chebop(lambda x_, y: y.diff() - y / x_ - x_ / y, domain=D)
    N.lbc = 1.0
    N.init = 1 + 0 * x
    y = N.solve(0.0)
    _show(y, lambda t: t * np.sqrt(1 + 2 * np.log(t)), "y' = y/x + x/y")


if __name__ == "__main__":
    run()
