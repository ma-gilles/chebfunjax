"""Wikipedia ODE examples.

Faithful replica of ode-linear/WikiODE.m by Mark Richardson
(September 2010): the three linear ODE problems from the Wikipedia
article on ordinary differential equations, solved with chebop
backslash and compared with the analytic solutions.

Original: https://www.chebfun.org/examples/ode-linear/WikiODE.html
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
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

FIG = [0]


def _plot(y, dom, title=None):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    t = np.linspace(dom[0], dom[1], 600)
    ax.plot(t, np.asarray(y(t)), lw=1.6)
    ax.grid(True)
    if title:
        ax.set_title(title, fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"WikiODE_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Problem 1: y'' - 4y' + 5y = 0, Dirichlet BCs
    x = cj.chebfun(lambda t: t, domain=(-1, 1))
    N = Chebop(lambda x, y: y.diff(2) - 4 * y.diff(1) + 5 * y,
               domain=(-1, 1))
    N.lbc = np.exp(-2) * np.cos(-1)
    N.rbc = np.exp(2) * np.cos(1)
    y = N.solve(0.0)
    y_exact = (2 * x).exp() * x.cos()
    print("ans =")
    print(f"     {float((y - y_exact).norm()):.15e}")
    _plot(y, (-1, 1))

    # Problem 2: y'' + pi^2 y = 0, y(-1) = -1, y'(1) = -pi
    N = Chebop(lambda x, y: y.diff(2) + np.pi**2 * y, domain=(-1, 1))
    N.lbc = -1
    N.rbc = lambda u: u.diff() + np.pi
    y = N.solve(0.0)
    y_exact = (np.pi * x).cos() + (np.pi * x).sin()
    print("ans =")
    print(f"     {float((y - y_exact).norm()):.15e}")
    _plot(y, (-1, 1))

    # Problem 3: first-order IVP y' + 3y = 2, y(0) = 2
    x3 = cj.chebfun(lambda t: t, domain=(0, 1))
    N = Chebop(lambda x, y: y.diff() + 3 * y - 2, domain=(0, 1))
    N.lbc = 2
    y = N.solve(0.0)
    y_exact = 2.0 / 3 + 4.0 / 3 * (-3 * x3).exp()
    print("ans =")
    print(f"     {float((y - y_exact).norm()):.15e}")
    _plot(y, (0, 1))


if __name__ == "__main__":
    run()
