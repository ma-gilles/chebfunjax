"""The Frank-Kamenetskii blowup equation.

Faithful replica of ode-nonlin/BlowupFK.m by Nick Trefethen
(September 2010): the nonlinear boundary-value problem

    u'' + A e^u = 0,   u(-1) = u(1) = 0,

whose solution grows rapidly as the parameter A approaches the
critical value near 0.878, beyond which no solution exists.

Original: https://www.chebfun.org/examples/ode-nonlin/BlowupFK.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    t = np.linspace(-1, 1, 1200)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    N = Chebop(domain=(-1, 1))
    N.bc = 'dirichlet'
    for A in (0.2, 0.4, 0.6, 0.8, 0.87):
        N.op = lambda u, _A=A: u.diff(2) + _A * u.exp()
        u = N.solve(0.0)
        umax = float(u.max()[1])
        ax.plot(t, np.asarray(u(t)), color=(0.6, 0, 0.5), lw=2)
        ax.text(-0.1, umax + 0.04, f"A = {A:g}", fontsize=14)
        print(f"A = {A:g}: max(u) = {umax:.12f}")
    ax.axis([-1, 1, 0, 1.2])
    ax.grid(True)
    ax.set_title("Frank-Kamenetskii blowup equation", fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "BlowupFK_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
