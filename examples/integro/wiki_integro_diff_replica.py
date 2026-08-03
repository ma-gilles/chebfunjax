"""Wikipedia integro-differential equation example.

Faithful replica of integro/WikiIntegroDiff.m by Mark Richardson
(September 2010): solve

    u'(x) + 2 u(x) + 5 int_0^x u(t) dt = 1,   u(0) = 0

on [0, 5] with chebop backslash, and compare with the analytic
solution u(x) = exp(-x) sin(2x) / 2.

Original: https://www.chebfun.org/examples/integro/WikiIntegroDiff.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'integro')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    x = cj.chebfun(lambda t: t, domain=(0, 5))
    N = Chebop(lambda x, u: u.diff() + 2 * u + 5 * u.cumsum(),
               domain=(0, 5))
    N.lbc = 0
    u = N.solve(1.0)
    u_exact = 0.5 * (-x).exp() * (2 * x).sin()
    print("accuracy =")
    print(f"     {float((u - u_exact).norm()):.15e}")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    t = np.linspace(0, 5, 600)
    ax.plot(t, np.asarray(u(t)), lw=1.6)
    ax.grid(True)
    ax.set_title("Solution of integro-differential equation",
                 fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "WikiIntegroDiff_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
