"""Lebesgue functions and Lebesgue constants.

Faithful replica of approx/LebesgueConst.m by Nick Trefethen (November
2010): Lebesgue functions of Chebyshev, equispaced, and random node
sets, with their constants.

Original: https://www.chebfun.org/examples/approx/LebesgueConst.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.lebesgue import lebesgue_constant, lebesgue_function
from chebfunjax.utils.quadrature import chebpts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def _panel(ax, nodes, title_fmt, log=False):
    t, lam = lebesgue_function(np.asarray(nodes), n_eval=4001)
    Lambda = lebesgue_constant(np.asarray(nodes), n_eval=20001)
    if log:
        ax.semilogy(t, lam, lw=1.6)
    else:
        ax.plot(t, lam, lw=1.6)
    ax.grid(True)
    ax.set_title(title_fmt(Lambda), fontsize=13)
    return Lambda


def run():
    os.makedirs(_IMG, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.6))
    L1 = _panel(axes[0], chebpts(10),
                lambda L: f"10 Chebyshev points    Lambda = {L:3.2f}")
    L2 = _panel(axes[1], np.linspace(-1, 1, 10),
                lambda L: f"10 equispaced points    Lambda = {L:4.2f}")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "LebesgueConst_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"cheb10 Lambda = {L1:.6f}")
    print(f"equi10 Lambda = {L2:.6f}")

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.6))
    L3 = _panel(axes[0], chebpts(40),
                lambda L: f"40 Chebyshev points    Lambda = {L:3.2f}",
                log=True)
    L4 = _panel(axes[1], np.linspace(-1, 1, 40),
                lambda L: f"40 equispaced points    Lambda = {L:5.2e}",
                log=True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "LebesgueConst_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"cheb40 Lambda = {L3:.6f}")
    print(f"equi40 Lambda = {L4:.6e}")

    rs = np.random.RandomState(5489)   # MATLAB rng(0) == MT default init 5489
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.6))
    L5 = _panel(axes[0], 2 * rs.random_sample(10) - 1,
                lambda L: f"10 random points    Lambda = {L:5.2e}",
                log=True)
    L6 = _panel(axes[1], 2 * rs.random_sample(30) - 1,
                lambda L: f"30 random points    Lambda = {L:5.2e}",
                log=True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "LebesgueConst_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"rand10 Lambda = {L5:.6e}")
    print(f"rand30 Lambda = {L6:.6e}")


if __name__ == "__main__":
    run()
