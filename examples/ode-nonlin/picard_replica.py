"""Picard iteration for ODE existence proof.

Faithful replica of ode-nonlin/Picard.m by Nick Trefethen (January
2016): the Picard-Lindelof iteration

    u^(k+1)(t) = u0 + int_0^t f(s, u^(k)(s)) ds,

carried out with ``cumsum`` on the problem u' = sin(u) + sin(t),
u(0) = 1, on [0, 8].

Original: https://www.chebfun.org/examples/ode-nonlin/Picard.html
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


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Picard_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    d = 8.0
    t = chebfun(lambda t: t, domain=(0, d))
    u0 = 1.0

    L = Chebop(lambda t, u: u.diff() - u.sin(), domain=(0, d))
    L.lbc = u0
    uexact = L.solve(t.sin())

    def f(u, t):
        return u.sin() + t.sin()

    tt = np.linspace(0, d, 2000)
    ex = np.asarray(uexact(tt))

    # Three plots of five successive iterates each.
    u = u0 + 0 * t
    for block, ylim in ((range(0, 5), (-3, 10)),
                        (range(5, 10), (0, 7)),
                        (range(10, 15), (1, 6))):
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for k in block:
            vals = np.asarray(u(tt))
            ax.plot(tt, vals, color="b")
            ax.text(1.015 * d, float(u(np.float64(d))), f"$k = {k}$")
            u = u0 + f(u, t).cumsum()
        ax.plot(tt, ex, color="r")
        ax.set_ylim(*ylim)
        ax.set_xlabel("t")
        ax.set_ylabel("u")
        ax.set_title(
            f"Picard iterates $k = {block[0]},\\dots,{block[-1]}$")
        _save(fig)

    # Errors of iterates 0..4: the k-th should be O(t^(k+1)).
    u = u0 + 0 * t
    tl = np.logspace(-2, np.log10(d), 600)
    ex_l = np.asarray(uexact(tl))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for k in range(5):
        err = np.abs(np.asarray(u(tl)) - ex_l)
        ax.loglog(tl, err, color="k")
        ax.text(8.7, err[0], f"$k = {k}$")
        # slope of the error on the small-t end gives the order
        lo = (tl > 1.2e-2) & (tl < 1e-1)
        slope = np.polyfit(np.log(tl[lo]), np.log(err[lo]), 1)[0]
        print(f"k = {k}: error ~ t^{slope:.3f}   (expect t^{k + 1})")
        u = u0 + f(u, t).cumsum()
    ax.set_xlabel("t")
    ax.set_ylabel("error")
    ax.axis([1e-2, 8, 1e-16, 1e3])
    ax.grid(True)
    ax.set_title("Errors of iterates $0,\\dots,4$")
    _save(fig)


if __name__ == "__main__":
    run()
