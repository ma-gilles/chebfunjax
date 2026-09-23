"""Eigenstates of the Schroedinger equation.

Translation of ode-eig/Eigenstates.m by Nick Trefethen (January
2012): `quantumstates` computes and plots eigenstates of

    L u = -h^2 u'' + V(x) u = lam u

for a sequence of potentials -- harmonic, square wells, absolute value
-- with each eigenfunction drawn at the height of its energy level, in
the style of the MATLAB original.

Original: https://www.chebfun.org/examples/ode-eig/Eigenstates.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun1d.chebfun import chebfun, quantumstates
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-eig')

FIG = [0]


def _draw(V, d, U, n=10, h=0.1):
    """The plot drawn by MATLAB @chebfun/quantumstates.m."""
    FIG[0] += 1
    xmin, xmax = float(V.domain.a), float(V.domain.b)
    fig, ax = plt.subplots()
    lw = 1
    jl = {'linestyle': '-', 'color': 'k'}
    matlab_plot(V, 'k', ax=ax, linewidth=lw, jumpline=jl)
    ax.set_title(f"h = {h:4g}      {n} eigenstates", fontsize=12)
    d = np.asarray(d, dtype=float)
    xx = np.linspace(xmin, xmax, 2001)
    Vv = np.asarray(V(xx))
    ymax = d.max()
    ymin = float(V.min()[1])            # min returns (x, f(x))
    ydiff = ymax - ymin
    ymax = ymax + .2 * ydiff
    Vxmin, Vxmax = float(V(xmin)), float(V(xmax))
    dx = .05 * (xmax - xmin)
    dy = .25 * ydiff / max(5, n)
    for j, u in enumerate(U):
        w = dy * u / float(u.norm())       # eigs returns L2-normalized modes
        (_, lo), (_, hi) = w.minandmax()
        if float(hi) < -float(lo):
            w = -w
        matlab_plot(w + float(d[j]), ax=ax, linewidth=lw,
                    color=f"C{j % 7}")
    ax.plot(xx, Vv, 'k', lw=lw)
    for xe, Ve in ((xmin, Vxmin), (xmax, Vxmax)):
        if ymax > Ve:
            ax.plot([xe, xe], [ymax, Ve], 'k', lw=lw)
    matlab_plot(V, 'k', ax=ax, linewidth=lw, jumpline=jl)
    ax.axis([xmin - dx, xmax + dx, ymin - dy, ymax])
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"Eigenstates_{FIG[0]:02d}.png"))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t_start = time.time()
    x = chebfun(lambda x: x, domain=(-3.0, 3.0))
    V = x**2

    # quantumstates(V) with no semicolon displays ans = d
    lam, funs = quantumstates(V)
    _draw(V, lam, funs)
    print("ans =")
    for v in np.asarray(lam):
        print(f"   {v:.15f}")

    for n, h in ((60, 0.1), (10, 0.01), (20, 0.5)):
        lam, funs = quantumstates(V, n=n, h=h)
        _draw(V, lam, funs, n, h)

    V = 10 - 10 * (abs(x) < 1)
    lam, funs = quantumstates(V)
    _draw(V, lam, funs)

    V = 1 - (abs(x) < 1)
    lam, funs = quantumstates(V, n=20)
    _draw(V, lam, funs, 20)

    lam, funs = quantumstates(abs(x))
    _draw(abs(x), lam, funs)

    V = (abs(x) + .1).sqrt()
    lam, funs = quantumstates(V)
    _draw(V, lam, funs)

    V = 0.5 * (abs(x - .5) < .5)
    lam, funs = quantumstates(V, n=18)
    _draw(V, lam, funs, 18)

    V = 0.5 * (-2 * (x - .5)**2).exp()
    lam, funs = quantumstates(V, n=18)
    _draw(V, lam, funs, 18)

    print(f"Elapsed time is {time.time() - t_start:.6f} seconds.")


if __name__ == "__main__":
    run()
