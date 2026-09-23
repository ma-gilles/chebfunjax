"""Double-well Schroedinger eigenstates.

Translation of ode-eig/DoubleWell.m by Nick Trefethen (November
2010): the first 12 eigenstates of

    -0.007 u'' + V(x) u = lam u,   u(-1) = u(1) = 0,

with V an indicator-function barrier (V = 1.5 on [-0.2, 0.3]), drawn
shifted up by their eigenvalues in the physicists' style, followed by a
`quantumstates` exploration of the potential max(|x|, 1-3|x|).

Original: https://www.chebfun.org/examples/ode-eig/DoubleWell.html
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
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-eig')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Sketch the potential.
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.plot([-1, -1, -.2, -.2, .3, .3, 1, 1],
            [3.3, 0, 0, 1.5, 1.5, 0, 0, 3.3], "k", lw=2)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-.05, 3.3)
    ax.set_axis_off()
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(_IMG, "DoubleWell_01.png"), size=(600, 270))

    # First 12 eigenvalues and eigenfunctions.
    t0 = time.time()
    x = chebfun(lambda t: t, domain=(-1.0, 1.0))
    V = 1.5 * (abs(x - 0.05) < 0.25)
    L = Chebop(lambda x_, u: -0.007 * u.diff(2) + V * u, domain=(-1, 1))
    L.bc = 0.0
    neigs = 12
    lam, EV = L.eigs(k=neigs, return_eigenfunctions=True)
    lam = np.asarray(lam)
    idx = np.argsort(lam)
    lam = lam[idx]
    EV = [EV[i] for i in idx]
    for v in lam:
        print(f"   {v:.15f}")
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")

    # Physicists' plot: eigenmodes shifted up by their eigenvalues.
    colors = [(1, 0, 0), (0, .8, 0), (.9, .9, 0),
              (0, 0, 1), (1, 0, 1), (0, .8, 1)]
    xx = np.linspace(-1, 1, 2000)
    for j in range(neigs):
        vv = np.asarray(EV[j](xx)) / 15.0
        if np.max(vv) < -np.min(vv):
            vv = -vv
        ax.plot(xx, lam[j] + vv, color=colors[j % 6], lw=1.6)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "DoubleWell_02.png"), size=(600, 270))
    plt.close(fig)

    # quantumstates exploration of max(|x|, 1-3|x|).
    x = chebfun(lambda t: t, domain=(-3.0, 3.0))
    print("x =")
    print(repr(x))
    absx = abs(x)
    V = absx.maximum(1 - 3 * absx)
    lam, funs = quantumstates(V)
    lam = np.asarray(lam)
    print("ans =")
    for v in lam:
        print(f"   {v:.15f}")

    # The quantumstates plot (@chebfun/quantumstates.m): V in black, each
    # eigenfunction scaled by dy and lifted by its eigenvalue.
    h, n = 0.1, len(lam)
    xmin, xmax = -3.0, 3.0
    xx = np.linspace(xmin, xmax, 2000)
    Vx = np.asarray(V(xx))
    ymin = float(V.min()[1])
    ymax = lam.max()
    ydiff = ymax - ymin
    ymax = ymax + .2 * ydiff
    dx = .05 * (xmax - xmin)
    dy = .25 * ydiff / max(5, n)
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    for j, f in enumerate(funs):
        w = dy * np.asarray(f(xx))
        if w.max() < -w.min():
            w = -w
        ax.plot(xx, w + lam[j], lw=1)
    ax.plot(xx, Vx, "k", lw=1)
    ax.axis([xmin - dx, xmax + dx, ymin - dy, ymax])
    ax.set_title(f"h = {h:4g}      {n} eigenstates", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "DoubleWell_03.png"), size=(600, 270))
    plt.close(fig)


if __name__ == "__main__":
    run()
