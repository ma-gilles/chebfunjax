"""Eigenstates of the Schroedinger equation.

Faithful replica of ode-eig/Eigenstates.m by Nick Trefethen (January
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
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-eig')

FIG = [0]


def _draw(V, evals, efuns, title=""):
    """MATLAB quantumstates-style plot: V plus states at their levels."""
    FIG[0] += 1
    a, b = float(V.domain.a), float(V.domain.b)
    xx = np.linspace(a, b, 2000)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(xx, np.asarray(V(xx)), "k", lw=1.6)
    lam = np.asarray(evals, dtype=float)
    gap = np.min(np.diff(lam)) if len(lam) > 1 else 1.0
    sc = 0.4 * max(gap, 1e-8)
    for lv, f in zip(lam, efuns):
        vals = np.asarray(f(xx))
        vals = vals / max(np.max(np.abs(vals)), 1e-300)
        ax.plot(xx, lv + sc * vals, lw=1.0)
        ax.plot([a, b], [lv, lv], color="0.8", lw=0.5, zorder=0)
    ax.set_xlim(a, b)
    lo = min(0.0, float(np.min(np.asarray(V(xx)))))
    ax.set_ylim(lo - 0.05 * abs(lam[-1]), lam[-1] + 6 * sc)
    ax.grid(True)
    if title:
        ax.set_title(title)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Eigenstates_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t_start = time.time()
    x = chebfun(lambda x: x, domain=(-3.0, 3.0))

    # harmonic oscillator, default 10 states, h = 0.1
    V = x**2
    lam, funs = quantumstates(V)
    _draw(V, lam, funs, "harmonic oscillator")
    print("ans =")
    for v in np.asarray(lam):
        print(f"   {v:.15f}")

    # more states
    lam, funs = quantumstates(V, n=60)
    _draw(V, lam, funs, "60 states")

    # smaller h
    lam, funs = quantumstates(V, h=0.01)
    _draw(V, lam, funs, "h = 0.01")

    # both
    lam, funs = quantumstates(V, n=20, h=0.5)
    _draw(V, lam, funs, "n = 20, h = 0.5")

    # deep square well
    V = 10 - 10 * (abs(x) < 1)
    lam, funs = quantumstates(V)
    _draw(V, lam, funs, "deep square well")

    # shallower square well
    V = 1 - 1 * (abs(x) < 1)
    lam, funs = quantumstates(V, n=20)
    _draw(V, lam, funs, "shallow square well")

    # absolute value
    lam, funs = quantumstates(abs(x))
    _draw(abs(x), lam, funs, "V = |x|")

    # square root
    V = (abs(x) + 0.1).sqrt()
    lam, funs = quantumstates(V)
    _draw(V, lam, funs, "V = sqrt(|x| + 0.1)")

    # off-centre barrier
    V = 0.5 * (abs(x - 0.5) < 0.5)
    lam, funs = quantumstates(V, n=18)
    _draw(V, lam, funs, "off-centre barrier")

    print(f"Elapsed time is {time.time() - t_start:.6f} seconds.")


if __name__ == "__main__":
    run()
