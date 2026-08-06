"""A nonlinear system of Guckenheimer and Holmes.

Faithful replica of ode-nonlin/GuckenheimerHolmes.m by Nick Trefethen
(February 2015): the three-variable system

    u' = u (1 - u^2 - b v^2 - c w^2)
    v' = v (1 - v^2 - b w^2 - c u^2)
    w' = w (1 - w^2 - b u^2 - c v^2)

with b = 0.55, c = 1.5, whose trajectory approaches a heteroclinic
cycle between the saddles (1,0,0), (0,1,0), (0,0,1): the system lingers
ever longer near each saddle, so the crossing times grow geometrically.

Original: https://www.chebfun.org/examples/ode-nonlin/GuckenheimerHolmes.html
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

FIG = [0]
b, c = 0.55, 1.5


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"GuckenheimerHolmes_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _solve(tmax):
    N = Chebop(lambda t, u, v, w: [
        u.diff() - u * (1 - u**2 - b * v**2 - c * w**2),
        v.diff() - v * (1 - v**2 - b * w**2 - c * u**2),
        w.diff() - w * (1 - w**2 - b * u**2 - c * v**2)],
        domain=(0, tmax))
    N.lbc = lambda u, v, w: [u - 0.5, v - 0.49, w - 0.49]
    return N.solve(0.0)


def _upcrossings(f, fp):
    r = np.sort(np.asarray((f - 0.5).roots(), dtype=float))
    return r[np.asarray(fp(r)) > 0]


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # --- t in [0, 800] ------------------------------------------------
    u, v, w = _solve(800)
    t = np.linspace(0, 800, 20000)
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.plot(t, np.asarray(v(t)), lw=0.9)
    ax.set_ylim(-0.5, 1.5)
    ax.grid(True)
    _save(fig)

    fig = plt.figure(figsize=(7.0, 6.2))
    ax = fig.add_subplot(projection="3d")
    ax.plot(np.asarray(u(t)), np.asarray(v(t)), np.asarray(w(t)),
            "k", lw=0.7)
    ax.view_init(elev=10, azim=10 - 90)     # MATLAB view(10,10)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_zlabel("w")
    _save(fig)

    # --- t in [0, 2000] and the crossing times ------------------------
    u, v, w = _solve(2000)
    t = np.linspace(0, 2000, 40000)
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.plot(t, np.asarray(v(t)), lw=0.7)
    ax.set_ylim(-0.5, 1.5)
    ax.grid(True)
    _save(fig)

    tu = _upcrossings(u, u.diff())
    tv = _upcrossings(v, v.diff())
    tw = _upcrossings(w, w.diff())
    nu, nv, nw = len(tu), len(tv), len(tw)
    print(f"[nu nv nw] = [{nu} {nv} {nw}]")

    # MATLAB: semilogy(2/3+(2:nu), diff(tu), ...) -- ALL gaps, plotted
    # against 2..nu (the first gap lands at x = 2, off the axis).
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.semilogy(2 / 3 + np.arange(2, nu + 1), np.diff(tu), ".",
                color=(0.9, 0, 0), label="u")
    ax.semilogy(1 / 3 + np.arange(2, nv + 1), np.diff(tv), ".",
                color=(0, 0.7, 0), label="v")
    ax.semilogy(0 / 3 + np.arange(2, nw + 1), np.diff(tw), ".",
                color=(0, 0, 1), label="w")
    ax.set_xlabel("crossing number")
    ax.set_ylabel("time")
    ax.set_title("Crossing times")
    ax.grid(True)
    ax.axis([5, 28, 6, 600])
    ax.legend(loc="lower right")
    _save(fig)

    # The geometric growth rate of the gaps: the sharp quantitative
    # signature of the heteroclinic cycle.
    d = np.diff(tv)
    tail = d[max(0, len(d) - 6):]
    ratios = tail[1:] / tail[:-1]
    print("late gap ratios (v):", np.round(ratios, 4))


if __name__ == "__main__":
    run()
