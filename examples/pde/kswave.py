"""Traveling waves of the KS and generalized KS equations.

Translation of pde/KSWave.m by Nick Trefethen: stable and
unstable periodic traveling waves of the Kuramoto-Sivashinsky
equation (X = 8 stable, X = 7 unstable under perturbation) and of the
generalized KS equation with delta = 0.8, eps = 0.6 (X = 10 stable,
X = 11 unstable), each shown as a 4-panel evolution plus the
distances between successive wave crests.

Perturbations use JAX randnfun keys (MATLAB rng not reproducible).

Original: https://www.chebfun.org/examples/pde/KSWave.html
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

import jax

import chebfunjax as cj
from chebfunjax.operators.spinop import Spinop, spin
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.randnfun import randnfun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'pde')

NPTS, DT = 256, 0.02
FIG = [0]
LW = 4 * 0.6


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout(h_pad=0.3)
    _savefig(fig, os.path.join(_IMG, f"KSWave_{FIG[0]:02d}.png"),
             size=(600, 269))
    plt.close(fig)


def _num(v):
    return f"{int(v)}" if float(v) == int(v) else f"{v:.15e}"


def _disp_spinop(S):
    """MATLAB display of a spinop object."""
    print("S = ")
    print("  spinop with properties:")
    print()
    a, b = S.domain
    print(f"     domain: [{_num(a)} {_num(b)}]")
    print("       init: [Inf×1 chebfun]")
    print(f"        lin: {S.lin_str}")
    print(f"     nonlin: {S.nonlin_str}")
    print(f"      tspan: [{_num(S.tspan[0])} {_num(S.tspan[1])}]")
    print("    numVars: 1")


def _panel(ax, f, color, text, X=None, ty=6.6, yt=(0, 5), xt=False):
    matlab_plot(f, color, ax=ax, linewidth=LW)
    ax.set_ylim(-3, 9)
    ax.grid(True)
    ax.text(5, ty, text, fontsize=26 * 0.35)
    if X is not None:
        ax.text(10 * X, ty, f"X = {X}", fontsize=26 * 0.35)
    if not xt:
        ax.set_xticks([])
    if yt is not None:
        ax.set_yticks(list(yt))


def _experiment(S, X, key, red_final=False, ty=6.6, pert_yt=(0, 5)):
    """The four-panel cell: initial condition, spin, perturb, spin."""
    S.domain = (0.0, 20.0 * X)
    S.init = cj.chebfun(lambda x: 2 * np.exp(np.sin(2 * np.pi * x / X)),
                        domain=list(S.domain), trig=True)
    fig, axes = plt.subplots(4, 1)
    _panel(axes[0], S.init, 'k', 'initial condition', X, ty)
    u = spin(S, NPTS, DT, 'plot', 'off', dealias=False)
    _panel(axes[1], u, 'C0', 'after 100 time units', ty=ty)
    S.init = u + .1 * randnfun(2.0, S.domain, key=jax.random.PRNGKey(key))
    _panel(axes[2], S.init, 'k', 'perturbation', ty=ty, yt=pert_yt)
    u = spin(S, NPTS, DT, 'plot', 'off', dealias=False)
    _panel(axes[3], u, 'r' if red_final else 'C0',
           'after 100 more time units', ty=ty, xt=True)
    _save(fig)
    return u


def _crests(u, X, red=False):
    """[a,b] = max(u,'local'); d = diff(b)'; and its plot."""
    b, _ = u.max('local')
    d = np.diff(np.asarray(b))
    fig, ax = plt.subplots()
    ax.plot([0, len(d) - 1], [X, X], 'k', lw=.7)
    ax.plot(np.arange(1, len(d) - 1), d[1:-1], '.', markersize=32 * 0.35,
            color=('r' if red else (0, 0, .6)))
    ax.set_xticks([])
    ax.grid(True)
    ax.axis([0, len(d) - 1, 0, 15])
    ax.set_title('distances between successive wave crests')
    _save(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    S = Spinop('ks')
    _disp_spinop(S)

    S.tspan = (0.0, 100.0)
    u = _experiment(S, 8, 80)
    _crests(u, 8)

    u = _experiment(S, 7, 70, red_final=True)
    _crests(u, 7, red=True)

    # S.lin = @(u) delta*(-diff(u,2)-diff(u,4)) - ep*diff(u,3)
    delta, ep = 0.8, 0.6
    S.lin_symbol = lambda om: delta * (om**2 - om**4) + ep * 1j * om**3

    u = _experiment(S, 10, 100, ty=7.2, pert_yt=None)
    _crests(u, 10)

    u = _experiment(S, 11, 110, red_final=True, ty=7.2)
    _crests(u, 11, red=True)


if __name__ == "__main__":
    run()
