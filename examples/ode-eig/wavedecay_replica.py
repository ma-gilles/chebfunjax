"""Wave equation with decay band.

Faithful replica of ode-eig/WaveDecay.m by Nick Trefethen (November
2010): eigenmodes 1, 2, 20, 40 of u'' on [-pi/2, pi/2] with Dirichlet
conditions, then the same for the operator with a decay band,

    L u = u'' + (2/a) 1_{|x|<=a} u',    a = 0.2,

whose indicator coefficient makes the discretization piecewise.

Original: https://www.chebfun.org/examples/ode-eig/WaveDecay.html
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

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-eig')


def _panel_plot(fname, modes, band=None):
    """Four stacked panels in the style of the MATLAB original."""
    fig, axes = plt.subplots(4, 1, figsize=(7.5, 6.0))
    xx = np.linspace(-np.pi / 2, np.pi / 2, 2000)
    for ax, (nmode, lam, v) in zip(axes, modes):
        if band is not None:
            ax.fill([-band, band, band, -band], [-1.6, -1.6, 2.2, 2.2],
                    color=(1, .8, .8), zorder=0)
        vv = np.asarray(v(xx)).real
        vv = vv / np.max(np.abs(vv))
        ax.plot(xx, vv)
        ax.set_xlim(-np.pi / 2, np.pi / 2)
        ax.set_ylim(-1.6, 2.2)
        ax.text(.3, 1.6, f"mode {nmode}         lam = {lam:6.3f}",
                fontsize=12)
    for ax in axes[:-1]:
        ax.set_xticks([])
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    nn = [1, 2, 20, 40]
    nmax = max(nn)

    # Pure wave equation: eigenmodes of u'' with Dirichlet conditions.
    L = Chebop(lambda u: u.diff(2), domain=(-np.pi / 2, np.pi / 2))
    L.bc = "dirichlet"
    lam, V = L.eigs(k=nmax, n=256, return_eigenfunctions=True)
    lam = np.asarray(lam).real
    idx = np.argsort(-lam)                      # sort descending
    lam, V = lam[idx], [V[i] for i in idx]
    _panel_plot("WaveDecay_repl_01.png",
                [(n, lam[n - 1], V[n - 1]) for n in nn])
    print("modes:", [f"{lam[n-1]:.3f}" for n in nn])

    # Wave equation with a decay band.
    a = 0.2
    x = chebfun(lambda t: t, domain=(-np.pi / 2, np.pi / 2))
    middle = (abs(x) <= a)
    L = Chebop(lambda x_, u: u.diff(2) + (2 / a) * middle * u.diff(),
               domain=(-np.pi / 2, np.pi / 2))
    L.bc = "dirichlet"
    lam, V = L.eigs(k=nmax, return_eigenfunctions=True)
    lam = np.asarray(lam).real
    idx = np.argsort(-lam)
    lam, V = lam[idx], [V[i] for i in idx]
    _panel_plot("WaveDecay_repl_02.png",
                [(n, lam[n - 1], V[n - 1]) for n in nn], band=a)
    print("decay modes:", [f"{lam[n-1]:.3f}" for n in nn])


if __name__ == "__main__":
    run()
