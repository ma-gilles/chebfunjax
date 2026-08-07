"""Orr-Sommerfeld eigenvalues.

Faithful replica of ode-eig/OrrSommerfeld.m by Toby Driscoll and Nick
Trefethen (October 2010): the rightmost 50 eigenvalues of the
Orr-Sommerfeld operator (fourth-order, complex, generalized) for
Re = 2000 and for the critical Re = 5772.22.

Note the original defines B = u'' - alph^2 u ONCE with alph = 1 and
does not update it when alph changes to 1.02 for the critical case;
this replica reproduces that exactly (with a matched-alpha B, the
rightmost eigenvalue sits at ~1e-7 instead of the published +5e-5).

Original: https://www.chebfun.org/examples/ode-eig/OrrSommerfeld.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-eig')


def _os_chebop(Re, alph):
    A = Chebop(lambda x, u:
               (u.diff(4) - 2 * alph**2 * u.diff(2) + alph**4 * u) / Re
               - 2j * alph * u
               - 1j * alph * (1 - x**2) * (u.diff(2) - alph**2 * u),
               domain=(-1, 1))
    A.lbc = [0, 0]
    A.rbc = [0, 0]
    return A


def _plot(e, title, fname):
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot(e.real, e.imag, 'r.', markersize=14, linestyle='none')
    ax.grid(True)
    ax.set_xlim(-0.9, 0.1)
    ax.set_ylim(-1, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # B is defined ONCE with alph = 1, exactly as in the original.
    alph0 = 1.0
    B = Chebop(lambda x, u: u.diff(2) - alph0**2 * u, domain=(-1, 1))

    Re, alph = 2000.0, 1.0
    A = _os_chebop(Re, alph)
    _, e = A.eigs_generalized(B, k=50, n=140, sort="LR")
    e = np.asarray(e)
    maxe = float(np.max(e.real))
    print(f"Re = 2000: lambda_r = {maxe:.7f}")
    _plot(e, f"Re = {Re:8.2f}   $\\lambda_r$ = {maxe:7.5f}",
          "OrrSommerfeld_repl_01.png")

    Re, alph = 5772.22, 1.02
    A = _os_chebop(Re, alph)
    _, e = A.eigs_generalized(B, k=50, n=140, sort="LR")
    e = np.asarray(e)
    maxe = float(np.max(e.real))
    print(f"Re = 5772.22: lambda_r = {maxe:.7f}")
    _plot(e, f"Re = {Re:5.0f},   $\\lambda_r$ = {maxe:7.5f}",
          "OrrSommerfeld_repl_02.png")


if __name__ == "__main__":
    run()
