"""Near-nonuniqueness in linear BVPs.

Faithful replica of ode-linear/NearNonuniqueness.m by Nick Trefethen
(December 2016): the problem

    0.01 u'' - x u' + u = 1,  u(-1) = u(1) = 0

is exponentially close to having a nontrivial null function, so the
computed "solution" is polluted by an odd component; eigs exposes the
near-zero eigenvalue and its null function.  The dual problem with
+xu' shows exponentially large solutions instead.

Original: https://www.chebfun.org/examples/ode-linear/NearNonuniqueness.html
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

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"NearNonuniqueness_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _solve(eps, sign=-1.0):
    L = Chebop(lambda x, u: eps * u.diff(2) + sign * x * u.diff() + u,
               domain=(-1, 1))
    L.lbc = 0
    L.rbc = 0
    return L, L.solve(1.0)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t = np.linspace(-1, 1, 1000)
    xf = Chebfun.identity(Domain((-1.0, 1.0)))

    L, u = _solve(0.01)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(t, np.asarray(u(t)), lw=1.6)
    ax.set_xticks([-1, 0, 1])
    ax.set_title("This function should be even", fontsize=12)
    ax.grid(True)
    _save(fig)
    resid = (0.01 * u.diff(2) - xf * u.diff() + u - 1).norm()
    print("residual_norm =")
    print(f"     {float(resid):.15e}")

    fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.4))
    for eps, ax in zip((0.005, 0.001), axs):
        _, ue = _solve(eps)
        ax.plot(t, np.asarray(ue(t)), lw=1.6)
        ax.set_xticks([-1, 0, 1])
        ax.grid(True)
    _save(fig)

    lam = np.sort(np.real(np.asarray(L.eigs(k=6))))
    print("ans =")
    for v in lam:
        print(f"  {v:19.15f}")

    lam1, funs = L.eigs(k=1, sigma="SM", return_eigenfunctions=True)
    v = funs[0]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(t, np.asarray(v(t)), 'm', lw=1.6)
    ax.set_xticks([-1, 0, 1])
    ax.set_title("null function", fontsize=12)
    ax.grid(True)
    _save(fig)

    for c in ([0.01, 1, 1], [0.01, -1, 1]):
        print("ans =")
        for r in np.roots(c):
            print(f"  {r:19.15f}")

    # Dual problem: +x u'
    _, u = _solve(0.1, sign=+1.0)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(t, np.asarray(u(t)), lw=1.6)
    ax.set_xticks([-1, 0, 1])
    ax.set_ylim(-200, 0)
    ax.set_yticks(np.arange(-200, 1, 100))
    ax.set_title("Solution to the dual problem", fontsize=12)
    ax.grid(True)
    _save(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.4))
    for eps, yl, ax in zip((0.05, 0.025), ((-3e4, 0), (-6e8, 0)), axs):
        _, ue = _solve(eps, sign=+1.0)
        ax.plot(t, np.asarray(ue(t)), lw=1.6)
        ax.set_xticks([-1, 0, 1])
        ax.set_ylim(*yl)
        ax.grid(True)
    _save(fig)

    for c in ([0.01, -1, 1], [0.01, 1, 1]):
        print("ans =")
        for r in np.roots(c):
            print(f"  {r:19.15f}")


if __name__ == "__main__":
    run()
