"""Boundary layers and matched asymptotics.

Faithful replica of ode-linear/MatchedAsymp.m by Nick Trefethen
(November 2010): the singularly perturbed BVP

    -eps u'' + (2 - x^2) u = 1,   u(-1) = u(1) = 0,

compared against the matched-asymptotics model
1/(2-x^2) - exp((x-1)/sqrt(eps)) - exp((-x-1)/sqrt(eps)).

Original: https://www.chebfun.org/examples/ode-linear/MatchedAsymp.html
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
        _IMG, f"MatchedAsymp_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _solve(eps):
    N = Chebop(lambda x, u: -eps * u.diff(2) + (2 - x**2) * u,
               domain=(-1, 1))
    N.lbc = 0
    N.rbc = 0
    return N.solve(1.0)


def _model(t, eps):
    return (1.0 / (2 - t**2) - np.exp((t - 1) / np.sqrt(eps))
            - np.exp((-t - 1) / np.sqrt(eps)))


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t = np.linspace(-1, 1, 2400)

    t0 = time.time()
    ys = []
    fig, axs = plt.subplots(2, 2, figsize=(10.0, 6.6))
    for j, ax in enumerate(axs.ravel(), start=1):
        ep = 10.0**(-j)
        yep = _solve(ep)
        ys.append(yep)
        ax.plot(t, np.asarray(yep(t)), lw=1.6)
        ax.grid(True)
        ax.axis([-1.05, 1.05, 0, 1])
        ax.set_title(f"eps = {ep:4.1e}     npts = {len(yep)}",
                     fontsize=8)
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")
    _save(fig)

    fig, axs = plt.subplots(2, 2, figsize=(10.0, 6.6))
    for j, ax in enumerate(axs.ravel(), start=1):
        ep = 10.0**(-j)
        ax.plot(t, np.asarray(ys[j - 1](t)), lw=1.6)
        ax.plot(t, _model(t, ep), '--r', lw=1.6)
        ax.grid(True)
        ax.axis([-1.05, 1.05, 0, 1])
    _save(fig)

    fig, axs = plt.subplots(2, 2, figsize=(10.0, 6.6))
    for j, ax in enumerate(axs.ravel(), start=1):
        ep = 10.0**(-j)
        dv = _model(t, ep) - np.asarray(ys[j - 1](t))
        ax.plot(t, dv, 'm', lw=1.6)
        ax.grid(True)
        ax.set_xlim(-1.05, 1.05)
        err = float(np.max(np.abs(dv)))
        ax.set_title(f"eps = {ep:4.1e}     err = {err:5.2e}",
                     fontsize=8)
    _save(fig)


if __name__ == "__main__":
    run()
