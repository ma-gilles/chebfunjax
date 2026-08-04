"""Green's functions and jump conditions.

Faithful replica of ode-linear/JumpGreen.m by Nick Trefethen
(May 2015): interior jump conditions imposed through the general .bc
field — Green's functions of the advection-diffusion operator
eta*u'' + u', a prescribed function-value jump, and a sweep of
Green's functions g(x; s) for s = 0.1, ..., 0.9.

Original: https://www.chebfun.org/examples/ode-linear/JumpGreen.html
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

from chebfunjax.chebfun1d.chebfun import jump
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

FIG = [0]
ETA = 0.2


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"JumpGreen_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _op():
    L = Chebop(lambda x, u: ETA * u.diff(2) + u.diff(), domain=(0, 1))
    L.lbc = 0
    L.rbc = 0
    return L


def _plot(ax, v, **kw):
    t = np.linspace(0, 1, 1000)
    ax.plot(t, np.asarray(v(t)), lw=1.6, **kw)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Green's function: derivative jump -eta at s = 1/2
    L = _op()
    L.bc = lambda x, u: [jump(u, 0.5),
                         jump(u.diff(), 0.5) + ETA]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    _plot(ax, L.solve(0.0))
    ax.grid(True)
    _save(fig)

    # One-sided values: u(0.7-) = 2, u(0.7+) = 1
    L = _op()
    L.bc = lambda x, u: [u(0.7, "left") - 2, u(0.7, "right") - 1]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    _plot(ax, L.solve(0.0))
    ax.grid(True)
    _save(fig)

    # Function-value jump of 1 at 0.2
    L = _op()
    L.bc = lambda x, u: [jump(u, 0.2) - 1, jump(u.diff(), 0.2)]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    _plot(ax, L.solve(0.0))
    ax.grid(True)
    _save(fig)

    # Green's functions at s = 0.75, 0.5, 0.25
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for s in (0.75, 0.5, 0.25):
        L = _op()
        L.bc = (lambda x, u, _s=s:
                [jump(u, _s), jump(u.diff(), _s) + ETA])
        _plot(ax, L.solve(0.0))
    ax.grid(True)
    _save(fig)

    # Sweep of Green's functions s = 0.1..0.9
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for s in np.arange(0.1, 0.95, 0.1):
        L = _op()
        L.bc = (lambda x, u, _s=float(s):
                [jump(u, _s), jump(u.diff(), _s) + ETA])
        _plot(ax, L.solve(0.0))
    ax.grid(True)
    _save(fig)


if __name__ == "__main__":
    run()
