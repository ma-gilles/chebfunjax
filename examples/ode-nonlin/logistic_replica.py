"""Logistic map and chaos.

Faithful replica of ode-nonlin/Logistic.m by Nick Trefethen (July 2013):
the logistic iteration x <- r x (1-x) carried out on a chebfun in the
parameter r, so that each step is a polynomial of twice the degree and
the period-doubling route to chaos appears as increasingly wild
oscillation near r = 4.

Original: https://www.chebfun.org/examples/ode-nonlin/Logistic.html
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

from chebfunjax import chebfun
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Logistic_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _panel(ax, x, n, textx):
    a, b = float(x.domain.a), float(x.domain.b)
    t = np.linspace(a, b, max(4000, 4 * len(x)))
    ax.plot(t, np.asarray(x(t)), lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlim(a, b)
    ax.set_ylabel(f"x({n})")
    ax.grid(True)
    ax.text(textx, 0.7, f"length(x) = {len(x)}", fontsize=9)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    r = chebfun(lambda r: r, domain=(0, 4))
    x = 0.5 + 0 * r

    # Steps 0-3, 4-7, 8-11: four panels each, on the full interval.
    for block in (range(0, 4), range(4, 8), range(8, 12)):
        fig, axes = plt.subplots(4, 1, figsize=(6.0, 4.2))
        for ax, n in zip(axes, block):
            _panel(ax, x, n, 0.2)
            print(f"n={n:2d}  length(x) = {len(x)}")
            x = r * x * (1 - x)
        _save(fig)

    # Zoom in on [3.5, 4] and continue: steps 12-15, then 16-18.
    r = r.restrict(3.5, 4)
    x = x.restrict(3.5, 4)
    fig, axes = plt.subplots(4, 1, figsize=(6.0, 4.2))
    for ax, n in zip(axes, range(12, 16)):
        _panel(ax, x, n, 3.52)
        print(f"n={n:2d}  length(x) = {len(x)}")
        x = r * x * (1 - x)
    _save(fig)

    fig, axes = plt.subplots(4, 1, figsize=(6.0, 4.2))
    for ax, n in zip(axes[:3], range(16, 19)):
        _panel(ax, x, n, 3.52)
        print(f"n={n:2d}  length(x) = {len(x)}")
        x = r * x * (1 - x)
    axes[3].set_axis_off()
    _save(fig)

    # The final iterate on its own, then zoomed into a small interval.
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    t = np.linspace(3.5, 4, 200000)
    ax.plot(t, np.asarray(x(t)), lw=0.4)
    ax.set_ylim(0, 1)
    ax.set_xlim(3.5, 4)
    ax.grid(True)
    _save(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    t = np.linspace(3.902, 3.908, 20000)
    ax.plot(t, np.asarray(x(t)), lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlim(3.902, 3.908)
    ax.grid(True)
    _save(fig)


if __name__ == "__main__":
    run()
