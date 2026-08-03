"""Stability regions of ODE formulas.

Faithful replica of ode-linear/Regions.m by Nick Trefethen
(February 2011): stability region boundaries of Adams-Bashforth,
Runge-Kutta, and backward differentiation formulas traced with
complex chebfun arithmetic on the unit circle z = exp(it).

Original: https://www.chebfun.org/examples/ode-linear/Regions.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

FIG = [0]
COLORS = ['b', 'r', 'g', 'm', 'y', 'c']


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Regions_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_curve(ax, w, color):
    t = np.linspace(0, 2 * np.pi, 1200)
    zv = np.asarray(w(jnp.asarray(t)))
    ax.plot(zv.real, zv.imag, color=color, lw=1.5)


def _axes(ax, lim):
    ax.plot([lim[0], lim[1]], [0, 0], 'k', lw=1)
    ax.plot([0, 0], [lim[2], lim[3]], 'k', lw=1)
    ax.axis(lim)
    ax.set_aspect("equal")
    ax.grid(True)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    t = cj.chebfun(lambda s: s, domain=(0, 2 * np.pi))
    z = (1j * t).exp()
    r = z - 1

    # Adams-Bashforth 1-3
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    ax.set_title("Adams-Bashforth orders 1,2,3", fontsize=12)
    _plot_curve(ax, r, COLORS[0])
    s2 = (3 - 1 / z) / 2
    _plot_curve(ax, r / s2, COLORS[1])
    s3 = (23 - 16 / z + 5 / z**2) / 12
    _plot_curve(ax, r / s3, COLORS[2])
    _axes(ax, [-2.5, 0.5, -1.5, 1.5])
    _save(fig)

    # Runge-Kutta 1-4 (Newton iterations on chebfuns)
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    ax.set_title("Runge-Kutta orders 1,2,3,4", fontsize=12)
    w = z - 1
    _plot_curve(ax, w, COLORS[0])
    for _ in range(3):
        w = w - (1 + w + 0.5 * w**2 - z**2) / (1 + w)
    _plot_curve(ax, w, COLORS[1])
    for _ in range(4):
        w = w - (1 + w + 0.5 * w**2 + w**3 / 6 - z**3) \
            / (1 + w + w**2 / 2)
    _plot_curve(ax, w, COLORS[2])
    for _ in range(4):
        w = w - (1 + w + 0.5 * w**2 + w**3 / 6 + w**4 / 24 - z**4) \
            / (1 + w + w**2 / 2 + w**3 / 6)
    _plot_curve(ax, w, COLORS[3])
    _axes(ax, [-5, 2, -3.5, 3.5])
    _save(fig)

    # Backward differentiation 1-6
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    ax.set_title("Backward differentiation orders 1-6 "
                 "(exteriors of curves)", fontsize=12)
    d = 1 - 1 / z
    rr = 0
    for i in range(1, 7):
        rr = rr + d**i / i
        _plot_curve(ax, rr, COLORS[i - 1])
    _axes(ax, [-15, 35, -25, 25])
    _save(fig)

    # Close-up
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    ax.set_title("Backward differentiation close-up", fontsize=12)
    d = 1 - 1 / z
    rr = 0
    for i in range(1, 7):
        rr = rr + d**i / i
        _plot_curve(ax, rr, COLORS[i - 1])
    _axes(ax, [-6, 6, -6, 6])
    _save(fig)


if __name__ == "__main__":
    run()
