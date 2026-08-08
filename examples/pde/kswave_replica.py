"""Traveling waves of the KS and generalized KS equations.

Faithful replica of pde/KSWave.m by Nick Trefethen: stable and
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

from chebfunjax.plotting import chebfun_style
from chebfunjax.spin.solver import spin
from chebfunjax.spin.spinop import SpinOp
from chebfunjax.utils.randnfun import randnfun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'pde')

NPTS, DT = 256, 0.02
FIG = [0]


def _run(dom, u0_vals_fn, lin):
    op = SpinOp(lin_coeff=lin, nonlin_vals=lambda u: -0.5 * u**2,
                nonlin_diff_order=1, domain=dom, tspan=(0.0, 100.0),
                u0=u0_vals_fn)
    return spin(op, NPTS, DT, dealias=False)


def _crest_gaps(x, u, X, red=False):
    """Distances between successive local maxima of the trig interpolant."""
    N = len(u)
    up = np.fft.irfft(np.fft.rfft(u), 16 * N) * 16
    xp = np.linspace(x[0], x[0] + (x[-1] - x[0]) * N / (N - 1), 16 * N,
                     endpoint=False)
    i = np.where((up[1:-1] > up[:-2]) & (up[1:-1] >= up[2:])
                 & (up[1:-1] > 0.5))[0] + 1
    # merge plateau duplicates
    pos = []
    for j in i:
        if not pos or xp[j] - pos[-1] > 1.0:
            pos.append(xp[j])
    d = np.diff(pos)
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.plot([0, len(d) - 1], [X, X], 'k', lw=0.7)
    ax.plot(range(1, len(d) - 1), d[1:-1], '.',
            markersize=14, color=('r' if red else (0, 0, 0.6)))
    ax.set_xticks([])
    ax.grid(True)
    ax.set_ylim(0, 15)
    ax.set_xlim(0, len(d) - 1)
    ax.set_title("distances between successive wave crests")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"KSWave_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"crest gaps: mean {np.mean(d[1:-1]):.3f} "
          f"std {np.std(d[1:-1]):.3f} (X = {X})", flush=True)


def _experiment(X, lin_fn, key, red_final=False):
    dom = (0.0, 20.0 * X)

    def u0(x):
        return 2 * np.exp(np.sin(2 * np.pi * x / X))

    x, _, u1 = _run(dom, u0, lin_fn)
    pert = 0.1 * np.asarray(
        randnfun(2.0, dom, key=jax.random.PRNGKey(key))(x))
    up = u1 + pert
    x, _, u2 = _run(dom, lambda xx: up, lin_fn)

    FIG[0] += 1
    fig, axes = plt.subplots(4, 1, figsize=(9.4, 8.0))
    for ax, (vals, ttl, col) in zip(axes, [
            (u0(x), f"initial condition        X = {X}", 'k'),
            (u1, "after 100 time units", 'C0'),
            (up, "perturbation", 'k'),
            (u2, "after 100 more time units",
             'r' if red_final else 'C0')]):
        ax.plot(x, vals, color=col, lw=2)
        ax.set_ylim(-3, 9)
        ax.grid(True)
        ax.set_xlim(*dom)
        ax.text(5, 6.6, ttl, fontsize=14)
        ax.set_yticks([0, 5])
    for ax in axes[:-1]:
        ax.set_xticks([])
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"KSWave_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    return x, u2


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    ks = lambda xi: xi**2 - xi**4                      # noqa: E731

    # 1. KS: X = 8 stable...
    x, u = _experiment(8, ks, 80)
    _crest_gaps(x, u, 8)

    # ... X = 7 unstable.
    x, u = _experiment(7, ks, 70, red_final=True)
    _crest_gaps(x, u, 7, red=True)

    # 2. Generalized KS: delta = 0.8, eps = 0.6.
    delta, ep = 0.8, 0.6
    gks = lambda xi: delta * (xi**2 - xi**4) + ep * 1j * xi**3  # noqa: E731

    x, u = _experiment(10, gks, 100)
    _crest_gaps(x, u, 10)

    x, u = _experiment(11, gks, 110, red_final=True)
    _crest_gaps(x, u, 11, red=True)


if __name__ == "__main__":
    run()
