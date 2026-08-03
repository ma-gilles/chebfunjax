"""Extrema of complicated functions.

Faithful replica of opt/ExtremeExtrema.m by Nick Trefethen
(September 2010): global and local extrema of a highly oscillatory
function after abs and min compositions.

Original: https://www.chebfun.org/examples/opt/ExtremeExtrema.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'opt')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"ExtremeExtrema_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pw(ax, f, color, dom, n=2500):
    bps = [float(v) for v in f.domain.breakpoints]
    for a, b in zip(bps[:-1], bps[1:]):
        m = max(8, int(n * (b - a) / (dom[1] - dom[0])))
        t = np.linspace(a, b, m)
        ax.plot(t, np.asarray(f(t)), color, lw=0.9)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()

    f = cj.chebfun(lambda x: jnp.cos(x) * jnp.sin(jnp.exp(x)),
                   domain=(0.0, 6.0))
    print("ans =")
    print(f"   {len(f)}")
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    _plot_pw(ax, f, 'g', (0, 6))
    ax.set_title("A complicated function")
    _save(fig)

    g = f.abs()
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    _plot_pw(ax, g, 'm', (0, 6))
    ax.axis([0, 6, 0, 1])
    ax.set_title("Absolute value")
    _save(fig)

    x = cj.chebfun(lambda t: t, domain=(0.0, 6.0))
    h = g.minimum(x * (1 / 8))
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    _plot_pw(ax, h, 'b', (0, 6))
    ax.axis([0, 6, 0, 1])
    ax.set_title("Minimum with x/8")
    _save(fig)

    h05 = h.restrict(0.0, 5.0)
    maxpos, maxval = h05.max()
    print("maxval =")
    print(f"   {float(maxval):.15f}")
    print("maxpos =")
    print(f"   {float(maxpos):.15f}")
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    _plot_pw(ax, h, 'b', (0, 6))
    ax.axis([0, 6, 0, 1])
    ax.plot(float(maxpos), float(maxval), '.r', ms=14)
    ax.set_title("Global maximum")
    _save(fig)

    # local maxima from dense sampling + parabolic refinement (the
    # chebfun diff().roots() route compiles one XLA program per piece
    # length across ~50 pieces and exhausts LLVM section memory)
    ts = np.linspace(0, 6, 20001)
    hv = np.asarray(h(ts))
    im = np.where((hv[1:-1] > hv[:-2]) & (hv[1:-1] >= hv[2:]))[0] + 1
    lmax = []
    for i in im:
        a, b, c = hv[i - 1], hv[i], hv[i + 1]
        denom = a - 2 * b + c
        dt = 0.5 * (a - c) / denom if denom != 0 else 0.0
        lmax.append(ts[i] + dt * (ts[1] - ts[0]))
    lmax = np.asarray(lmax)
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    _plot_pw(ax, h, 'b', (0, 6))
    ax.axis([0, 6, 0, 1])
    ax.plot(lmax, np.asarray(h(lmax)), '.k', ms=8)
    ax.plot(float(maxpos), float(maxval), '.r', ms=14)
    ax.set_title("Local maxima")
    _save(fig)
    print("Total_time =")
    print(f"   {time.time()-t0:.6f}")


if __name__ == "__main__":
    run()
