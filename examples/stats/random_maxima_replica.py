"""How many local maxima does a random function have?

Faithful replica of stats/RandomMaxima.m by Nick Trefethen
(March 2017): counting local maxima of band-limited random functions
— the count grows linearly with the interval length.

randn draws are not bit-reproducible vs MATLAB; the counts are our
own draws with the same statistics.

Original: https://www.chebfun.org/examples/stats/RandomMaxima.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.randnfun import randnfun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')

FIG = [0]


def _local_maxima(f):
    """Positions/values of local maxima (interior maxima of f)."""
    df = f.diff()
    r = np.asarray(df.roots(nojump=True))
    d2 = df.diff()
    keep = np.asarray(d2(r)) < 0
    pos = r[keep]
    val = np.asarray(f(pos))
    return val, pos


def _panel(f, dom, ms=10, lw=1.6):
    FIG[0] += 1
    val, pos = _local_maxima(f)
    xs = np.linspace(*dom, 3000)
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.plot(xs, np.asarray(f(xs)), 'k', lw=lw)
    ax.grid(True)
    ax.plot(pos, val, '.r', ms=ms)
    ax.set_title(f"{len(val)} maxima", fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"RandomMaxima_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)
    return len(val)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()

    dx = 1.0
    f = randnfun(dx, domain=(0.0, 20.0), key=jax.random.PRNGKey(0))
    _panel(f, (0, 20))

    f = randnfun(dx, domain=(0.0, 40.0), key=jax.random.PRNGKey(1))
    _panel(f, (0, 40))

    Lvec = 2.0 ** np.arange(0, 11)
    nmax = []
    for i, L in enumerate(Lvec):
        f = randnfun(dx, domain=(0.0, float(L)),
                     key=jax.random.PRNGKey(10 + i))
        val, _ = _local_maxima(f)
        nmax.append(len(val))
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.loglog(Lvec, Lvec, '-r', lw=2)
    ax.loglog(Lvec, np.maximum(nmax, 0.8), '.', ms=16)
    ax.grid(True)
    ax.axis([0.8, 1300, 0.8, 1300])
    ax.set_xlabel("length of interval", fontsize=13)
    ax.set_ylabel("no. of maxima", fontsize=13)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"RandomMaxima_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("counts:", nmax)
    print("time_in_seconds =")
    print(f"    {time.time()-t0:.4f}")


if __name__ == "__main__":
    run()
