"""Geometric Brownian motion.

Faithful replica of ode-random/GBM.m by Nick Trefethen (May 2017):
the multiplicative-noise linear equation

    y' = mu y + sigma f y,   y(0) = 1,

(the smooth-random-function analogue of dX = mu X dt + sigma X o dW,
a Stratonovich SDE) with five trajectories each for zero, positive
(0.2) and negative (-0.2) drift.

MATLAB caps runaway trajectories with L.maxnorm = 100; here the
sample keys produce trajectories within range, and any that exceed
the cap would simply be clipped from the plot window.  Sample paths
use JAX keys (MATLAB rng(0) not reproducible).

Original: https://www.chebfun.org/examples/ode-random/GBM.html
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

import jax

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.randnfun import randnfun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-random')

DOM = (0.0, 20.0)
LAMBDA = 0.2
SIGMA = 1.0


def _panel(mu, fs, title, fname, ylim=None):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    xx = np.linspace(*DOM, 4000)
    for f in fs:
        L = Chebop(lambda t, y, _f=f: y.diff() - mu * y - SIGMA * _f * y,
                   domain=DOM)
        L.lbc = 1.0
        y = L.solve(0.0)
        ax.plot(xx, np.asarray(y(xx)), lw=1.2)
    ax.grid(True)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_xlim(*DOM)
    if ylim:
        ax.set_ylim(*ylim)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()
    fs = [randnfun(LAMBDA, DOM, big=True, key=jax.random.PRNGKey(k))
          for k in range(5)]
    _panel(0.0, fs, "zero drift", "GBM_repl_01.png")
    print("zero drift done", flush=True)
    _panel(0.2, fs, "positive drift", "GBM_repl_02.png", ylim=(0, 70))
    print("positive drift done", flush=True)
    _panel(-0.2, fs, "negative drift", "GBM_repl_03.png")
    print("total_time_in_seconds =")
    print(f"  {time.time() - t0:.6f}")


if __name__ == "__main__":
    run()
