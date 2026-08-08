"""Pitchfork bifurcation triggered by noise.

Faithful replica of ode-random/Pitchfork.m by Nick Trefethen (May
2017): the slowly-swept second-order ODE

    y'' = 2 c(t) y - 4 y^3 + 0.003 f(t),  c(t) = -1 + t/300,

on [0, 600] with y(0) = y'(0) = 0.  Without noise the solution rides
the unstable branch y = 0 forever; with small smooth random noise it
deviates at random onto one of the pitchfork branches +-sqrt(c/2).
A 0.2 y' damping term then suppresses the big oscillations.

Sample paths use JAX keys (MATLAB rng(0) not reproducible); keys are
chosen so the two noisy trajectories take opposite branches, as in
the published figure (the original also flipped a sign for this).

Original: https://www.chebfun.org/examples/ode-random/Pitchfork.html
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

DOM = (0.0, 600.0)
LAMBDA = 2.0


def _plot(ys, styles, title, fname):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    xx = np.linspace(*DOM, 8000)
    for y, st in zip(ys, styles):
        ax.plot(xx, np.asarray(y(xx)), st, lw=2.5)
    ax.set_xlabel("t", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.set_title(title, fontsize=18)
    ax.set_xlim(*DOM)
    ax.set_ylim(-0.8, 0.8)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()

    N = Chebop(lambda t, y: (y.diff(2) - 2 * (-1 + t / 300) * y
                             + 4 * y**3), domain=DOM)
    N.lbc = [0.0, 0.0]
    y1 = N.solve(0.0)
    f1 = 0.003 * randnfun(LAMBDA, DOM, big=True,
                          key=jax.random.PRNGKey(1))
    y2 = N.solve(f1)
    # Both samples happened to pick the + branch; flip one sign so the
    # figure shows both branches (the original did the same for its
    # damped panel: "we flipped the sign on one of them").
    f2 = -0.003 * randnfun(LAMBDA, DOM, big=True,
                           key=jax.random.PRNGKey(3))
    y3 = N.solve(f2)
    print("branches:", float(y2(600.0)), float(y3(600.0)), flush=True)
    _plot([y1, y2, y3], ["--k", "b", "r"], "Pitchfork",
          "Pitchfork_repl_01.png")

    # With damping.
    N = Chebop(lambda t, y: (y.diff(2) - 2 * (-1 + t / 300) * y
                             + 4 * y**3 + 0.2 * y.diff()), domain=DOM)
    N.lbc = [0.0, 0.0]
    y2 = N.solve(f1)
    y3 = N.solve(f2)
    print("damped branches:", float(y2(600.0)), float(y3(600.0)),
          flush=True)
    _plot([y1, y2, y3], ["--k", "b", "r"], "Pitchfork with damping",
          "Pitchfork_repl_02.png")

    print("total_time_in_seconds =")
    print(f"  {time.time() - t0:.6f}")


if __name__ == "__main__":
    run()
