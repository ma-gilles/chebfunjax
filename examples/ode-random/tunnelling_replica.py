"""Tunnelling.

Faithful replica of ode-random/Tunnelling.m by Nick Trefethen (May
2017): the bistable equation with additive smooth random noise,

    y' = y - y^3 + f,   y(0) = 0,   f = 0.45 randnfun(0.5, 'big'),

six trajectories settling near +-1, then a long [0, 800] trajectory
showing metastable switching ("tunnelling"), and the same with the
noise coefficient raised to 0.60 -- faster switching.

Trajectories are sample paths: MATLAB's rng(4) randn stream cannot be
reproduced in numpy/JAX, so these are different samples of the same
law (documented on the page).

Original: https://www.chebfun.org/examples/ode-random/Tunnelling.html
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

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.randnfun import randnfun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-random')

LAMBDA = 0.5


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Six trajectories on [0, 30].
    dom = (0.0, 30.0)
    N = Chebop(lambda t, y: y.diff() - y + y**3, domain=dom)
    N.lbc = 0.0
    xx = np.linspace(*dom, 2000)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for k in range(6):
        f = 0.45 * randnfun(LAMBDA, dom, big=True,
                            key=jax.random.PRNGKey(400 + k))
        y = N.solve(f)
        ax.plot(xx, np.asarray(y(xx)), lw=1.2)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title("Bistability")
    ax.set_ylim(-1.7, 1.7)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Tunnelling_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("six trajectories done", flush=True)

    # A long trajectory on [0, 800]: metastable switching.
    dom = (0.0, 800.0)
    N = Chebop(lambda t, y: y.diff() - y + y**3, domain=dom)
    N.lbc = 0.0
    f = 0.45 * randnfun(LAMBDA, dom, big=True, key=jax.random.PRNGKey(41))
    y = N.solve(f)
    xx = np.linspace(*dom, 20000)
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.plot(xx, np.asarray(y(xx)), lw=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_ylim(-1.7, 1.7)
    ax.grid(True)
    ax.set_title("Tunnelling")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Tunnelling_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("long trajectory done", flush=True)

    # Larger noise means faster tunnelling.
    f = (0.60 / 0.45) * f
    y = N.solve(f)
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.plot(xx, np.asarray(y(xx)), lw=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_ylim(-1.7, 1.7)
    ax.grid(True)
    ax.set_title("Larger noise means faster tunnelling")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Tunnelling_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("done", flush=True)


if __name__ == "__main__":
    run()
