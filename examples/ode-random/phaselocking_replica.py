"""Phase-locking in a Duffing-type equation.

Faithful replica of ode-random/PhaseLocking.m by Kevin Burrage and
Nick Trefethen (May 2017): the bistable equation

    y' = t y - y^3 + f,   y(0) = 0,

whose local fixed points +-sqrt(t) separate as t grows: noise crosses
the gap easily at small t, but every trajectory eventually locks onto
one branch forever.  Six paths at lambda = 0.2, six at lambda = 0.05,
then sixty at lambda = 0.05.

Sample paths use JAX keys (MATLAB rng(0) not reproducible).

Original: https://www.chebfun.org/examples/ode-random/PhaseLocking.html
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

DOM = (0.0, 6.0)


def _panel(lam, npaths, key0, fname):
    t0 = time.time()
    N = Chebop(lambda t, y: y.diff() - t * y + y**3, domain=DOM)
    N.lbc = 0.0
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xx = np.linspace(*DOM, 2000)
    for k in range(npaths):
        f = randnfun(lam, DOM, big=True,
                     key=jax.random.PRNGKey(key0 + k))
        y = N.solve(f)
        ax.plot(xx, np.asarray(y(xx)), lw=1.0)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title(f"lambda = {lam}, {npaths} paths")
    ax.set_xlim(*DOM)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{fname}: {time.time()-t0:.1f}s", flush=True)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    _panel(0.2, 6, 100, "PhaseLocking_repl_01.png")
    _panel(0.05, 6, 200, "PhaseLocking_repl_02.png")
    _panel(0.05, 60, 300, "PhaseLocking_repl_03.png")


if __name__ == "__main__":
    run()
