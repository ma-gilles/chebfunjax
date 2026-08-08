"""Random level hopping.

Faithful replica of ode-random/LevelHopping.m by Nick Trefethen (May
2017): y' = -2 sin(2 pi y) + f has stable fixed points at the
integers; smooth random noise makes the process hop between them.
Two trajectories on [0, 100]: lambda = 0.4 and 0.2.

Sample paths use JAX keys (MATLAB rng(0) not reproducible).

Original: https://www.chebfun.org/examples/ode-random/LevelHopping.html
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

DOM = (0.0, 100.0)


def _run_one(lam, key, lw, fname):
    N = Chebop(lambda y: y.diff() + 2 * (2 * np.pi * y).sin(), domain=DOM)
    N.lbc = 0.0
    f = randnfun(lam, DOM, big=True, key=key)
    y = N.solve(f)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    xx = np.linspace(*DOM, 8000)
    ax.plot(xx, np.asarray(y(xx)), lw=lw)
    ax.grid(True)
    ax.set_xlabel("t", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.set_xlim(*DOM)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"lambda={lam}: levels visited "
          f"{sorted(set(np.round(np.asarray(y(xx))).astype(int)))}",
          flush=True)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()
    _run_one(0.4, jax.random.PRNGKey(0), 2, "LevelHopping_repl_01.png")
    _run_one(0.2, jax.random.PRNGKey(1), 1, "LevelHopping_repl_02.png")
    print("total_time_in_seconds =")
    print(f"  {time.time() - t0:.6f}")


if __name__ == "__main__":
    run()
