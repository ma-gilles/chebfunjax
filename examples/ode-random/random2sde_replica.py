"""From random functions to SDEs.

Faithful replica of ode-random/Random2SDE.m by Nick Trefethen and
Abdul-Lateef Haji-Ali (May 2017): three "smooth random walk" sample
paths -- cumsum of normalized ('big') random functions with
lambda = 0.001 on [0, 1] -- which for small lambda look to the eye
like Brownian motion, the Stratonovich SDE limit.

Sample paths use JAX keys (MATLAB rng(0) not reproducible).

Original: https://www.chebfun.org/examples/ode-random/Random2SDE.html
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

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.randnfun import randnfun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-random')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xx = np.linspace(0, 1, 8000)
    for k in range(3):
        u = randnfun(0.001, (0.0, 1.0), big=True,
                     key=jax.random.PRNGKey(k))
        w = u.cumsum()
        ax.plot(xx, np.asarray(w(xx)), lw=1.0)
    ax.grid(True)
    ax.set_ylim(-2, 2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("t")
    ax.set_ylabel("u")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Random2SDE_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("total_time_in_seconds =")
    print(f"  {time.time() - t0:.6f}")


if __name__ == "__main__":
    run()
