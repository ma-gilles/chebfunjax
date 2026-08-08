"""The white noise paradox.

Faithful replica of ode-random/WhiteNoiseParadox.m by Nick Trefethen
(May 2017): smooth random functions with lambda = 1/4, 1/16, 1/64 in
the 'big' normalization -- as lambda shrinks the amplitude grows,
illustrating the infinite-energy paradox of true white noise, which
Chebfun resolves by cutting off the wave numbers at O(2pi/lambda).

Sample paths use JAX keys: MATLAB's rng(1) randn stream cannot be
reproduced, so these are different samples of the same law.

Original: https://www.chebfun.org/examples/ode-random/WhiteNoiseParadox.html
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
from chebfunjax.utils.randnfun import randnfun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-random')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    xx = np.linspace(-1, 1, 4000)
    for k, ax in enumerate(axes, start=1):
        lam = 1.0 / 4**k
        f = randnfun(lam, (-1.0, 1.0), big=True,
                     key=jax.random.PRNGKey(k))
        ax.plot(xx, np.asarray(f(xx)), lw=1 - 0.2 * k)
        ax.set_title(f"lambda = 1/{4**k}")
        ax.grid(True)
        ax.set_ylim(-30, 30)
        ax.set_xlim(-1, 1)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "WhiteNoiseParadox_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    run()
