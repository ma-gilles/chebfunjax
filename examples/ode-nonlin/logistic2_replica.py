"""Logistic map and chaos.

Faithful replica of ode-nonlin/Logistic2.m by Nick Trefethen
(August 2013): iterates of the logistic map x -> r x(1-x) computed as
chebfun compositions, whose lengths explode as the dynamics become
chaotic.

Original: https://www.chebfun.org/examples/ode-nonlin/Logistic2.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]


def _plot(x, r, n):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    dom = (float(x.domain.a), float(x.domain.b))
    t = np.linspace(dom[0], dom[1], 4000)
    ax.plot(t, np.asarray(x(t)), lw=1)
    ax.set_title(f"r={r:4.2f}     n={n}     length(x)={len(x)}",
                 fontsize=12)
    ax.axis([0, 1, 0, 1])
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Logistic2_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()

    r, n = 3.75, 10
    x = cj.chebfun(lambda t: t, domain=(0, 1))
    for _ in range(n):
        x = r * x * (1 - x)
    _plot(x, r, n)

    r = 3.25
    x = cj.chebfun(lambda t: t, domain=(0, 1))
    for _ in range(n):
        x = r * x * (1 - x)
    _plot(x, r, n)

    n = 20
    x0 = cj.chebfun(lambda t: t, domain=(0.02, 0.98))
    x = x0
    for _ in range(n):
        x = r * x * (1 - x)
    _plot(x, r, n)
    for p in (0.5, 0.8):
        print("ans =")
        print(f"   {float(x(jnp.array(p))):.15f}")

    r = 3.5
    x = x0
    for _ in range(n):
        x = r * x * (1 - x)
    _plot(x, r, n)
    for p in (0.5, 0.62, 0.77, 0.83):
        print("ans =")
        print(f"   {float(x(jnp.array(p))):.15f}")

    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")


if __name__ == "__main__":
    run()
