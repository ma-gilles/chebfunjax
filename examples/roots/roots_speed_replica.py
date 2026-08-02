"""Speed and accuracy of Chebfun roots.

Faithful replica of roots/RootsSpeed.m by Nick Trefethen
(October 2012): 2001 roots of exp(x)sin(1000 pi x) via the recursive
subdivided colleague algorithm, accuracy at machine precision, and
timing of polynomial rootfinding at various degrees.

Original: https://www.chebfun.org/examples/roots/RootsSpeed.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')


def run():
    os.makedirs(_IMG, exist_ok=True)

    f = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(1000 * jnp.pi * x))
    print("n =")
    print(f"        {len(f)}")

    exact = np.linspace(-1, 1, 2001)
    r = np.asarray(f.roots())
    t0 = time.time()
    r = np.asarray(f.roots())
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    print("ans =")
    print(f"     {np.max(np.abs(r - exact)):.15e}")

    d = (-0.0105, 0.0105)
    xs = np.linspace(*d, 800)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xs, np.asarray(f(xs)), 'b', lw=1.2)
    ax.axis([d[0], d[1], -1.5, 1.5])
    ax.grid(True)
    rin = r[(r > d[0]) & (r < d[1])]
    ax.plot(rin, np.asarray(f(rin)), '.r', ms=9)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "RootsSpeed_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    rs = np.random.RandomState(5489)
    for ntest in [250, 500, 1000, 2000]:
        c = rs.randn(ntest)
        t0 = time.time()
        np.roots(c)
        print(f"Elapsed time is {time.time()-t0:.6f} seconds.")


if __name__ == "__main__":
    run()
