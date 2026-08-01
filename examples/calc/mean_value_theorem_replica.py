"""Mean Value Theorem.

Faithful replica of calc/MeanValueTheorem.m by Kuan Xu, October 2012.

Original: https://www.chebfun.org/examples/calc/MeanValueTheorem.html
Copyright 2012 by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'calc')


def run():
    os.makedirs(_IMG, exist_ok=True)
    a, b = -6.0, 6.0
    f = cj.chebfun(lambda x: (x - 1) * (x - 2) * (x - 3), domain=[a, b])
    fa = float(np.asarray(f(jnp.asarray([a])))[0])
    fb = float(np.asarray(f(jnp.asarray([b])))[0])
    sl = (fb - fa) / (b - a)
    c = np.sort(np.asarray((f.diff() - sl).roots()))
    print("c =")
    for v in c:
        print(f"    {v:.0f}" if abs(v - round(v)) < 1e-10
              else f"   {v:.15f}")

    xs = np.linspace(a, b, 800)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(xs, np.asarray(f(jnp.asarray(xs))), "b", lw=1.4)
    ax.plot([a, b], [fa, fb], "--k", lw=1.0)
    c1 = float(c[0])
    fc1 = float(np.asarray(f(jnp.asarray([c1])))[0])
    ax.plot([c1], [fc1], ".r", ms=12)
    L = 2.0
    ax.plot([c1 - L, c1 + L], [fc1 - L * sl, fc1 + L * sl], "r", lw=1.4)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "MeanValueTheorem_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


if __name__ == "__main__":
    run()
