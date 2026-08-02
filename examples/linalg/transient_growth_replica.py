"""Transient growth in linear systems.

Faithful replica of linalg/TransientGrowth.m by Nick Trefethen
(May 2011, after a laser-physics example of Kestutis Staliunas): the
norm and energy of exp(tA) for a stable 7x7 matrix showing enormous
transient amplification before eventual decay.

Original: https://www.chebfun.org/examples/linalg/TransientGrowth.html
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
import scipy.linalg as sla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

A = np.array([
    [-1, 0, 0, 0, 0, 0, -625],
    [0, -1, -30, 400, 0, 0, 250],
    [-2, 0, -1, 0, 0, 0, 30],
    [5, -1, 5, -1, 0, 0, 200],
    [11, 1, 25, -10, -1, 1, -200],
    [200, 0, 0, -150, -100, -1, -1000],
    [1, 0, 0, 0, 0, 0, -1]], dtype=float)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()

    print("A =")
    for row in A:
        print("  " + "".join(f"{int(v):7d}" for v in row))

    def op(t_arr):
        t_arr = np.atleast_1d(np.asarray(t_arr, dtype=float))
        out = np.empty_like(t_arr)
        for i, t in enumerate(t_arr.ravel()):
            out.ravel()[i] = np.linalg.norm(sla.expm(t * A), 2)
        return out.reshape(np.shape(t_arr))

    e = cj.chebfun(lambda t: jnp.asarray(op(np.asarray(t))),
                   domain=(0.0, 2.5), splitting=True)
    xs = np.linspace(0, 2.5, 900)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(xs, np.asarray(e(xs)), 'b', lw=2)
    ax.set_xlabel("t", fontsize=14)
    ax.set_ylabel(r"$\|e^{tA}\|$", fontsize=14)
    ax.set_title("amplitude", fontsize=16)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "TransientGrowth_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    e2 = e ** 2
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(xs, np.asarray(e2(xs)), 'b', lw=2)
    ax.set_xlabel("t", fontsize=14)
    ax.set_ylabel(r"$\|e^{tA}\|^2$", fontsize=14)
    ax.set_title("energy", fontsize=16)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "TransientGrowth_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    _, m2 = e2.max()
    print(f"Maximum energy = {float(m2):15.8f}")
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")


if __name__ == "__main__":
    run()
