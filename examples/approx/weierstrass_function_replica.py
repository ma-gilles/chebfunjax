"""The Weierstrass function.

Faithful replica of approx/WeierstrassFunction.m by Hrothgar (October
2013): partial sums of Weierstrass's continuous-nowhere-differentiable
function, their integral, and the cost of computing minima of the
increasingly pathological approximants.

Original: https://www.chebfun.org/examples/approx/WeierstrassFunction.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_IMG, exist_ok=True)

    def f_k(k):
        return lambda x: 2.0**-k * jnp.cos(np.pi / 2 * x * 4.0**k)

    F = [cj.chebfun(f_k(0))]
    for k in range(1, 9):
        F.append(F[k - 1] + cj.chebfun(f_k(k), max_length=2**18))

    xs = np.linspace(-1, 1, 6000)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(xs, np.asarray(F[8](jnp.asarray(xs))), 'k', lw=0.7)
    ax.set_title("A pathological function of Weierstrass", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "WeierstrassFunction_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    xs2 = np.linspace(0, 0.005, 4000)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(xs2, np.asarray(F[8](jnp.asarray(xs2))), 'k', lw=0.9)
    ax.set_title("Close-up of Weierstrass approximant", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "WeierstrassFunction_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    err = float(F[8].sum()) - 4 / np.pi
    print("error =")
    print(f"     {err:.15e}")

    rows = []
    for k in range(0, 8, 2):     # F{1}, F{3}, F{5}, F{7} in MATLAB
        t0 = time.time()
        x_min, m = F[k].min()
        rows.append((k + 2, x_min, m, time.time() - t0))
    print(f"{'k':>2} {'x_min':>11} {'F_k(x_min)':>16} {'computation time':>19}")
    print("-" * 52)
    for k, x, m, t in rows:
        print(f"{k:2d} {x:12.7f} {m:+15.7f} {t:11.2f} sec")


if __name__ == "__main__":
    run()
