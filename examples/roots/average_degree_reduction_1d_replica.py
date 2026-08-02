"""The average degree reduction of subdivision (1D).

Faithful replica of roots/AverageDegreeReduction1D.m by Alex Townsend
(August 2013): the parameter tau measuring how polynomial degrees
shrink under interval subdivision in the recursive rootfinding
algorithm, for oscillatory, non-smooth, and near-singular functions.

Original: https://www.chebfun.org/examples/roots/AverageDegreeReduction1D.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')

FIG = [0]


def _len(f):
    return sum(p.tech.coeffs.shape[0] for p in f.funs)


def compute_tau(op, N):
    f_glob = cj.chebfun(op)
    length = _len(f_glob)
    bps = tuple(-1 + 2 * np.arange(N + 1) / N)
    f_sub = cj.chebfun(op, domain=bps)
    newlen = _len(f_sub) / N
    tau = (newlen / length) ** (1 / np.log2(N))
    print(f"tau = {tau:.5f}")


def subdivision_diagram(op, levels):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(9.4, 1.6 + 1.3 * levels))
    for lvl in range(levels + 1):
        ax.plot([-1, 1], [lvl, lvl], 'k', lw=2)
        xx = np.linspace(-1, 1, 2**lvl + 1)
        newg = cj.chebfun(op, domain=tuple(xx))
        for j, x in enumerate(xx[:-1]):
            ax.plot([x, x], [lvl - 0.1, lvl + 0.1], 'k', lw=2)
            n_piece = newg.funs[j].tech.coeffs.shape[0] - 1
            ax.text(x + 2.0**(-lvl) - 0.09, lvl + 0.12,
                    f"N = {n_piece}", fontsize=9)
        ax.plot([1, 1], [lvl - 0.1, lvl + 0.1], 'k', lw=2)
    ax.set_yticks([])
    ax.set_ylim(-0.5, levels + 0.6)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"AverageDegreeReduction1D_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    M = 5e2
    op = lambda x: jnp.sin(M * x)  # noqa: E731
    compute_tau(op, 2**3)
    subdivision_diagram(op, 3)

    op = lambda x: jnp.abs(x)**3  # noqa: E731
    compute_tau(op, 2)
    subdivision_diagram(op, 1)

    op = lambda x: jnp.abs(x - 0.01)**7  # noqa: E731
    compute_tau(op, 2**3)

    op = lambda x: 1.0 / (x - 1.0001)  # noqa: E731
    compute_tau(op, 8)
    subdivision_diagram(op, 3)

    m = []
    c = 1.0001
    eps_ = np.finfo(float).eps
    for lvl in range(1, 5):
        a, b = 1 - 2.0**(-lvl + 1), 1.0
        A = -(-2 * c + b + a) / (b - a)
        m.append(int(np.ceil(
            np.log(-4 * (a - c) / eps_ / (b - a)
                   / np.sqrt(A**2 - 1))
            / np.log(A + np.sqrt(A**2 - 1)))))
    print("m =")
    for v in m:
        print(f"        {v}")

    xx = np.linspace(-1, 1, 10)

    def op_prod(x):
        return jnp.abs(
            jnp.prod(jnp.asarray(x)[..., None] - xx, axis=-1))**3

    compute_tau(op_prod, 2)

    M = 2000
    f = cj.chebfun(lambda x: jnp.sin(M * x))
    t0 = time.time()
    f.roots()
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")

    c2 = -1.0001
    f = cj.chebfun(lambda x: 1.0 / (x + c2))
    t0 = time.time()
    f.roots()
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")


if __name__ == "__main__":
    run()
