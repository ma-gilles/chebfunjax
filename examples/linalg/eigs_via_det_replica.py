"""Eigenvalues of a tridiagonal matrix via the determinant.

Faithful replica of linalg/EigsViaDet.m by Nick Trefethen
(December 2011): eigenvalues of a random tridiagonal matrix computed
as roots of the characteristic polynomial det(xI - A), evaluated by
the tridiagonal three-term recurrence, as a chebfun — globally, on a
subinterval (better conditioned), and via sign + edge detection
(best of all).

MATLAB's a-vector (rand, rng(2)) is bit-reproducible via
RandomState(2); the b-vector uses randn, which never is — so the
spectrum differs from the published run while every accuracy
comparison replicates.

Original: https://www.chebfun.org/examples/linalg/EigsViaDet.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"EigsViaDet_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    t0 = time.time()

    N = 100
    rs = np.random.RandomState(2)
    a = 10 * rs.rand(N) - 5
    b = rs.randn(N - 1)
    A = np.diag(a) + np.diag(b, -1) + np.diag(b, 1)

    def fdet(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        dold = np.ones_like(x)
        d = x - a[0]
        for k in range(N - 1):
            dnew = (x - a[k + 1]) * d - b[k] ** 2 * dold
            dold, d = d, dnew
        return d

    e = np.linalg.eigvalsh(A)
    e_exact = np.sort(e[np.abs(e) <= 1])
    print("e_exact =")
    for v in e_exact:
        print(f"  {v:>19.15f}")

    op = lambda x: jnp.asarray(fdet(np.asarray(x)))  # noqa: E731
    c = cj.chebfun(op, domain=(-1.0, 1.0))
    xs = np.linspace(-1, 1, 1000)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xs, np.asarray(c(xs)), 'b', lw=1.2)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_title("det(xI-A) as a chebfun", fontsize=12)
    _save(fig)

    e_inexact = np.asarray(c.roots())
    print("         exact              inexact            difference")
    for u, v in zip(e_exact, e_inexact):
        print(f"  {u:>19.15f} {v:>19.15f} {u - v:>19.15f}")

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    with np.errstate(all="ignore"):
        ax.semilogy(xs, np.abs(np.asarray(c(xs))), 'b', lw=1.2)
    ax.set_ylim(1e22, 1e32)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_title("|det(xI-A)| on a log scale", fontsize=12)
    _save(fig)

    e_exact_neg = np.sort(e[(e < 0) & (np.abs(e) < 1)])
    c_neg = cj.chebfun(op, domain=(-1.0, 0.0))
    xs2 = np.linspace(-1, 0, 600)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xs2, np.asarray(c_neg(xs2)), 'b', lw=1.2)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_title("det(xI-A) on a smaller interval", fontsize=12)
    _save(fig)

    e_inexact2 = np.asarray(c_neg.roots())
    print("         exact              inexact            difference")
    print(f"ans =\n    {len(e_exact_neg)}     1")
    print(f"ans =\n    {len(e_inexact2)}     1")
    for u, v in zip(e_exact_neg, e_inexact2):
        print(f"  {u:>19.15f} {v:>19.15f} {u - v:>19.15f}")

    c2 = cj.chebfun(lambda x: jnp.sign(jnp.asarray(fdet(
        np.asarray(x)))), domain=(-1.0, 1.0), splitting=True,
        min_samples=100)
    e_edge = np.asarray(c2.roots(nojump=False))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xs, np.asarray(c2(xs)), 'b', lw=1.0)
    ax.set_ylim(-1.4, 1.4)
    ax.grid(True)
    ax.plot(e_edge, np.zeros_like(e_edge), '.r', ms=8)
    _save(fig)

    print("         exact        via edge detection      difference")
    print(f"ans =\n    {len(e_exact)}     1")
    print(f"ans =\n    {len(e_edge)}     1")
    for u, v in zip(e_exact, e_edge):
        print(f"  {u:>19.15f} {v:>19.15f} {u - v:>19.15f}")
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")


if __name__ == "__main__":
    run()
