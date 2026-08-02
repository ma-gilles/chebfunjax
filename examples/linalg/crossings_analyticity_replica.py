"""Eigenvalue near-crossings and analyticity.

Faithful replica of linalg/CrossingsAnalyticity.m by Nick Trefethen
(June 2021): near-crossing eigenvalue curves of (1-t)A + tB are
analytic, but only in a narrow strip — revealed by the poles of AAA
approximants; symmetric functions of the eigenvalues are analytic in
a much wider strip.

rng(1) randn draws are not bit-reproducible vs MATLAB; the
phenomenon replicates on our draw.

Original: https://www.chebfun.org/examples/linalg/CrossingsAnalyticity.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.aaa import aaa

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"CrossingsAnalyticity_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    n = 10
    rs = np.random.RandomState(1)
    A = rs.randn(n, n)
    A = A + A.T
    B = rs.randn(n, n)
    B = B + B.T

    def eigk(t_arr, k):
        t_arr = np.atleast_1d(np.asarray(t_arr, dtype=float))
        out = np.empty_like(t_arr)
        for i, t in enumerate(t_arr.ravel()):
            out.ravel()[i] = np.sort(np.linalg.eigvalsh(
                (1 - t) * A + t * B))[k]
        return out.reshape(np.shape(t_arr))

    # pick the adjacent pair with the closest interior approach
    E = [cj.chebfun(lambda t, _k=k: jnp.asarray(
        eigk(np.asarray(t), _k)), domain=(0.0, 1.0))
        for k in range(n)]
    gaps = [(k, (E[k + 1] - E[k]).min()) for k in range(n - 1)]
    interior = [g for g in gaps
                if 1e-3 < float(g[1][0]) < 1 - 1e-3]
    k0, _ = min(interior if interior else gaps,
                key=lambda g: float(g[1][1]))
    E1, E2 = E[k0], E[k0 + 1]

    xs = np.linspace(0, 1, 800)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(xs, np.asarray(E1(xs)), lw=1.6)
    ax.plot(xs, np.asarray(E2(xs)), lw=1.6)
    ax.grid(True)
    ax.set_title("Near-crossing of two eigenvalues", fontsize=12)
    ax.set_xlabel("t")
    _save(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(xs, np.asarray(E1.diff()(xs)), lw=1.6)
    ax.plot(xs, np.asarray(E2.diff()(xs)), lw=1.6)
    ax.grid(True)
    ax.set_title("derivatives of the eigenvalue functions",
                 fontsize=12)
    _save(fig)

    X = np.linspace(0, 1, 1000)
    _, pol, *_ = aaa(jnp.asarray(np.asarray(E1(X))),
                     jnp.asarray(X))
    pol = np.asarray(pol)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(pol.real, pol.imag, '.r', ms=8)
    ax.grid(True)
    ax.axis([0, 1, -0.2, 0.2])
    ax.set_title("narrow strip of analyticity", fontsize=12)
    _save(fig)

    Esum = E1 + E2
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(xs, np.asarray(Esum(xs)), lw=1.6)
    ax.grid(True)
    ax.set_title("sum of the two eigenvalues", fontsize=12)
    _save(fig)

    _, pol2, *_ = aaa(jnp.asarray(np.asarray(Esum(X))),
                      jnp.asarray(X))
    pol2 = np.asarray(pol2)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(pol2.real, pol2.imag, '.r', ms=9)
    ax.grid(True)
    ax.axis([0, 1, -0.2, 0.2])
    ax.set_title("for the sum, a wider strip of analyticity",
                 fontsize=12)
    _save(fig)


if __name__ == "__main__":
    run()
