"""The analytic SVD.

Faithful replica of linalg/AnalyticSVD.m by Yuji Nakatsukasa and
Vanni Noferini (May 2016): the singular values of the matrix family
A t + B(1-t) as functions of t.  Sorted singular values have kinks
where branches cross; flipping signs across the crossings recovers
the analytic SVD, in which singular values may go negative but every
branch is smooth.

rng(10) randn is not bit-reproducible vs MATLAB; the crossing
structure of our draw differs while the phenomenon replicates.

Original: https://www.chebfun.org/examples/linalg/AnalyticSVD.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

M = N = 4
FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"AnalyticSVD_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()

    rs = np.random.RandomState(10)
    A = rs.randn(M, N)
    B = rs.randn(M, N)

    def AA(t):
        return A * t + B * (1 - t)

    def sigma_k(t_arr, k):
        t_arr = np.atleast_1d(np.asarray(t_arr, dtype=float))
        out = np.empty_like(t_arr)
        for i, t in enumerate(t_arr.ravel()):
            out.ravel()[i] = np.linalg.svd(AA(t),
                                           compute_uv=False)[k]
        return out.reshape(np.shape(t_arr))

    # sorted singular values: chebfuns with splitting; kinks marked
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for k in range(N):
        f = cj.chebfun(
            lambda t, _k=k: jnp.asarray(sigma_k(np.asarray(t), _k)),
            splitting=True)
        xs = np.linspace(-1, 1, 800)
        ax.plot(xs, np.asarray(f(xs)), lw=2)
        for bp in [float(v) for v in f.domain.breakpoints][1:-1]:
            ax.plot([bp, bp], [0, 4], 'k', lw=0.8)
            ax.plot(bp, float(f(bp)), 'ko', ms=5, mfc='none')
    ax.grid(True)
    ax.set_title("sorted singular values, kinks at crossings",
                 fontsize=12)
    _save(fig)

    # analytic SVD via dense-grid continuation: track each branch
    # smoothly through the crossings, flipping signs (the numpy
    # realization of the example's chebfun sign-surgery)
    m = 4001
    ts = np.linspace(-1, 1, m)
    S = np.empty((m, N))
    U = np.empty((m, N))
    V = np.empty((m, N))
    prevU = None
    prevV = None
    for i, t in enumerate(ts):
        Ui, si, Vti = np.linalg.svd(AA(t))
        Vi = Vti.T
        si = si.copy()
        if prevU is not None:
            # match branches to the previous step by maximal overlap
            overlap = np.abs(prevU.T @ Ui)
            perm = np.full(N, -1)
            used = set()
            for r in range(N):
                order = np.argsort(-overlap[r])
                for c in order:
                    if c not in used:
                        perm[r] = c
                        used.add(c)
                        break
            Ui, Vi, si = Ui[:, perm], Vi[:, perm], si[perm]
            # sign continuity: (u,v,s) -> (su*u, sv*v, su*sv*s)
            # preserves A = sum s u v'
            sgn_u = np.sign(np.sum(prevU * Ui, axis=0))
            sgn_u[sgn_u == 0] = 1.0
            sgn_v = np.sign(np.sum(prevV * Vi, axis=0))
            sgn_v[sgn_v == 0] = 1.0
            Ui = Ui * sgn_u
            Vi = Vi * sgn_v
            si = si * sgn_u * sgn_v
        prevU, prevV = Ui, Vi
        S[i] = si
        U[i] = Ui[0]
        V[i] = Vi[0]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    for k in range(N):
        axes[0].plot(ts, S[:, k], lw=2)
        axes[1].plot(ts, U[:, k], lw=2)
        axes[2].plot(ts, V[:, k], lw=2)
    axes[0].set_title("singular values")
    axes[1].set_title("U")
    axes[2].set_title("V")
    for ax in axes:
        ax.grid(True)
    _save(fig)

    print("time_in_seconds =")
    print(f"     {time.time() - t0:.9e}")


if __name__ == "__main__":
    run()
