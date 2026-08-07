"""Continuous analogue of Wilkinson matrix.

Faithful replica of ode-eig/ContinuousWilkinson.m by Nick Trefethen
(March 2017): the Wilkinson tridiagonal matrix with nearly degenerate
extreme eigenvalues, and its Sturm-Liouville analogue

    L u = u'' + |x| u,   -N <= x <= N,   u(+-N) = 0,

whose top eigenvalues are exponentially close pairs.  Sums and
differences of the near-degenerate eigenfunctions give localized
pseudo-eigenfunctions.

Original: https://www.chebfun.org/examples/ode-eig/ContinuousWilkinson.html
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

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-eig')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # The Wilkinson matrix and its nearly degenerate extreme eigenvalues.
    N = 8
    diag_entries = np.concatenate([np.arange(N, 0, -1), np.arange(0, N + 1)])
    A = (np.diag(diag_entries)
         + np.diag(np.ones(2 * N), 1) + np.diag(np.ones(2 * N), -1))
    e = np.sort(np.linalg.eigvalsh(A))
    print("ans =")
    for v in e[-4:]:
        print(f"   {v:.15f}")

    # The Sturm-Liouville analogue on [-N, 0, N].
    L = Chebop(lambda x, u: u.diff(2) + abs(x) * u, domain=(-N, 0, N))
    L.bc = "dirichlet"
    lam, V = L.eigs(k=4, sigma="LR", return_eigenfunctions=True)
    lam = np.asarray(lam).real
    idx = np.argsort(lam)
    lam, V = lam[idx], [V[i] for i in idx]
    print("e =")
    for v in lam:
        print(f"   {v:.15f}")

    # Even and odd nearly degenerate eigenfunctions.
    xx = np.linspace(-N, N, 4000)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for j in (2, 3):
        ax.plot(xx, np.asarray(V[j](xx)), lw=1.6)
    ax.grid(True)
    ax.set_title("Even and odd eigenfunctions, nearly degenerate")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ContinuousWilkinson_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Localized left/right pseudo-eigenfunctions.
    right = V[3] + V[2]
    left = V[3] - V[2]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(xx, np.asarray(left(xx)), lw=1.6)
    ax.plot(xx, np.asarray(right(xx)), lw=1.6)
    ax.set_title("Left and right pseudo-eigenfunctions, localized")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ContinuousWilkinson_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Residual checks: eigenfunctions ARE eigenfunctions ...
    v = V[3]
    lmbda = lam[3]
    print("ans =")
    print(f"   {float((L(v) - lmbda * v).norm(2)):.4e}")

    # ... while "left" is NEARLY an eigenfunction.
    v = left
    lmbda = np.mean(lam[2:4])
    print("ans =")
    print(f"   {float((L(v) - lmbda * v).norm(2)):.4e}")


if __name__ == "__main__":
    run()
