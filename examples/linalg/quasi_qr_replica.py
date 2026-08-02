"""QR factorization of a quasimatrix.

Faithful replica of linalg/QuasiQR.m by Nick Trefethen (June 2019):
continuous QR factorization of the ill-conditioned quasimatrix with
columns 1/(1 + k(x-0.1)^2), k = 1..10.

Original: https://www.chebfun.org/examples/linalg/QuasiQR.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix, svd_quasimatrix
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

FIG = [0]


def _plotcoeffs(cols, title):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for c in cols:
        cc = np.abs(np.asarray(c.funs[0].tech.coeffs))
        cc = np.maximum(cc, 1e-40)
        ax.semilogy(np.arange(len(cc)), cc, '.', ms=3)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Degree of Chebyshev polynomial")
    ax.set_ylabel("Magnitude of coefficient")
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"QuasiQR_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    cols = [cj.chebfun(lambda x, _k=k: 1.0 / (1 + _k * (x - 0.1)**2))
            for k in range(1, 11)]
    A = Quasimatrix(cols, cols[0].domain)
    _plotcoeffs(cols, "A")
    print("ans =")
    print(f"     {A.cond():.15e}")

    Q, R = A.qr()
    _plotcoeffs(Q.cols, "Q")

    # norm(A - Q*R): largest singular value of the residual quasimatrix
    R = np.asarray(R)
    resid_cols = []
    for j in range(10):
        qr_col = None
        for i in range(10):
            term = Q.cols[i] * float(R[i, j])
            qr_col = term if qr_col is None else qr_col + term
        resid_cols.append(cols[j] - qr_col)
    _, S, _ = svd_quasimatrix(
        Quasimatrix(resid_cols, cols[0].domain))
    print("ans =")
    print(f"     {float(np.max(np.asarray(S))):.15e}")


if __name__ == "__main__":
    run()
