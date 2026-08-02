"""Vandermonde with Arnoldi.

Faithful replica of linalg/VandermondeArnoldi.m by Nick Trefethen
(July 2020, after Brubeck-Nakatsukasa-Trefethen, SIAM Review 2021):
Vandermonde matrices and quasimatrices are exponentially
ill-conditioned; orthogonalizing on the fly with Arnoldi fixes
polynomial least-squares fitting without changing the mathematics.

Original: https://www.chebfun.org/examples/linalg/VandermondeArnoldi.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.lebesgue import lebesgue_constant
from chebfunjax.utils.quadrature import chebpts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')


# Continuous L2 inner products on [-1,1] realized by 400-point
# Gauss-Legendre quadrature — exact for polynomials up to degree 799,
# so every quantity below is the true continuous one.
_GX, _GW = np.polynomial.legendre.leggauss(400)


def _qm_vals(n):
    """Value matrix of the monomial quasimatrix x^(0:n-1) on the
    Gauss grid, and the weight vector."""
    return _GX[:, None] ** np.arange(n)


def _cond_qm(n):
    A = _qm_vals(n)
    return np.linalg.cond(np.sqrt(_GW)[:, None] * A)


def run():
    os.makedirs(_IMG, exist_ok=True)

    for m in (17, 33):
        pts = np.asarray(chebpts(m))
        V = np.vander(pts, increasing=False)
        print("ans =")
        print(f"   {np.linalg.cond(V):.4e}")
    for m in (17, 33):
        pts = np.asarray(chebpts(m))
        A = pts[:, None] ** np.arange(m)
        print("ans =")
        print(f"   {np.linalg.cond(A):.4e}")

    for m in (17, 33):
        L = lebesgue_constant(np.asarray(chebpts(m)))
        print(f"L{m-1} =")
        print(f"    {float(L):.4f}")

    for m in (17, 33):
        print("ans =")
        print(f"   {_cond_qm(m):.4e}")
    for m in (17, 33):
        pts = np.linspace(-1, 1, m)
        print("ans =")
        print(f"   {np.linalg.cond(np.vander(pts)):.4e}")

    # ill-conditioned monomial least-squares fit of |x|, degree 80
    n = 80
    fvals = np.abs(_GX)
    A = _qm_vals(n + 1)
    sw = np.sqrt(_GW)
    Qd, Rd = np.linalg.qr(sw[:, None] * A)
    c = np.linalg.solve(Rd, Qd.T @ (sw * fvals))
    xs = np.linspace(-1, 1, 1201)
    y = (xs[:, None] ** np.arange(n + 1)) @ c
    print("max(y) =")
    print(f"    {np.max(y):.4f}")
    print("norm(c,inf) =")
    print(f"   {np.max(np.abs(c)):.4e}")

    # Arnoldi version: orthogonalize the powers on the fly in the
    # same continuous inner product
    W = np.ones((len(_GX), 1))
    H = np.zeros((n + 1, n))
    for k in range(n):
        q = _GX * W[:, k]
        for j in range(k + 1):
            H[j, k] = np.sum(_GW * W[:, j] * q)
            q = q - H[j, k] * W[:, j]
        H[k + 1, k] = np.sqrt(np.sum(_GW * q * q))
        W = np.column_stack([W, q / H[k + 1, k]])
    # d = Q\\f: weighted least squares (the first Arnoldi
    # column is the unnormalized constant, as in MATLAB)
    sww = np.sqrt(_GW)
    d, *_ = np.linalg.lstsq(sww[:, None] * W, sww * fvals,
                            rcond=None)
    # evaluate the Arnoldi basis at plotting points via the recurrence
    Wx = np.ones((len(xs), 1))
    for k in range(n):
        w = xs * Wx[:, k]
        for j in range(k + 1):
            w = w - H[j, k] * Wx[:, j]
        Wx = np.column_stack([Wx, w / H[k + 1, k]])
    yA = Wx @ d
    print("yA endpoint values:",
          f"{yA[0]:.6f} {yA[-1]:.6f}")
    print("max(yA) =")
    print(f"    {np.max(yA):.4f}")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(xs, y, lw=1.2, label="monomial fit y")
    ax.plot(xs, yA, lw=1.2, label="Arnoldi fit yA")
    ax.legend()
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "VandermondeArnoldi_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
