"""Constrained least squares with quasimatrices.

Faithful replica of linalg/ConstrainedLeastSquares.m by Nick Hale
(March 2017): least-squares fitting with linear equality constraints
via the generalized QR factorization — for discrete matrices
digit-for-digit, and for quasimatrices with interpolation and
integral constraints (the continuous inner products are realized by
400-point Gauss-Legendre quadrature, exact for polynomials and
accurate to machine precision for the smooth functions used here).

Original: https://www.chebfun.org/examples/linalg/ConstrainedLeastSquares.html
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

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

_GX, _GW = np.polynomial.legendre.leggauss(400)
FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"ConstrainedLeastSquares_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def gqr(A, B):
    """Generalized QR: A' = Q R U, B' = Q S (MATLAB gqr, economy)."""
    Q, S = np.linalg.qr(B.T, mode="complete")
    S = Q.T @ B.T
    AQ = A @ Q
    U0, R0 = np.linalg.qr(np.flipud(np.fliplr(AQ)))
    R = np.rot90(R0.T, 2)
    U = np.flipud(np.fliplr(U0.T))
    return Q, U, R, S


def lsqcon_disc(A, b, B, d):
    """min |Ax-b| s.t. Bx=d (discrete, via gqr)."""
    Q, U, R, S = gqr(A, B)
    n = A.shape[1]
    p = B.shape[0]
    i1 = slice(0, p)
    i2 = slice(p, n)
    y1 = np.linalg.solve(S[i1, :p].T, d)
    y2 = np.linalg.solve(R[i2, i2].T,
                         U[i2, :] @ b - R[i1, i2].T @ y1)
    return Q @ np.concatenate([y1, y2])


def _wlstsq(Avals, fvals):
    sw = np.sqrt(_GW)
    c, *_ = np.linalg.lstsq(sw[:, None] * Avals, sw * fvals,
                            rcond=None)
    return c


def lsqcon_cont(Avals, fvals, Bmat, d):
    """min ||A c - f||_L2 s.t. B c = d, by null-space elimination."""
    Bmat = np.atleast_2d(np.asarray(Bmat, dtype=float))
    d = np.atleast_1d(np.asarray(d, dtype=float))
    c_p, *_ = np.linalg.lstsq(Bmat, d, rcond=None)
    _, sv, Vt = np.linalg.svd(Bmat)
    rank = int(np.sum(sv > 1e-12 * max(sv, default=1)))
    Z = Vt[rank:].T
    resid = fvals - Avals @ c_p
    y = _wlstsq(Avals @ Z, resid)
    return c_p + Z @ y


def run():
    os.makedirs(_IMG, exist_ok=True)

    xs = _GX
    Avals = xs[:, None] ** np.arange(6)
    fvals = np.exp(xs) * np.sin(6 * xs)
    c = _wlstsq(Avals, fvals)
    xp = np.linspace(-1, 1, 800)
    Ap = xp[:, None] ** np.arange(6)
    fp = np.exp(xp) * np.sin(6 * xp)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(xp, fp, lw=1.6, label="f")
    ax.plot(xp, Ap @ c, lw=1.6, label="ffit")
    ax.legend()
    ax.grid(True)
    _save(fig)

    # discrete generalized QR: digit-for-digit with MATLAB
    A = np.array([[1, 1, 1], [1, 3, 1], [1, -1, 1], [1, 1, 1]],
                 dtype=float)
    B = np.array([[1, 1, 1], [1, 1, -1]], dtype=float)
    Q, U, R, S = gqr(A, B)
    for name, M in (("Q", Q), ("U", U), ("R", R), ("S", S)):
        print(f"{name} =")
        for row in np.atleast_2d(M):
            print("  " + "".join(f"{v:10.4f}" for v in row))
    err = (np.linalg.norm(A.T - Q @ R @ U)
           + np.linalg.norm(B.T - Q @ S)
           + np.linalg.norm(Q @ Q.T - np.eye(3))
           + np.linalg.norm(U @ U.T - np.eye(3)))
    print("err =")
    print(f"   {err:.4e}")

    b = np.array([1.0, 2, 3, 4])
    d = np.array([7.0, 4])
    x = lsqcon_disc(A, b, B, d)
    print("x =")
    for v in x:
        print(f"    {v:.4f}")
    sol = np.array([46, -2, 12]) / 8
    print("err =")
    print(f"   {np.linalg.norm(x - sol):.4e}")

    # interpolation constraints at z = [-0.5, 0]
    z = np.array([-0.5, 0.0])
    Bz = z[:, None] ** np.arange(6)
    dz = np.exp(z) * np.sin(6 * z)
    c2 = lsqcon_cont(Avals, fvals, Bz, dz)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(xp, fp, lw=1.6, label="f")
    ax.plot(xp, Ap @ c, lw=1.6, label="ffit")
    ax.plot(xp, Ap @ c2, lw=1.6, label="ffit2")
    ax.plot(z, dz, 'xk', ms=9)
    ax.legend()
    ax.grid(True)
    _save(fig)
    print("interp constraint residual =")
    print(f"   {np.max(np.abs(Bz @ c2 - dz)):.4e}")

    # integral constraint sum(u) = 0 via the functional row
    Brow = (_GW @ Avals).reshape(1, -1)
    c3 = lsqcon_cont(Avals, fvals, Brow, [0.0])
    print("err =")
    print(f"   {float((Brow @ c3)[0]):.4e}")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(xp, fp, lw=1.6, label="f")
    ax.plot(xp, Ap @ c, lw=1.6, label="ffit")
    ax.plot(xp, Ap @ c2, lw=1.6, label="ffit2")
    ax.plot(xp, Ap @ c3, lw=1.6, label="ffit3")
    ax.legend()
    ax.grid(True)
    _save(fig)

    # Gaussian basis with endpoint + integral constraints
    centers = np.arange(-3, 4) / 3
    Ag = np.exp(-5 * (xs[:, None] - centers) ** 2)
    Agp = np.exp(-5 * (xp[:, None] - centers) ** 2)
    c1g = _wlstsq(Ag, fvals)
    ends = np.array([-1.0, 1.0])
    Bg = np.vstack([np.exp(-5 * (ends[:, None] - centers) ** 2),
                    (_GW @ Ag).reshape(1, -1)])
    dg = np.concatenate([np.exp(ends) * np.sin(6 * ends), [0.0]])
    c2g = lsqcon_cont(Ag, fvals, Bg, dg)
    ffit2_ends = np.exp(-5 * (ends[:, None] - centers) ** 2) @ c2g
    err = np.sqrt(float(_GW @ (Ag @ c2g)) ** 2
                  + np.sum(np.exp(ends) * np.sin(6 * ends)
                           - ffit2_ends) ** 2)
    print("err =")
    print(f"   {err:.4e}")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(xp, fp, lw=1.6, label="f")
    ax.plot(xp, Agp @ c1g, lw=1.6, label="ffit")
    ax.plot(xp, Agp @ c2g, lw=1.6, label="ffit2")
    ax.legend()
    ax.grid(True)
    _save(fig)


if __name__ == "__main__":
    run()
