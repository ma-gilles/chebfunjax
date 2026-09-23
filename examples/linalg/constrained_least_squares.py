"""Constrained least squares with quasimatrices.

Translation of linalg/ConstrainedLeastSquares.m by Nick Hale
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

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import chebfun_qr, chebfun_svd
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

_GX, _GW = np.polynomial.legendre.leggauss(400)
FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"ConstrainedLeastSquares_{FIG[0]:02d}.png"))
    plt.close(fig)


def _short(v):
    """One entry in MATLAB ``format short`` matrix display."""
    return f"{'0':>10}" if v == 0 else f"{v:10.4f}"


def gqr(A, B):
    """Generalized QR: A' = Q R U, B' = Q S (MATLAB gqr, economy)."""
    Q, S = np.linalg.qr(B.T, mode="complete")
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


def _combine(cols, coef):
    """Quasimatrix-times-matrix: columns sum_k cols[k] * coef[k, j]."""
    out = []
    for j in range(coef.shape[1]):
        g = 0 * cols[0]
        for k, c in enumerate(cols):
            g = g + float(coef[k, j]) * c
        out.append(g)
    return out


def gqr_quasi(Acols, B):
    """Generalized QR of a quasimatrix A and a matrix B (MATLAB gqr(A,B,0)).

    Returns Q, R, S (matrices) and the rows of U as chebfuns, with
    A' = Q R U and B' = Q S.
    """
    Q, S = np.linalg.qr(B.T, mode="complete")
    AQ = _combine(Acols, Q)
    # flipud(fliplr(A*Q)): reverse the columns and reflect x -> -x
    flipped = [cj.chebfun(lambda t, _g=g: _g(-t)) for g in AQ[::-1]]
    U0, R0 = chebfun_qr(flipped)
    R = np.rot90(np.asarray(R0).T, 2)
    U = [cj.chebfun(lambda t, _u=u: _u(-t)) for u in list(U0.cols)[::-1]]
    return Q, U, R, S


def _norm2(cols):
    """2-norm of a quasimatrix (largest singular value)."""
    return float(np.max(np.asarray(chebfun_svd(cols)[1])))


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


def lsqcone(Avals, fvals, Bmat, d):
    """min ||A c - f||_L2 s.t. B c = d, by elimination (MATLAB lsqcone)."""
    Qb, Rb, P = scipy.linalg.qr(Bmat, mode="economic", pivoting=True)
    n = Avals.shape[1]
    p = Bmat.shape[0]
    j1, j2 = P[:p], P[p:]
    R1, R2 = Rb[:, :p], Rb[:, p:]
    A1, A2 = Avals[:, j1], Avals[:, j2]
    AA = A2 - A1 @ np.linalg.solve(R1, R2)
    c = Qb.T @ d
    bb = fvals - A1 @ np.linalg.solve(R1, c)
    y2 = _wlstsq(AA, bb)
    y1 = np.linalg.solve(R1, c - R2 @ y2)
    x = np.zeros(n)
    x[P] = np.concatenate([y1, y2])
    return x


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
            print("".join(_short(v) for v in row))
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
        print(_short(v))
    sol = np.array([46, -2, 12]) / 8
    print("sol =")
    for v in sol:
        print(_short(v))
    print("err =")
    print(f"   {np.linalg.norm(x - sol):.4e}")

    # interpolation constraints at z = [-0.5, 0]
    z = np.array([-0.5, 0.0])
    Bz = z[:, None] ** np.arange(6)
    dz = np.exp(z) * np.sin(6 * z)

    # generalized QR of the quasimatrix [1, x, ..., x^5] and B = A(z)
    Acols = [cj.chebfun(lambda t, _k=k: t**_k + 0 * t) for k in range(6)]
    Qg, Ug, Rg, Sg = gqr_quasi(Acols, Bz)
    QR = Qg @ Rg
    resid = [Acols[j] - _combine(Ug, QR[[j], :].T)[0] for j in range(6)]
    UUt = np.array([[float((u * w).sum()) for w in Ug] for u in Ug])
    err = (_norm2(resid)
           + np.linalg.norm(Bz.T - Qg @ Sg, 2)
           + np.linalg.norm(Qg @ Qg.T - np.eye(6), 2)
           + np.linalg.norm(UUt - np.eye(6), 2))
    print("err =")
    print(f"   {err:.4e}")
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.7))
    for ax, M, name in ((axes[0], Rg, "R"), (axes[1], Sg, "S")):
        ax.spy(M, marker='.', markersize=8, color=(0, 0, 0.8))
        ax.set_title(name)
        ax.set_xlabel(f"nz = {np.count_nonzero(M)}")
        ax.xaxis.set_ticks_position('bottom')
        # MATLAB indices are 1-based
        ax.set_xticks(range(M.shape[1]), range(1, M.shape[1] + 1))
        ax.set_yticks(range(M.shape[0]), range(1, M.shape[0] + 1))
    _save(fig)

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
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    for k, ck in enumerate(centers):
        gk = cj.chebfun(lambda t, _c=ck: jnp.exp(-5 * (t - _c) ** 2))
        ax.plot(xp, np.asarray(gk(jnp.asarray(xp))), lw=1.6)
    ax.grid(True)
    _save(fig)
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

    c3g = lsqcone(Ag, fvals, Bg, dg)
    print("err =")
    print(f"   {np.linalg.norm(c2g - c3g):.4e}")


if __name__ == "__main__":
    run()
