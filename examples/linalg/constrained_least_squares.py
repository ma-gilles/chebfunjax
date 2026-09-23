"""Constrained least squares with quasimatrices.

Translation of linalg/ConstrainedLeastSquares.m by Nick Hale
(March 2017): least-squares fitting with linear equality constraints
via the generalized QR factorization -- for discrete matrices and for
quasimatrices with interpolation and integral constraints.  The
quasimatrix algebra (QR, products, evaluation, integrals) is
chebfunjax's :class:`Quasimatrix`.

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
from chebfunjax.chebfun1d.linalg import Quasimatrix, chebfun_qr, chebfun_svd
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

FIG = [0]


def _save(fig, close=True):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.set_size_inches(6.0, 2.7)
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"ConstrainedLeastSquares_{FIG[0]:02d}.png"))
    if close:
        plt.close(fig)


def _short(v):
    """One entry in MATLAB ``format short`` matrix display."""
    return f"{'0':>10}" if v == 0 else f"{v:10.4f}"


def _quasi(cols):
    return Quasimatrix(list(cols), cols[0].domain)


def _backslash(A, f):
    """A\\f for a quasimatrix A (MATLAB @chebfun/mldivide: QR)."""
    Q, R = chebfun_qr(list(A.cols))
    return np.linalg.solve(np.asarray(R),
                           np.array([float(q.inner(f)) for q in Q.cols]))


def gqr(A, B):
    """Generalized QR: A' = Q R U, B' = Q S (MATLAB gqr(A, B, 0)).

    For a quasimatrix A the rows of U are returned as a list of chebfuns
    and flipud/fliplr of a quasimatrix reflect x and reverse columns.
    """
    Q, S = np.linalg.qr(B.T, mode="complete")
    if isinstance(A, Quasimatrix):
        AQ = A @ Q
        U0, R0 = chebfun_qr([g.flipud() for g in AQ.cols[::-1]])
        U = [u.flipud() for u in list(U0.cols)[::-1]]
    else:
        U0, R0 = np.linalg.qr(np.flipud(np.fliplr(A @ Q)))
        U = np.flipud(np.fliplr(U0.T))
    R = np.rot90(np.asarray(R0).T, 2)
    return Q, U, R, S


def lsqcon(A, b, B, d):
    """min |Ax-b|_2 s.t. Bx = d (MATLAB lsqcon); B may be a functional."""
    if callable(B):
        B = B(A)
    Q, U, R, S = gqr(A, B)
    n = len(A.cols) if isinstance(A, Quasimatrix) else A.shape[1]
    p = S.shape[1]
    y1 = np.linalg.solve(S[:p, :p].T, d)
    if isinstance(A, Quasimatrix):
        Ub = np.array([float((u * b).sum()) for u in U[p:]])
    else:
        Ub = U[p:, :] @ b
    y2 = np.linalg.solve(R[p:n, p:n].T, Ub - R[:p, p:n].T @ y1)
    return Q @ np.concatenate([y1, y2])


def lsqcone(A, b, B, d):
    """min |Ax-b|_2 s.t. Bx = d via elimination (MATLAB lsqcone)."""
    if callable(B):
        B = B(A)
    Qb, Rb, P = scipy.linalg.qr(B, mode="economic", pivoting=True)
    n = len(A.cols)
    p = B.shape[0]
    j1, j2 = P[:p], P[p:]
    R1, R2 = Rb[:, :p], Rb[:, p:]
    A1 = _quasi([A.cols[j] for j in j1])
    A2 = _quasi([A.cols[j] for j in j2])
    AA = A2 - A1 @ np.linalg.solve(R1, R2)
    c = Qb.T @ d
    bb = b - (A1 @ np.linalg.solve(R1, c)).cols[0]
    y2 = _backslash(AA, bb)
    y1 = np.linalg.solve(R1, c - R2 @ y2)
    x = np.zeros(n)
    x[P] = np.concatenate([y1, y2])
    return x


def _norm2(cols):
    """2-norm of a quasimatrix (largest singular value)."""
    return float(np.max(np.asarray(chebfun_svd(list(cols))[1])))


def run():
    os.makedirs(_IMG, exist_ok=True)

    x = cj.chebfun(lambda t: t)
    A = _quasi([x**k if k else 1 + 0 * x for k in range(6)])
    f = (x.exp()) * (6 * x).sin()
    c = _backslash(A, f)
    ffit = (A @ c).cols[0]
    fig, ax = plt.subplots()
    matlab_plot(_quasi([f, ffit]), ax=ax)
    ax.legend(["f", "ffit"])
    _save(fig)

    # discrete generalized QR
    Ad = np.array([[1, 1, 1], [1, 3, 1], [1, -1, 1], [1, 1, 1]],
                  dtype=float)
    Bd = np.array([[1, 1, 1], [1, 1, -1]], dtype=float)
    Q, U, R, S = gqr(Ad, Bd)
    for name, M in (("Q", Q), ("U", U), ("R", R), ("S", S)):
        print(f"{name} =")
        for row in np.atleast_2d(M):
            print("".join(_short(v) for v in row))
    I3 = np.eye(3)
    err = (np.linalg.norm(Ad.T - Q @ R @ U, 2) + np.linalg.norm(Bd.T - Q @ S, 2)
           + np.linalg.norm(Q @ Q.T - I3, 2) + np.linalg.norm(U @ U.T - I3, 2))
    print("err =")
    print(f"   {err:.4e}")

    b = np.array([1.0, 2, 3, 4])
    d = np.array([7.0, 4])
    xd = lsqcon(Ad, b, Bd, d)
    print("x =")
    for v in xd:
        print(_short(v))
    sol = np.array([46, -2, 12]) / 8
    print("sol =")
    for v in sol:
        print(_short(v))
    print("err =")
    print(f"   {np.linalg.norm(xd - sol):.4e}")

    # Example 1: interpolation constraints at z = [-0.5, 0]
    z = np.array([-0.5, 0.0])
    B = np.asarray(A(jnp.asarray(z)))
    d = np.asarray(f(jnp.asarray(z)))
    Q, U, R, S = gqr(A, B)
    QR = Q @ R
    resid = [A.cols[j] - sum((float(QR[j, i]) * U[i] for i in range(1, 6)),
                             float(QR[j, 0]) * U[0]) for j in range(6)]
    UUt = np.array([[float((u * w).sum()) for w in U] for u in U])
    err = (_norm2(resid) + np.linalg.norm(B.T - Q @ S, 2)
           + np.linalg.norm(Q @ Q.T - np.eye(6), 2)
           + np.linalg.norm(UUt - np.eye(6), 2))
    print("err =")
    print(f"   {err:.4e}")
    fig, axes = plt.subplots(1, 2)
    for ax, M, name in ((axes[0], R, "R"), (axes[1], S, "S")):
        ax.spy(M, marker='.', markersize=8, color=(0, 0, 0.8))
        ax.set_title(name)
        ax.set_xlabel(f"nz = {np.count_nonzero(M)}")
        ax.xaxis.set_ticks_position('bottom')
        # MATLAB indices are 1-based
        ax.set_xticks(range(M.shape[1]), range(1, M.shape[1] + 1))
        ax.set_yticks(range(M.shape[0]), range(1, M.shape[0] + 1))
    _save(fig)

    c = lsqcon(A, f, B, d)
    ffit2 = (A @ c).cols[0]
    fig, ax = plt.subplots()
    matlab_plot(_quasi([f, ffit, ffit2]), '-', z, d, 'xk', ax=ax)
    ax.legend(["f", "ffit", "ffit2"])
    _save(fig, close=False)

    # Example 2: integral constraint sum(u) = 0
    def Bsum(Aq):
        return np.asarray(Aq.sum()).reshape(1, -1)
    c = lsqcon(A, f, Bsum, np.array([0.0]))
    ffit3 = (A @ c).cols[0]
    print("err =")
    print(f"   {float(ffit3.sum()):.4e}")
    matlab_plot(ffit3, ax=ax)
    handles = ax.get_lines()
    ax.legend([handles[k] for k in (0, 1, 2, 4)],
              ["f", "ffit", "ffit2", "ffit3"])
    _save(fig)

    # Example 3: Gaussian basis with endpoint + integral constraints
    A = _quasi([(-5 * (x - k / 3) ** 2).exp() for k in range(-3, 4)])
    fig, ax = plt.subplots()
    matlab_plot(A, ax=ax)
    _save(fig)
    c1 = _backslash(A, f)
    ffit = (A @ c1).cols[0]

    ends = jnp.asarray([-1.0, 1.0])

    def Bg(Aq):
        return np.vstack([np.asarray(Aq(ends)),
                          np.asarray(Aq.sum()).reshape(1, -1)])
    d = np.concatenate([np.asarray(f(ends)), [0.0]])
    c2 = lsqcon(A, f, Bg, d)
    ffit2 = (A @ c2).cols[0]
    err = np.sqrt(float(ffit2.sum()) ** 2
                  + float(np.sum(np.asarray(f(ends)) - np.asarray(ffit2(ends))))
                  ** 2)
    print("err =")
    print(f"   {err:.4e}")
    fig, ax = plt.subplots()
    matlab_plot(f, '-', ffit, '-', ffit2, '-', ax=ax)
    ax.legend(["f", "ffit", "ffit2"])
    _save(fig)

    c3 = lsqcone(A, f, Bg, d)
    print("err =")
    print(f"   {np.linalg.norm(c2 - c3):.4e}")


if __name__ == "__main__":
    run()
