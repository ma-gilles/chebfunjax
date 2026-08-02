"""Computing common roots of two bivariate functions.

Faithful replica of roots/BivariateRoots.m by Yuji Nakatsukasa,
Vanni Noferini, and Alex Townsend (February 2013): intersections of
two parametrized curves via the Bezout matrix A(y) and the DLP
(Lancaster-type) linearization, solved as a generalized eigenvalue
problem.

Original: https://www.chebfun.org/examples/roots/BivariateRoots.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.quadrature import chebpts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')

N = 25


def _chebcoeffs(vals_fun, n):
    """Chebyshev coefficients of the degree-(n-1) interpolant."""
    x = np.asarray(chebpts(n))
    f = cj.Chebfun.from_values(jnp.asarray(vals_fun(x)))
    c = np.asarray(f.funs[0].tech.coeffs)
    out = np.zeros(n)
    out[:len(c)] = c[:n]
    return out


def _spdiags_M(a, b, c, k):
    """MATLAB spdiags([a b c],[0 1 2],k,k+1) for k rows (m < n case:
    M[i, i+d] = B[i])."""
    M = np.zeros((k, k + 1))
    for i in range(k):
        M[i, i] = a[i]
        if i + 1 <= k:
            M[i, i + 1] = b[i]
        if i + 2 <= k:
            M[i, i + 2] = c[i]
    return M


def DLP(AA, v, a, b, c):
    """Direct port of the DLP subfunction of BivariateRoots.m."""
    n, m = AA.shape
    k = m // n - 1
    s = n * k
    Msmall = _spdiags_M(a, b, c, k)
    M = np.kron(Msmall, np.eye(n))
    S = np.kron(v.reshape(-1, 1), AA)
    AA = AA.copy()
    for j in range(k):
        jj = slice(n * j, n * j + n)
        AA[:, jj] = AA[:, jj].T
    T = np.kron(v.reshape(1, -1), AA.T)
    R = M.T @ S - T @ M
    X = np.zeros((s, s))
    Y = np.zeros((s, s))
    ii = slice(n, s + n)
    nn = slice(0, n)
    Y[nn, :] = R[nn, ii] / Msmall[0, 0]
    X[nn, :] = T[nn, :] / Msmall[0, 0]
    if k > 1:
        n2 = slice(n, 2 * n)
        M1np1 = M[0, n] if n < M.shape[1] else 0.0
        Y[n2, :] = (R[n2, ii] - M1np1 * Y[nn, :]
                    + Y[nn, :] @ M[:, n:s + n]) / M[n, n]
        X[n2, :] = (T[n2, :] - Y[nn, :]
                    - M1np1 * X[nn, :]) / M[n, n]
    for i in range(3, k + 1):
        ni = n * i
        jj = slice(ni - n, ni)
        j0 = slice(ni - 3 * n, ni - 2 * n)
        j1 = slice(ni - 2 * n, ni - n)
        M0 = M[ni - 2 * n - 1, ni - 1]   # M(ni-2n, ni)
        M1 = M[ni - n - 1, ni - 1]       # M(ni-n, ni)
        mm = M[ni - 1, ni - 1]       # M(ni, ni)
        Y0, Y1 = Y[j0, :], Y[j1, :]
        X0, X1 = X[j0, :], X[j1, :]
        Y[jj, :] = (R[jj, ii] - M1 * Y1 - M0 * Y0
                    + Y1 @ M[:, n:s + n]) / mm
        X[jj, :] = (T[jj, :] - Y1 - M1 * X1 - M0 * X0) / mm
    return X, Y


def run():
    os.makedirs(_IMG, exist_ok=True)

    c1 = lambda x: np.sin(np.exp(1j * np.pi * x)) \
        * np.exp(-1j * np.pi / 4)                       # noqa: E731
    c2 = lambda y: np.sin(np.exp(1j * np.pi * y)) \
        * np.exp(1j * np.pi / 3)                        # noqa: E731

    xs = np.linspace(-1, 1, 800)
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    v1c = c1(xs)
    v2c = c2(xs)
    ax.plot(v1c.real, v1c.imag, 'b-', lw=2)
    ax.plot(v2c.real, v2c.imag, 'r-', lw=2)
    ax.set_xlabel("Re", fontsize=14)
    ax.set_ylabel("Im", fontsize=14)
    ax.grid(True)
    ax.set_aspect("equal")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "BivariateRoots_repl_01.png"),
                dpi=150, bbox_inches="tight")

    n = N
    u1 = _chebcoeffs(lambda x: np.real(c1(x)), n)
    v1 = _chebcoeffs(lambda x: np.imag(c1(x)), n)
    u2 = _chebcoeffs(lambda y: np.real(c2(y)), n)
    v2 = _chebcoeffs(lambda y: np.imag(c2(y)), n)

    F = np.zeros((n, n))
    G = np.zeros((n, n))
    F[-1, :] = u1[::-1]
    F[:, -1] -= u2[::-1]
    G[-1, :] = v1[::-1]
    G[:, -1] -= v2[::-1]

    A = np.zeros((n - 1, n - 1, 2 * n - 1))
    a_in = np.concatenate([np.ones(n - 1), [2.0]]) / 2
    b_in = np.zeros(n)
    c_in = np.ones(n) / 2
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            AA = np.concatenate([[0.0], F[n - i, :]]).reshape(1, -1)
            vv = G[n - j, :]
            X, _ = DLP(AA, vv, a_in, b_in, c_in)
            if i == 1 or j == 1:
                cc = np.zeros(max(i, j))
                cc[0] = 1.0
            else:
                cc = np.zeros(i + j - 1)
                cc[-1] = 0.5
                cc[abs(i - j)] = 0.5
                cc = cc[::-1].copy()
            for kk in range(1, len(cc) + 1):
                A[:, :, A.shape[2] - kk] += (X[1:, 1:]
                                             * cc[len(cc) - kk])

    nrmA = np.linalg.norm(A[:, :, -1], 'fro')
    ii0 = 0
    for t in range(A.shape[2]):
        if np.linalg.norm(A[:, :, t], 'fro') / nrmA > 1e-20:
            ii0 = t
            break
    A = A[:, :, ii0:]
    nblk = A.shape[0]
    AA = np.concatenate([A[:, :, t] for t in range(A.shape[2])],
                        axis=1)
    rs = np.random.RandomState(0)
    v = rs.randn(nblk)
    a_o = np.concatenate([np.ones(nblk - 1), [2.0]]) / 2
    b_o = np.zeros(nblk)
    c_o = np.ones(nblk) / 2
    X, Y = DLP(AA, v, a_o, b_o, c_o)

    yvals, V = sla.eig(Y, -X)
    y = yvals
    x = V[-2, :] / V[-1, :]
    t = x
    keep_x = (np.abs(y.imag) < 1e-10) & (np.abs(y) < 1) \
        & (np.abs(t) < 1)
    xr = x[keep_x].real
    yr = y[keep_x].real
    pts = c2(yr)
    ax.plot(pts.real, pts.imag, 'xk', ms=12, mew=2)
    fig.savefig(os.path.join(_IMG, "BivariateRoots_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    errors = np.abs(c2(yr) - c1(xr))
    order = np.argsort(errors)
    print("errors =")
    for e in errors[order]:
        print(f"   {e:.6e}")


if __name__ == "__main__":
    run()
