"""Mercer's theorem and the Karhunen-Loeve expansion.

Faithful replica of stats/MercerKarhunenLoeve.m by Toby Driscoll
(December 2011): eigen-decomposition of the covariance kernel
K(s,t) = exp(-|s-t|) (an Ornstein-Uhlenbeck process), verification
of Mercer's theorem, and Karhunen-Loeve simulation of the process.

Original: https://www.chebfun.org/examples/stats/MercerKarhunenLoeve.html
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

from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"MercerKarhunenLoeve_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fred_eigs(kernel, nmodes, n=200):
    """Eigenpairs of the Fredholm operator with the given kernel on
    [-1,1], discretized with Clenshaw-Curtis quadrature (the
    chebfun2/eig.m approach)."""
    from chebfunjax.utils.quadrature import chebpts, chebweights
    x = np.asarray(chebpts(n))
    w = np.asarray(chebweights(n))
    Kmat = kernel(x[:, None], x[None, :])
    A = Kmat * w[None, :]
    lam, V = np.linalg.eig(A)
    idx = np.argsort(-np.abs(lam))[:nmodes]
    lam = np.real(lam[idx])
    V = np.real(V[:, idx])
    # normalize eigenfunctions in L2 with the same quadrature
    for j in range(V.shape[1]):
        nrm = np.sqrt(np.sum(w * V[:, j] ** 2))
        V[:, j] /= nrm
        if V[np.argmax(np.abs(V[:, j])), j] < 0:
            V[:, j] *= -1
    return lam, V, x, w


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    K = lambda s, t: np.exp(-np.abs(s - t))  # noqa: E731
    lam, Psi, x, w = _fred_eigs(K, 20)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for j in (0, 1, 4, 9):
        ax.plot(x, Psi[:, j], lw=2)
    ax.set_title("First four Mercer eigenfunctions", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\Psi(x)$")
    ax.grid(True)
    _save(fig)

    G = (Psi[:, :6].T * w) @ Psi[:, :6]
    print("ans =")
    for row in G:
        print("  " + "".join(f"{v:10.4f}" for v in row))

    def mercer_sum(x0):
        i0 = np.argmin(np.abs(x - x0))
        return float(np.sum(lam * Psi[i0, :] ** 2))

    print("ans =")
    print(f"    {mercer_sum(0.0):.4f}")
    print("ans =")
    print(f"    {mercer_sum(0.95):.4f}")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.loglog(np.arange(1, 21), np.abs(lam), '.', ms=10)
    ax.set_xlabel("n")
    ax.set_ylabel(r"$|\lambda_n|$")
    ax.grid(True)
    _save(fig)
    captured = np.sum(lam[:10]) / 2
    print("captured =")
    print(f"    {captured:.4f}")

    # Karhunen-Loeve realizations
    rs = np.random.RandomState(5489)
    Z = rs.randn(10, 400)
    L = np.diag(np.sqrt(lam[:10]))
    X = Psi[:, :10] @ (L @ Z)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for j in range(40):
        ax.plot(x, X[:, j], lw=0.6)
    mu = X.sum(axis=1) / 400
    ax.plot(x, mu, 'k', lw=2)
    ax.set_title("Random realizations, and the mean", fontsize=12)
    ax.grid(True)
    _save(fig)

    # empirical covariance vs kernel
    pts = np.arange(-1, 1.01, 0.05)
    idxs = [np.argmin(np.abs(x - p)) for p in pts]
    Xp = X[idxs, :]
    C = np.cov(Xp)
    S, T = np.meshgrid(pts, pts)
    fig = plt.figure(figsize=(8.6, 6.4))
    ax = fig.add_subplot(projection="3d")
    ax.plot_wireframe(S, T, C, rstride=2, cstride=2, lw=0.5)
    ax.scatter(S.ravel(), T.ravel(), K(S, T).ravel(), c='k', s=3)
    ax.set_title("Empirical covariance vs kernel", fontsize=12)
    _save(fig)

    # faster-decaying correlation captures less in 10 modes
    K4 = lambda s, t: np.exp(-4 * np.abs(s - t))  # noqa: E731
    lam4, *_ = _fred_eigs(K4, 24)
    captured4 = np.sum(lam4[:10]) / 2
    print("captured =")
    print(f"    {captured4:.4f}")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.loglog(np.arange(1, 21), np.abs(lam), '.', ms=10,
              label="exp(-|s-t|)")
    ax.loglog(np.arange(1, 25), np.abs(lam4), '.r', ms=10,
              label="exp(-4|s-t|)")
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("n")
    ax.set_ylabel(r"$|\lambda_n|$")
    _save(fig)


if __name__ == "__main__":
    run()
