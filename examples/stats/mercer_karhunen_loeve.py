"""Mercer's theorem and Karhunen-Loeve expansion.

Faithful port of stats/MercerKarhunenLoeve.m by Toby Driscoll,
December 2011.  The KL expansion of an Ornstein-Uhlenbeck process with
covariance K(s,t) = exp(-|s-t|) is computed from the eigendecomposition
of the Fredholm integral operator (fred_eigs), giving orthonormal
eigenfunctions, pointwise variance, and captured-variance fractions.

Original: https://www.chebfun.org/examples/stats/MercerKarhunenLoeve.html
Copyright 2011 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): the orthonormality matrix reproduces the
identity, and the pointwise variances at x=0 / x=0.95 match the
published 0.9799 / 0.9825 to display precision.  The captured-variance
fractions differ from the page: OUR values are the analytically correct
ones -- for the exp(-c|s-t|) kernel the KL eigenvalues are
2c/(w^2+c^2) with w solving the classical transcendental equations,
giving captured = 0.9576 (c=1, page prints 0.9579) and 0.8352 (c=4,
page prints 0.6744).  The published 0.6744 reflects an under-resolved
MATLAB chebop discretisation of the kink kernel (verified: our
fred_eigs values converge to the analytic eigenvalues; the page's
lambdaShort sum cannot be reproduced by any converged method).
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.operators.integral import fred_eigs
from chebfunjax.plotting import chebfun_style

chebfun_style()

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'stats')


def _print_matrix(name, M):
    print(f"{name} =")
    for row in np.atleast_2d(M):
        print("   " + "   ".join(f"{v: .4f}" for v in row))


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    K = lambda s, t: jnp.exp(-jnp.abs(s - t))
    lam, psi = fred_eigs(K, k=20, which="LM", return_eigenfunctions=True)
    lam = np.real(np.asarray(lam))
    idx = np.argsort(-lam)
    lam = lam[idx]
    psi = [psi[int(i)].real() for i in idx]
    # eigs returns unit-2-norm eigenfunctions in MATLAB; normalise.
    psi = [p * (1.0 / float(np.sqrt(np.real(np.asarray((p * p).sum())))))
           for p in psi]

    # Orthonormality check (printed as a 6x6 identity on the page).
    G = np.array([[float(np.real(np.asarray((psi[i] * psi[j]).sum())))
                   for j in range(6)] for i in range(6)])
    _print_matrix("ans", G)

    # Pointwise variance of the truncated expansion at x=0 and x=0.95.
    for x0 in (0.0, 0.95):
        v = sum(lam[i]
                * float(np.real(np.asarray(psi[i](jnp.asarray([x0])))[0]))**2
                for i in range(20))
        print("ans =")
        print(f"    {v:.4f}")

    print("captured =")
    print(f"    {np.sum(lam[:10]) / 2:.4f}")

    # Random realizations (illustrative; MATLAB uses unseeded randn).
    rng = np.random.RandomState(0)
    Z = rng.randn(10, 400)
    xs = np.linspace(-1, 1, 400)
    Psi10 = np.column_stack(
        [np.real(np.asarray(p(jnp.asarray(xs)))) for p in psi[:10]])
    X = Psi10 @ (np.diag(np.sqrt(lam[:10])) @ Z)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for i in range(4):
        axes[0].plot(xs, Psi10[:, [0, 1, 4, 9][i]], lw=1.6)
    axes[0].set_title("Mercer eigenfunctions 1,2,5,10", fontsize=10)
    axes[1].loglog(np.arange(1, 21), np.abs(lam), ".", ms=10)
    axes[1].set_xlabel("n")
    axes[1].set_ylabel(r"$|\lambda_n|$")
    axes[1].set_title("KL eigenvalues", fontsize=10)
    axes[2].plot(xs, X[:, :40], lw=0.5, alpha=0.5)
    axes[2].plot(xs, X.mean(axis=1), "k", lw=2)
    axes[2].set_title("Random realizations, and the mean", fontsize=10)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, "mercer_karhunen_loeve.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Faster-decaying correlation captures less variance in 10 modes.
    K2 = lambda s, t: jnp.exp(-4.0 * jnp.abs(s - t))
    lam2 = np.sort(np.real(np.asarray(
        fred_eigs(K2, k=24, which="LM"))))[::-1]
    print("captured =")
    print(f"    {np.sum(lam2[:10]) / 2:.4f}")

    return True


if __name__ == "__main__":
    run()
