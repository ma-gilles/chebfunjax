"""Condition numbers and quasimatrices.

Demonstrates condition numbers of various function bases and quasimatrix QR,
following linalg/CondNos.m by Nick Trefethen (September 2010) and
linalg/QuasiQR.m by Hale & Trefethen (March 2022).

Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.utils.quadrature import chebpts, chebweights
from chebfunjax.utils.polynomials import chebpoly, legpoly


def eval_Tj(j, xs_np):
    """Evaluate Chebyshev polynomial T_j at numpy array xs_np."""
    coeffs = chebpoly(j)
    f = cj.chebfun.from_coeffs(coeffs)
    return np.array([float(f(jnp.array(float(x)))) for x in xs_np])


def eval_Pj(j, xs_np):
    """Evaluate Legendre polynomial P_j at numpy array xs_np."""
    from scipy.special import legendre
    P = legendre(j)
    return P(xs_np)


def run():
    print("=" * 60)
    print("Condition numbers and quasimatrix QR")
    print("=" * 60)

    # --- Condition numbers of monomial vs Chebyshev basis ---
    print("\nCondition numbers of Vandermonde vs. Chebyshev matrices:")
    for n in [5, 10, 15]:
        xs_n = np.array(chebpts(n))
        V = np.vander(xs_n, n, increasing=True)
        cond_V = np.linalg.cond(V)
        T = np.column_stack([eval_Tj(j, xs_n) for j in range(n)])
        cond_T = np.linalg.cond(T)
        print(f"  n={n:2d}: cond(Vandermonde) = {cond_V:.2e},  cond(Chebyshev) = {cond_T:.2e}")
        assert cond_T < cond_V + 1  # Chebyshev not worse

    # --- Quasimatrix inner products ---
    print("\nQuasimatrix Gram matrix (Legendre polynomials, n=6):")
    n_qm = 6
    n_pts = 200
    xs_pts = np.array(chebpts(n_pts))
    ws_pts = np.array(chebweights(n_pts))

    # Build matrix of Legendre polynomial evaluations
    P = np.column_stack([eval_Pj(j, xs_pts) for j in range(n_qm)])

    # Gram matrix = P^T W P (Clenshaw-Curtis weights)
    G = P.T @ (ws_pts[:, None] * P)
    print("Gram matrix structure (orthogonality check):")
    # Check orthogonality (off-diagonal ~ 0)
    max_off = np.max(np.abs(G - np.diag(np.diag(G))))
    print(f"Max off-diagonal element: {max_off:.2e}  (expected: ~0)")
    assert max_off < 0.05, f"Orthogonality failed: max off-diag = {max_off}"

    # Diagonal elements should be positive
    for j in range(n_qm):
        print(f"  <P_{j},P_{j}> = {G[j,j]:.8f}  (positive: {G[j,j] > 0})")
        assert G[j, j] > 0, f"P_{j} has non-positive norm"

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Condition number growth
    ns_plot = list(range(3, 15))
    cond_vand = []
    cond_cheb = []
    for n in ns_plot:
        xn = np.array(chebpts(n))
        Vn = np.vander(xn, n, increasing=True)
        Tn = np.column_stack([eval_Tj(j, xn) for j in range(n)])
        cond_vand.append(np.linalg.cond(Vn))
        cond_cheb.append(np.linalg.cond(Tn))

    axes[0].semilogy(ns_plot, cond_vand, 'b-o', markersize=4, label='Monomial (Vandermonde)')
    axes[0].semilogy(ns_plot, cond_cheb, 'r-s', markersize=4, label='Chebyshev')
    axes[0].set_title("Condition numbers vs. n", fontsize=12)
    axes[0].set_xlabel("n"); axes[0].set_ylabel("Condition number")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Gram matrix heatmap
    im = axes[1].imshow(np.abs(G), cmap='Blues', aspect='auto')
    axes[1].set_title("Gram matrix |G| (Legendre, n=6)", fontsize=12)
    axes[1].set_xlabel("j"); axes[1].set_ylabel("i")
    fig.colorbar(im, ax=axes[1])

    fig.suptitle("Condition numbers and orthogonality", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "condition_numbers.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
