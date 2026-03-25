"""Condition numbers of Chebyshev vs Vandermonde matrices.

Explores how Chebyshev interpolation gives much better-conditioned matrices
than Vandermonde matrices for equispaced nodes. Based on Chebfun example
linalg/CondVandermonde.m.

Original: https://www.chebfun.org/examples/linalg/CondVandermonde.html
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


def vandermonde_matrix(nodes):
    """Vandermonde matrix for the given nodes."""
    nodes = np.asarray(nodes)
    n = len(nodes)
    V = np.ones((n, n))
    for j in range(1, n):
        V[:, j] = V[:, j - 1] * nodes
    return V


def chebyshev_vandermonde(nodes):
    """Chebyshev-Vandermonde matrix: V_{kj} = T_j(x_k)."""
    nodes = np.asarray(nodes)
    n = len(nodes)
    V = np.ones((n, n))
    if n > 1:
        V[:, 1] = nodes
    for j in range(2, n):
        V[:, j] = 2 * nodes * V[:, j - 1] - V[:, j - 2]
    return V


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/linalg')
    os.makedirs(outdir, exist_ok=True)

    nn = list(range(3, 26))
    cond_vander_equi = []
    cond_vander_cheb = []
    cond_chebv_cheb = []

    for n in nn:
        # Equispaced nodes
        equi = np.linspace(-1, 1, n)
        # Chebyshev-2 nodes
        k = np.arange(n)
        cheb = np.cos(np.pi * k / (n - 1))[::-1]  # ascending

        V_equi = vandermonde_matrix(equi)
        V_cheb = vandermonde_matrix(cheb)
        CV_cheb = chebyshev_vandermonde(cheb)

        cond_vander_equi.append(np.linalg.cond(V_equi))
        cond_vander_cheb.append(np.linalg.cond(V_cheb))
        cond_chebv_cheb.append(np.linalg.cond(CV_cheb))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(nn, cond_vander_equi, 'r.-', markersize=8, linewidth=1.5,
                label='Vandermonde (equispaced)')
    ax.semilogy(nn, cond_vander_cheb, 'b.-', markersize=8, linewidth=1.5,
                label='Vandermonde (Chebyshev nodes)')
    ax.semilogy(nn, cond_chebv_cheb, 'g.-', markersize=8, linewidth=1.5,
                label='Chebyshev-Vandermonde (Cheb nodes)')
    ax.set_xlabel('$n$ (number of nodes)', fontsize=12)
    ax.set_ylabel('Condition number', fontsize=12)
    ax.set_title('Condition numbers of Vandermonde matrices', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'condition_numbers.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("Condition numbers at n=20:")
    i20 = nn.index(20)
    print(f"  Vandermonde (equispaced): {cond_vander_equi[i20]:.2e}")
    print(f"  Vandermonde (Cheb nodes): {cond_vander_cheb[i20]:.2e}")
    print(f"  Cheb-Vandermonde (Cheb):  {cond_chebv_cheb[i20]:.2e}")

    # The Chebyshev-Vandermonde with Chebyshev nodes should be well-conditioned
    assert cond_chebv_cheb[i20] < 1e4, "Chebyshev-Vandermonde condition too large"
    # The Vandermonde with equispaced nodes should be exponentially ill-conditioned
    assert cond_vander_equi[i20] > 1e8, "Vandermonde equispaced should be ill-conditioned"

    print("condition_numbers: done")
    return True


if __name__ == "__main__":
    run()
