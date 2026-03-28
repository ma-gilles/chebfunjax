"""Mercer's theorem and Karhunen-Loeve expansion.

Demonstrates the Karhunen-Loeve expansion of stochastic processes via
eigendecomposition of integral kernels. Translated from
stats/MercerKarhunenLoeve.m.

Original: https://www.chebfun.org/examples/stats/MercerKarhunenLoeve.html
Author: Toby Driscoll, December 2011
"""

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

def discretize_kernel(K, xs):
    """Discretize integral kernel K(s,t) on grid xs."""
    dx = xs[1] - xs[0]
    S, T = np.meshgrid(xs, xs, indexing='ij')
    K_mat = K(S, T) * dx
    return K_mat

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3)

    # Exponential covariance kernel: K(s,t) = exp(-|s-t|)
    K1 = lambda s, t: np.exp(-np.abs(s - t))

    # Discretize on [-1, 1]
    n_grid = 200
    xs = np.linspace(-1, 1, n_grid)
    K_mat = discretize_kernel(K1, xs)

    # Eigendecomposition (symmetric positive definite)
    eigenvalues, eigenvectors = np.linalg.eigh(K_mat)
    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep positive eigenvalues
    pos_mask = eigenvalues > 1e-10
    eigenvalues = eigenvalues[pos_mask]
    eigenvectors = eigenvectors[:, pos_mask]

    # Normalize eigenvectors to unit L2 norm over [-1,1]
    dx = xs[1] - xs[0]
    for j in range(eigenvectors.shape[1]):
        norm = np.sqrt(np.trapezoid(eigenvectors[:, j]**2, xs))
        eigenvectors[:, j] /= norm

    print(f"Number of positive eigenvalues: {len(eigenvalues)}")
    print(f"First 5 eigenvalues: {eigenvalues[:5]}")

    # Plot first 4 eigenfunctions
    for j, (color, lw) in enumerate(zip(['b', 'r', 'g', 'm'], [2.5, 2, 1.5, 1.5])):
        if j < eigenvectors.shape[1]:
            axes[0].plot(xs, eigenvectors[:, j], color=color,
                         linewidth=lw, label=f'ψ_{j+1}')
    axes[0].set_title('First 4 Mercer eigenfunctions', fontsize=11)
    axes[0].legend(fontsize=9)

    # Eigenvalue decay
    n_eig = min(20, len(eigenvalues))
    axes[1].loglog(np.arange(1, n_eig+1), eigenvalues[:n_eig], '.b', markersize=12)
    axes[1].loglog(np.arange(1, n_eig+1), 2.0 / (np.arange(1, n_eig+1))**2,
                   '-r', linewidth=2, label='O(n^{-2})')
    axes[1].set_title('Eigenvalue decay: O(n^{-2})', fontsize=11)
    axes[1].legend(fontsize=9)

    # KL expansion: generate random realizations
    n_modes = min(10, len(eigenvalues))
    rng = np.random.default_rng(42)
    n_samples = 30

    # Variance captured by first 10 modes
    total_variance = float(np.trapezoid(np.diag(K1(xs[:, None], xs[None, :])), xs))
    captured = np.sum(eigenvalues[:n_modes]) / total_variance * 100
    print(f"Variance captured by first {n_modes} modes: {captured:.1f}%")

    # Generate realizations: X(t) = sum_j sqrt(lambda_j) * psi_j(t) * Z_j
    Z = rng.standard_normal((n_modes, n_samples))
    sqrt_lambda = np.sqrt(eigenvalues[:n_modes])
    realizations = eigenvectors[:, :n_modes] @ (np.diag(sqrt_lambda) @ Z)

    for i in range(min(15, n_samples)):
        axes[2].plot(xs, realizations[:, i], color='#0072BD', linestyle='-', linewidth=0.5, alpha=0.4)
    mean_realization = np.mean(realizations, axis=1)
    axes[2].plot(xs, mean_realization, 'k-', linewidth=2, label='Mean')
    axes[2].set_title(f'KL realizations ({captured:.0f}% var in {n_modes} modes)',
                      fontsize=10)
    axes[2].legend(fontsize=9)

    fig.suptitle("Mercer's Theorem and Karhunen-Loeve Expansion", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'mercer_karhunen_loeve.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("mercer_karhunen_loeve: done")
    return True

if __name__ == "__main__":
    run()
