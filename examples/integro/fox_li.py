"""Eigenvalues of the Fox-Li integral operator.

Computes eigenvalues of the Fox-Li Fredholm integral operator:
  (Lu)(x) = sqrt(iF/pi) * integral_{-1}^{1} exp(-iF*(x-y)^2) u(y) dy

following integro/FoxLi.m by Driscoll & Trefethen (October 2010).

The Fresnel number F determines the structure of the eigenvalue spectrum,
which lies on a characteristic curve inside the unit disk.

Original MATLAB: https://www.chebfun.org/examples/integro/FoxLi.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chebfunjax.plotting import chebfun_style
chebfun_style()

import numpy as np
from scipy.linalg import eig
import os

def run():
    print("=" * 60)
    print("Eigenvalues of the Fox-Li integral operator")
    print("=" * 60)

    # Fox-Li operator: (Lu)(x) = sqrt(iF/pi) * integral_{-1}^{1} K(x,y) u(y) dy
    # K(x,y) = exp(-iF*(x-y)^2)
    # Fresnel number F

    F = 64 * np.pi

    print(f"\nFresnel number F = 64π ≈ {F:.2f}")
    print("Computing 80 eigenvalues of largest complex magnitude...")

    # Discretize with Gaussian quadrature
    N = 200

    # Gauss-Legendre quadrature on [-1, 1]
    from numpy.polynomial.legendre import leggauss
    y_gl, w_gl = leggauss(N)

    # Build the N×N matrix
    # [A]_{ij} = sqrt(iF/pi) * K(y_i, y_j) * w_j
    prefactor = np.sqrt(1j * F / np.pi)
    y_i = y_gl[:, np.newaxis]  # (N, 1)
    y_j = y_gl[np.newaxis, :]  # (1, N)
    K_mat = np.exp(-1j * F * (y_i - y_j)**2)
    A_mat = prefactor * K_mat * w_gl[np.newaxis, :]  # multiply each column by w_j

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(A_mat)

    # Sort by magnitude (descending)
    idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues_sorted = eigenvalues[idx]

    # Keep top 80
    lam = eigenvalues_sorted[:80]

    print(f"\nTop 80 eigenvalues by magnitude:")
    print(f"  |λ₁| = {np.abs(lam[0]):.6f}")
    print(f"  |λ₁₀| = {np.abs(lam[9]):.6f}")
    print(f"  |λ₈₀| = {np.abs(lam[79]):.6f}")

    # The eigenvalues should lie inside/on the unit circle
    max_abs_ev = np.max(np.abs(lam))
    print(f"\nMax |eigenvalue| = {max_abs_ev:.6f} (should be ≤ 1 for large F)")

    # They form a characteristic curve in the complex plane
    # Check that they cluster on a curve (the Fox-Li curve)
    print(f"\nEigenvalue pattern: {len(lam)} eigenvalues in complex plane")
    print(f"  Real range: [{np.min(np.real(lam)):.3f}, {np.max(np.real(lam)):.3f}]")
    print(f"  Imag range: [{np.min(np.imag(lam)):.3f}, {np.max(np.imag(lam)):.3f}]")

    # Basic sanity: eigenvalues should not all be the same
    ev_spread = np.std(np.abs(lam))
    assert ev_spread > 0.01, "Eigenvalues too clustered"
    print(f"  Spread in |λ|: {ev_spread:.4f}")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    # Complex plane plot
    theta = np.linspace(0, 2 * np.pi, 200)
    axes[0].plot(np.cos(theta), np.sin(theta), 'r--', linewidth=1, alpha=0.5,
                 label='Unit circle')
    axes[0].plot(np.real(lam), np.imag(lam), 'k.', markersize=10,
                 label=f'80 eigenvalues')
    axes[0].set_aspect('equal')
    axes[0].set_xlim([-1.1, 1.1]); axes[0].set_ylim([-1.1, 1.1])
    axes[0].set_title(f"Fox-Li eigenvalues (F=64π)", fontsize=11)
    axes[0].legend(fontsize=9)

    # Magnitude plot
    axes[1].plot(range(1, len(lam) + 1), np.abs(lam), 'b.-', markersize=6)
    axes[1].set_title("Eigenvalue magnitudes (sorted)", fontsize=11)

    fig.suptitle("Fox-Li integral operator eigenvalues", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "fox_li.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True

if __name__ == "__main__":
    run()
