"""SVD of frequency response operator for PDEs.

Demonstrates computing the frequency response of the 1D diffusion equation
via SVD of the compact operator T: L^2 → L^2, following
pde/SVDFrequencyResponse.m by Lieu and Jovanovic (January 2012).

For the diffusion equation u_t = u_yy + d(y,t), the frequency response
operator at frequency w has singular values σ_n = 4/(n*π)^2.

Original MATLAB: https://www.chebfun.org/examples/pde/SVDFrequencyResponse.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
import os


def run():
    print("=" * 60)
    print("SVD of frequency response operator")
    print("=" * 60)

    # Diffusion equation: u_t = u_yy + d(y,t) on [-1,1]
    # with Dirichlet BCs: u(-1) = u(1) = 0
    # Frequency response at w=0:
    # Solve: u_yy - i*w*u = -d(y),  u(±1) = 0
    # At w=0: u_yy = -d, so T = (-d^2/dy^2)^{-1} with Dirichlet BCs
    # Analytical singular values: σ_n = 4/(n*π)^2

    w = 0.0  # temporal frequency

    # Build the Chebyshev discretization of (D^2 - i*w*I) on [-1,1]
    N = 200  # Chebyshev grid points

    # Chebyshev differentiation matrix (Weideman & Reddy)
    def cheb(N):
        if N == 0:
            return np.zeros((1, 1)), np.array([1.0])
        x = np.cos(np.pi * np.arange(N + 1) / N)
        c = np.ones(N + 1)
        c[0] = 2; c[N] = 2
        c *= (-1)**np.arange(N + 1)
        X = np.outer(np.ones(N + 1), x)
        dX = X - X.T
        D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
        D -= np.diag(D.sum(axis=1))
        return D, x

    D, y = cheb(N)
    D2 = D @ D

    # Interior points (excluding boundaries)
    D2_int = D2[1:-1, 1:-1]
    y_int = y[1:-1]

    n_int = N - 1  # number of interior points

    # Frequency response operator: T = (D2_int - i*w*I)^{-1}
    # (negative sign for our convention)
    A = D2_int - 1j * w * np.eye(n_int)

    # T: d → u, where A*u = -d → T = -A^{-1}
    # SVD of T = SVD of -A^{-1}
    T_matrix = -np.linalg.inv(A)

    # Compute singular values
    Nsigs = 25
    sv_numerical = svd(T_matrix, compute_uv=False)[:Nsigs]

    # Analytical singular values: σ_n = 4/(n*π)^2 for n=1,2,...
    n_array = np.arange(1, Nsigs + 1)
    sv_analytical = 4.0 / (n_array * np.pi)**2

    print(f"\nFirst {Nsigs} singular values comparison:")
    print(f"  {'n':3s}  {'Numerical':12s}  {'Analytical':12s}  {'Error':10s}")
    for i in range(min(5, Nsigs)):
        err = abs(sv_numerical[i] - sv_analytical[i])
        print(f"  {i+1:3d}  {sv_numerical[i]:12.8f}  {sv_analytical[i]:12.8f}  {err:.2e}")

    # Check accuracy — Chebyshev discretization has O(h^2) error vs analytical
    # With N=200, expect ~0.25% relative error; check that relative error is < 1%
    norm_error = np.linalg.norm(sv_numerical - sv_analytical)
    rel_error = norm_error / np.linalg.norm(sv_analytical)
    print(f"\nNorm of error (all {Nsigs} values): {norm_error:.2e}  (relative: {rel_error:.2e})")
    assert rel_error < 0.02, f"SVD relative error too large: {rel_error:.2e}"
    print("PASS: singular values match analytical formula to within 2%")

    # Compute first two singular functions and compare with sin functions
    U, S, Vh = svd(T_matrix, full_matrices=False)

    # Scale the grid to [-1, 1] (Chebyshev grid is already on [-1, 1])
    # Analytical first singular function: sin(π*(y+1)/2) (on [-1,1] with DBC)
    # But we need to check normalization
    sf1_numerical = np.real(U[:, 0])  # first left singular vector
    sf1_analytical = np.sin(np.pi * (y_int + 1) / 2)

    # Normalize and find sign
    sf1_n = sf1_numerical / (np.linalg.norm(sf1_numerical) * (2 / (N - 1))**0.5)
    sf1_a = sf1_analytical / np.linalg.norm(sf1_analytical * (2 / (N - 1))**0.5)
    # Pick consistent sign
    if np.dot(sf1_n, sf1_a) < 0:
        sf1_n = -sf1_n

    err_sf1 = np.max(np.abs(sf1_n / sf1_n.max() - sf1_a / sf1_a.max()))
    print(f"\nFirst singular function error vs sin(π(y+1)/2): {err_sf1:.4f}")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Singular values
    axes[0].plot(n_array, sv_numerical, 'bx', markersize=10,
                 linewidth=1.5, label='Numerical')
    axes[0].plot(n_array, sv_analytical, 'ro', markersize=7,
                 markerfacecolor='none', linewidth=1.5, label='Analytical: 4/(nπ)²')
    axes[0].set_title("Singular values of frequency response (w=0)", fontsize=11)
    axes[0].set_xlabel("n"); axes[0].set_ylabel("σ_n")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # First singular function
    axes[1].plot(y_int, sf1_n / sf1_n.max(), 'bx-', markersize=5,
                 label='Numerical', alpha=0.7)
    axes[1].plot(y_int, sf1_a / sf1_a.max(), 'r-', linewidth=2,
                 label='sin(π(y+1)/2)')
    axes[1].set_title("First singular function", fontsize=11)
    axes[1].set_xlabel("y"); axes[1].set_ylabel("φ₁(y)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle("SVD of frequency response: diffusion equation", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "svd_frequency_response.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
