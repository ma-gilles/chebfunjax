"""Generalized polynomial chaos (gPC).

Demonstrates the gPC method for representing stochastic quantities
with spectral accuracy using orthogonal polynomial bases.
Translated from stats/GeneralizedPolynomialChaos.m.

Original: https://www.chebfun.org/examples/stats/GeneralizedPolynomialChaos.html
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



def hermite_polynomials(z, N):
    """Compute probabilist's Hermite polynomials H_0 ... H_N on grid z.

    Three-term recurrence: H_{n+1}(z) = z*H_n(z) - n*H_{n-1}(z).
    """
    H = np.zeros((len(z), N + 1))
    H[:, 0] = 1.0
    if N >= 1:
        H[:, 1] = z
    for n in range(1, N):
        H[:, n+1] = z * H[:, n] - n * H[:, n-1]
    return H


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- 1. Hermite polynomial basis ---
    z = np.linspace(-4, 4, 500)
    N = 5
    H = hermite_polynomials(z, N)

    # Gaussian density (weight for orthogonality)
    rho = np.exp(-z**2 / 2)
    rho /= np.trapezoid(rho, z)

    for j in range(min(4, N+1)):
        axes[0].plot(z, H[:, j], linewidth=2, label=f'H_{j}')
    axes[0].set_title('Hermite polynomials', fontsize=11)
    axes[0].set_xlabel('z'); axes[0].legend(fontsize=9)
    axes[0].set_ylim(-15, 15); axes[0].grid(True, alpha=0.3)

    # Verify orthogonality
    print("Gram matrix of Hermite polynomials (should be diagonal):")
    for i in range(min(4, N+1)):
        for j in range(min(4, N+1)):
            inner = np.trapezoid(H[:, i] * H[:, j] * rho, z)
            if i == j:
                print(f"  <H_{i}, H_{j}> = {inner:.4f}  (non-zero)")
            elif abs(inner) > 0.01:
                print(f"  <H_{i}, H_{j}> = {inner:.4f}  (should be ~0)")

    # --- 2. Strong gPC approximation: lognormal Y = exp(mu + sigma*Z) ---
    mu, sigma = 1.0, 0.5
    z_trunc = np.linspace(-10, 10, 2000)
    rho_trunc = np.exp(-z_trunc**2 / 2) / np.sqrt(2 * np.pi)

    y_vals = np.exp(mu + sigma * z_trunc)
    H_trunc = hermite_polynomials(z_trunc, N)

    # gPC coefficients: c_k = <Y, H_k> / <H_k, H_k>
    dz = z_trunc[1] - z_trunc[0]
    c = np.zeros(N + 1)
    for k in range(N + 1):
        norm_sq = np.trapezoid(H_trunc[:, k]**2 * rho_trunc, z_trunc)
        c[k] = np.trapezoid(y_vals * H_trunc[:, k] * rho_trunc, z_trunc) / norm_sq

    # Reconstruct approximation
    y_approx = H_trunc @ c

    axes[1].semilogy(z_trunc[(z_trunc >= -3) & (z_trunc <= 3)],
                     y_vals[(z_trunc >= -3) & (z_trunc <= 3)],
                     'k-', linewidth=2, label='Y = exp(μ+σZ)')
    axes[1].semilogy(z_trunc[(z_trunc >= -3) & (z_trunc <= 3)],
                     np.abs(y_approx[(z_trunc >= -3) & (z_trunc <= 3)]),
                     'r--', linewidth=2, label=f'gPC degree {N}')
    axes[1].set_title('gPC approximation of lognormal Y', fontsize=11)
    axes[1].set_xlabel('z'); axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Statistics from gPC
    mean_gpc = c[0]  # H_0 = 1
    # For lognormal: E[Y] = exp(mu + sigma^2/2)
    mean_exact = np.exp(mu + sigma**2 / 2)
    print(f"\ngPC mean = {mean_gpc:.6f}  (exact: {mean_exact:.6f})")
    print(f"gPC coefficients: {c[:6]}")

    # --- 3. Convergence with polynomial degree ---
    degrees = np.arange(1, 15)
    errors = []
    for deg in degrees:
        H_d = hermite_polynomials(z_trunc, deg)
        c_d = np.zeros(deg + 1)
        for k in range(deg + 1):
            norm_sq = np.trapezoid(H_d[:, k]**2 * rho_trunc, z_trunc)
            c_d[k] = np.trapezoid(y_vals * H_d[:, k] * rho_trunc, z_trunc) / norm_sq
        y_approx_d = H_d @ c_d
        # L2 error with weight rho
        mask = np.isfinite(y_approx_d)
        err = np.sqrt(np.trapezoid((y_vals[mask] - y_approx_d[mask])**2 * rho_trunc[mask],
                               z_trunc[mask]))
        errors.append(err)

    axes[2].semilogy(degrees, errors, 'b.-', markersize=12, linewidth=2)
    axes[2].set_title('gPC convergence: lognormal Y', fontsize=11)
    axes[2].set_xlabel('Polynomial degree'); axes[2].set_ylabel('L2 error (w/ Gaussian weight)')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Generalized Polynomial Chaos', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'generalized_polynomial_chaos.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("generalized_polynomial_chaos: done")
    return True


if __name__ == "__main__":
    run()
