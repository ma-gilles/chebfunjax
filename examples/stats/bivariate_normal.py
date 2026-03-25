"""Bivariate normal distribution.

Demonstrates computations with the bivariate normal distribution:
joint density, marginals, and conditional distributions.
Translated from stats/BivariateNormalDistribution.m.

Original: https://www.chebfun.org/examples/stats/BivariateNormalDistribution.html
Author: Alex Townsend, March 2013
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def bivariate_normal_pdf(X, Y, mu1, mu2, sigma1, sigma2, rho):
    """Bivariate normal PDF."""
    z = ((X - mu1)**2 / sigma1**2
         - 2 * rho * (X - mu1) * (Y - mu2) / (sigma1 * sigma2)
         + (Y - mu2)**2 / sigma2**2)
    norm = 2 * np.pi * sigma1 * sigma2 * np.sqrt(1 - rho**2)
    return np.exp(-z / (2 * (1 - rho**2))) / norm


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    # Parameters
    mu1, mu2 = 0.0, 0.0
    sigma1, sigma2 = 1.0, 1.0
    rho = 0.5

    # Grid on truncated domain [-5, 5]^2
    xs = np.linspace(-5, 5, 300)
    ys = np.linspace(-5, 5, 300)
    X, Y = np.meshgrid(xs, ys)

    p = bivariate_normal_pdf(X, Y, mu1, mu2, sigma1, sigma2, rho)

    # Verify integral ≈ 1
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    integral = np.sum(p) * dx * dy
    print(f"Integral of bivariate normal PDF ≈ {integral:.10f}  (exact: 1.0)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- 1. Contour plot of joint PDF ---
    levels = np.linspace(0.001, np.max(p), 15)
    axes[0].contour(X, Y, p, levels=levels)
    axes[0].set_title('Bivariate normal distribution\n(ρ=0.5)', fontsize=11)
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)

    # --- 2. Marginal distributions ---
    # Marginal of X: integrate over y
    marginal_x = np.trapezoid(p, ys, axis=0)  # sum over y rows
    marginal_y = np.trapezoid(p, xs, axis=1)  # sum over x cols

    # Exact marginals: N(mu1, sigma1) and N(mu2, sigma2)
    exact_marginal_x = np.exp(-0.5 * (xs - mu1)**2 / sigma1**2) / (sigma1 * np.sqrt(2 * np.pi))
    exact_marginal_y = np.exp(-0.5 * (ys - mu2)**2 / sigma2**2) / (sigma2 * np.sqrt(2 * np.pi))

    axes[1].plot(xs, marginal_x, 'b-', linewidth=2, label='Marginal X (numerical)')
    axes[1].plot(xs, exact_marginal_x, 'r--', linewidth=2, label='N(0,1) exact')
    axes[1].plot(ys, marginal_y, 'g-', linewidth=2, label='Marginal Y (numerical)')
    axes[1].set_title('Marginal distributions', fontsize=11)
    axes[1].set_xlabel('x'); axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    err_x = np.max(np.abs(marginal_x - exact_marginal_x))
    print(f"Max error in marginal X: {err_x:.2e}")
    assert err_x < 0.01

    # --- 3. Conditional distribution P(Y | X = x0) ---
    x0 = 1.0
    # P(Y|X=x0) = p(x0,y) / p_X(x0)  -- a Gaussian with mu_Y|X and sigma_Y|X
    mu_Y_given_X = mu2 + rho * sigma2 / sigma1 * (x0 - mu1)
    sigma_Y_given_X = sigma2 * np.sqrt(1 - rho**2)

    idx_x0 = np.argmin(np.abs(xs - x0))
    cond_pdf = p[:, idx_x0] / (np.trapezoid(p[:, idx_x0], ys))
    exact_cond = (np.exp(-0.5 * (ys - mu_Y_given_X)**2 / sigma_Y_given_X**2)
                  / (sigma_Y_given_X * np.sqrt(2 * np.pi)))

    axes[2].plot(ys, cond_pdf, 'b-', linewidth=2,
                 label=f'P(Y|X={x0}) numerical')
    axes[2].plot(ys, exact_cond, 'r--', linewidth=2,
                 label=f'N({mu_Y_given_X:.2f}, {sigma_Y_given_X:.2f})')
    axes[2].set_title(f'Conditional P(Y|X={x0})', fontsize=11)
    axes[2].set_xlabel('y'); axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    print(f"\nConditional Y|X={x0}: mu={mu_Y_given_X:.4f}, sigma={sigma_Y_given_X:.4f}")

    fig.suptitle('Bivariate Normal Distribution', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'bivariate_normal.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("bivariate_normal: done")
    return True


if __name__ == "__main__":
    run()
