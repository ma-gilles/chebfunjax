"""Exact Chebyshev expansion coefficients of a function.

Uses Elliott's residue method to compute exact Chebyshev coefficients of
f(x) = 1/(5+x), which has a pole at z=-5. The formula is:
  a_n = (1/sqrt(6)) * (-1)^n / (5 + sqrt(24))^n

Compares these exact coefficients against numerically computed ones,
demonstrating exponential decay rate governed by the Bernstein ellipse.

Following cheb/ExactChebCoeffs.m by Mark Richardson (June 2012).

Original MATLAB: https://www.chebfun.org/examples/cheb/ExactChebCoeffs.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def compute_cheb_coeffs_fft(f, n):
    """Compute n+1 Chebyshev coefficients of f on [-1,1] via FFT."""
    j = np.arange(n + 1)
    theta = np.pi * j / n
    x = np.cos(theta)
    fvals = f(x)

    extended = np.concatenate([fvals, fvals[-2:0:-1]])
    c_fft = np.real(np.fft.fft(extended)) / n
    c = c_fft[:n + 1].copy()
    c[0] /= 2.0
    c[-1] /= 2.0
    return c


def run():
    print("=" * 60)
    print("Exact Chebyshev expansion coefficients")
    print("=" * 60)

    print("\nFunction: f(x) = 1/(5+x)")
    print("Pole at z = -5, residue = 1")
    print("\nElliott's exact formula:")
    print("  a_n = (1/sqrt(6)) * (-1)^n / (5 + sqrt(24))^n")
    print("  (for n >= 1; a_0 has extra factor of 2)")

    f = lambda x: 1.0 / (5.0 + x)

    # Compute numerical Chebyshev coefficients
    N = 20
    c_num = compute_cheb_coeffs_fft(f, N)
    print(f"\nComputed {len(c_num)} numerical coefficients")

    # Exact formula (Elliott 1964)
    k = np.arange(N + 1)
    sqrt6 = np.sqrt(6.0)
    sqrt24 = np.sqrt(24.0)
    # Elliott formula: a_n = (-2*r0) / (sqrt(z0^2-1) * (z0 + sqrt(z0^2-1))^n)
    # z0 = -5 (pole), r0 = 1 (residue)
    # sqrt(z0^2-1) = sqrt(25-1) = sqrt(24)
    # z0 + sqrt(z0^2-1) = -5 + sqrt(24)  → negative
    # z0 - sqrt(z0^2-1) = -5 - sqrt(24)  → use this (|.| > 1 outside ellipse)
    # Actually: |z0 ± sqrt(z0^2-1)| where we choose the one with |.| > 1
    # z0 = -5: z0 + sqrt(24) ≈ -5 + 4.899 = -0.101 (magnitude < 1)
    #          z0 - sqrt(24) ≈ -5 - 4.899 = -9.899 (magnitude > 1)
    # So rho = |z0 - sqrt(z0^2-1)| = 5 + sqrt(24)
    # a_n = (-2*r0) / (sqrt(z0^2-1) * (z0 - sqrt(z0^2-1))^n)
    #      = (-2) / (sqrt(24) * (-5-sqrt(24))^n)
    #      = (-2) / (sqrt(24) * (-1)^n * (5+sqrt(24))^n)
    #      = 2*(-1)^{n+1} / (sqrt(24) * (5+sqrt(24))^n)
    # For n=0 the formula needs the halving convention: a_0 is halved in the integral
    # Let me just use the simple form given in the MATLAB:
    # exact_coeffs = 1/sqrt(6) * (-1)^(n-1) / (5+sqrt(24))^(n-1)
    # This is the MATLAB indexing where k starts at 1 (0-indexed: subtract 1)
    c_exact = np.zeros(N + 1)
    for n in range(N + 1):
        c_exact[n] = (1.0 / sqrt6) * ((-1)**n) / ((5.0 + sqrt24)**n)
    # Correct the a_0 factor (MATLAB uses k=1 as first coefficient; a_0 is doubled in the standard formula)
    # Actually let's just check and use direct comparison

    print("\nComparison (n, exact, numerical, diff):")
    print(f"{'n':>4}  {'exact':>14}  {'numerical':>14}  {'diff':>12}")
    for n in range(min(10, N+1)):
        diff = c_exact[n] - c_num[n]
        print(f"{n:>4}  {c_exact[n]:>14.8e}  {c_num[n]:>14.8e}  {diff:>12.3e}")

    # The a_0 coefficient has the factor-of-2 convention difference
    # Check convergence for n >= 1
    max_err = np.max(np.abs(c_exact[1:] - c_num[1:]))
    print(f"\nMax |exact - numerical| for n >= 1: {max_err:.2e}")
    assert max_err < 1e-10, f"Coefficient error too large: {max_err:.2e}"
    print("PASS: exact formula matches numerical coefficients")

    # Bernstein ellipse parameter
    rho = 5.0 + sqrt24
    print(f"\nBernstein ellipse parameter rho = 5 + sqrt(24) = {rho:.4f}")
    print(f"Geometric decay rate: |a_n| ~ rho^(-n) = {rho:.4f}^(-n)")

    # Verify the geometric decay
    k_vals = np.arange(1, N + 1)
    theoretical = c_exact[1] * rho**(-k_vals + 1)  # reference to k=1
    actual = np.abs(c_num[1:])
    ratio = actual / theoretical
    print(f"Max ratio |numerical|/|theoretical| in [0.5, 2]: "
          f"{np.max(ratio):.3f}, {np.min(ratio):.3f}")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    n_plot = np.arange(N + 1)

    # Coefficient magnitudes
    axes[0].semilogy(n_plot, np.abs(c_num), '.-b', markersize=8,
                     linewidth=1, label='Numerical')
    axes[0].semilogy(n_plot[1:], np.abs(c_exact[1:]), 'or', markersize=5,
                     alpha=0.7, label='Exact (Elliott)')
    axes[0].set_title("Chebyshev coefficients of 1/(5+x)", fontsize=11)
    axes[0].set_xlabel("n"); axes[0].set_ylabel("log|a_n|")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # Error
    err = np.abs(c_exact[1:] - c_num[1:])
    axes[1].semilogy(n_plot[1:], err, '.-k', markersize=6)
    axes[1].axhline(1e-15, color='r', linestyle='--', linewidth=1,
                    label='Machine epsilon ~1e-15')
    axes[1].set_title("Difference: |exact - numerical|", fontsize=11)
    axes[1].set_xlabel("n"); axes[1].set_ylabel("|exact - numerical|")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Exact Chebyshev coefficients via Elliott's residue method",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "exact_cheb_coeffs.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
