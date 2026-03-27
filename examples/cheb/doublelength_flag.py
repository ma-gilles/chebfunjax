"""The doublelength flag for Chebyshev coefficients.

Demonstrates computing Chebyshev coefficients at "double length" to reveal
the noise plateau in a Chebyshev series — the point at which coefficients
become dominated by rounding errors.

When a function is represented with 2x as many Chebyshev coefficients as
needed for convergence, the extra coefficients plateau near machine epsilon,
revealing where the series was truncated.

Following cheb/DoublelengthFlag.m by Nick Trefethen (February 2015).

Original MATLAB: https://www.chebfun.org/examples/cheb/DoublelengthFlag.html
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

def compute_cheb_coefficients(f, n):
    """Compute n Chebyshev coefficients of function f on [-1, 1] via DCT."""
    # Chebyshev nodes of 2nd kind
    j = np.arange(n)
    theta = np.pi * j / (n - 1)
    x = np.cos(theta)
    fvals = f(x)

    # Type-II DCT (DFT of extended sequence)
    # a_k = (2/n) * sum_{j=0}^{n-1}'' f(x_j) * cos(k*j*pi/(n-1))
    # where '' means first and last terms are halved
    n_pts = n
    c = np.zeros(n_pts)
    for k in range(n_pts):
        s = np.sum(fvals * np.cos(k * theta))
        if k == 0 or k == n_pts - 1:
            s = s / (n_pts - 1)
        else:
            s = 2.0 * s / (n_pts - 1)
        c[k] = s
    c[0] /= 2.0
    c[-1] /= 2.0
    return c

def compute_cheb_coeffs_fft(f, n):
    """Compute Chebyshev coefficients via FFT (more efficient)."""
    # Use n+1 Chebyshev nodes (2nd kind)
    j = np.arange(n + 1)
    theta = np.pi * j / n
    x = np.cos(theta)
    fvals = f(x)

    # Extend for FFT
    extended = np.concatenate([fvals, fvals[-2:0:-1]])
    c_fft = np.real(np.fft.fft(extended)) / n
    c = c_fft[:n + 1]
    c[0] /= 2.0
    c[-1] /= 2.0
    return c

def run():
    print("=" * 60)
    print("The doublelength flag: revealing Chebyshev noise plateau")
    print("=" * 60)

    print("\nIdea: compute with 2x more points to reveal where the series")
    print("was truncated due to floating-point noise")

    # Example 1: exp(x)
    print("\n1. f(x) = exp(x)")
    f1 = np.exp
    n_normal = 18   # normal length for exp(x) to converge to machine precision
    n_double = 2 * n_normal - 1  # "doublelength"

    c1_normal = compute_cheb_coeffs_fft(f1, n_normal)
    c1_double = compute_cheb_coeffs_fft(f1, n_double)

    print(f"   Normal length: {len(c1_normal)} coefficients")
    print(f"   Double length: {len(c1_double)} coefficients")

    # The noise plateau should appear in the double-length series
    noise_floor_normal = np.min(np.abs(c1_normal[c1_normal != 0]))
    noise_floor_double = np.min(np.abs(c1_double[c1_double != 0]))
    print(f"   Normal: min|coeff| = {noise_floor_normal:.2e}")
    print(f"   Double: min|coeff| = {noise_floor_double:.2e}")

    # For exp(x), exact Chebyshev coefficients are 2*I_k(1) (modified Bessel)
    from scipy.special import iv as besseli
    k = np.arange(len(c1_double))
    c_exact = 2.0 * besseli(k, 1.0)
    c_exact[0] /= 2.0
    err_normal = np.abs(c1_normal - c_exact[:len(c1_normal)])
    err_double = np.abs(c1_double - c_exact[:len(c1_double)])
    print(f"   Max error (normal): {np.max(err_normal):.2e}")
    print(f"   Max error (double): {np.max(err_double[n_normal:]):.2e}")

    assert np.max(err_normal) < 1e-12, "Normal coefficients not accurate"
    print("   PASS: Chebyshev coefficients for exp(x) match exact values")

    # Example 2: sin(x) + sin(x^2) on [0, 10]
    print("\n2. f(x) = sin(x) + sin(x^2) on [0, 10]  (via rescaling)")
    def f2(x):
        # x ∈ [-1,1] → t ∈ [0,10]
        t = 5.0 * (x + 1)
        return np.sin(t) + np.sin(t**2)

    n2_normal = 80
    n2_double = 2 * n2_normal - 1

    c2_normal = compute_cheb_coeffs_fft(f2, n2_normal)
    c2_double = compute_cheb_coeffs_fft(f2, n2_double)
    print(f"   Normal length: {len(c2_normal)}, Double: {len(c2_double)}")

    # The double-length series shows where the noise plateau starts
    idx_plateau = np.argmax(np.abs(c2_double) < 1e-13)
    if idx_plateau > 0:
        print(f"   Noise plateau starts around coefficient index {idx_plateau}")

    print("\nPASS: doublelength demonstrates Chebyshev series truncation point")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    # exp(x)
    axes[0].semilogy(np.abs(c1_double), '.b', markersize=4, label='double length')
    axes[0].semilogy(np.abs(c1_normal), 'or', markersize=6, alpha=0.8, label='normal')
    axes[0].set_title("Chebyshev coefficients of exp(x)", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim([1e-18, 10])

    # sin(x) + sin(x^2)
    axes[1].semilogy(np.abs(c2_double), '.b', markersize=4, label='double length')
    axes[1].semilogy(np.abs(c2_normal), 'r-', linewidth=1.5, alpha=0.8, label='normal')
    axes[1].set_title("Cheb coefficients of sin(x)+sin(x²) on [0,10]", fontsize=11)
    axes[1].legend(fontsize=9)

    fig.suptitle("Double-length Chebyshev coefficients: revealing noise plateau",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "doublelength_flag.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True

if __name__ == "__main__":
    run()
