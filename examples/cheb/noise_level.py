"""Estimating the noise level of a Chebyshev series.

Demonstrates two methods for estimating the rounding error (noise) plateau
in a Chebyshev series:
1. extend1: pad with zeros, forward+inverse transform
2. extend2: also perturb sample points by machine epsilon

The second method accounts for "horizontal" perturbations (in the argument x),
and better estimates the noise floor for rapidly varying functions.

Following cheb/NoiseLevel.m by Nick Trefethen (6 July 2016).

Original MATLAB: https://www.chebfun.org/examples/cheb/NoiseLevel.html
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


def extend1(c):
    """Estimate noise by padding with zeros, forward + inverse transform."""
    n = len(c)
    m = 4 * n
    # Pad with zeros
    c_padded = np.zeros(m)
    c_padded[:n] = c

    # Convert to function values (inverse Chebyshev transform)
    # then back to coefficients — rounding errors fill the noise floor
    j = np.arange(m)
    theta = np.pi * j / (m - 1)
    # Inverse: fvals = sum_k c_k T_k(x_j)
    x = np.cos(theta)
    fvals = np.zeros(m)
    for k in range(n):
        fvals += c_padded[k] * np.cos(k * theta)
    # Forward: c_out = DCT(fvals)
    c_out = np.zeros(m)
    for k in range(m):
        s = np.sum(fvals * np.cos(k * theta))
        if k == 0 or k == m - 1:
            c_out[k] = s / (m - 1)
        else:
            c_out[k] = 2.0 * s / (m - 1)
    c_out[0] /= 2.0
    c_out[-1] /= 2.0
    return c_out


def extend2(c):
    """Estimate noise accounting for x-perturbations (horizontal errors)."""
    n = len(c)
    m = 4 * n
    eps_m = np.finfo(float).eps

    c_padded = np.zeros(m)
    c_padded[:n] = c

    j = np.arange(m)
    theta = np.pi * j / (m - 1)
    x = np.cos(theta)
    # Evaluate
    fvals = np.zeros(m)
    for k in range(n):
        fvals += c_padded[k] * np.cos(k * theta)

    # Horizontal perturbation: blend with neighboring values
    # grid spacing in x is O(pi/(m-1))
    dx = np.pi / (m - 1)
    # Perturb x slightly in one direction
    perturb = eps_m / dx
    fvals_perturbed = fvals.copy()
    for i in range(1, m - 1):
        # Blend with neighbors based on machine epsilon / grid spacing
        alpha = eps_m / (x[i-1] - x[i+1]) if abs(x[i-1] - x[i+1]) > 1e-15 else 0
        fvals_perturbed[i] += alpha * (fvals[i-1] - fvals[i+1]) / 2.0

    # Forward transform
    c_out = np.zeros(m)
    for k in range(m):
        s = np.sum(fvals_perturbed * np.cos(k * theta))
        if k == 0 or k == m - 1:
            c_out[k] = s / (m - 1)
        else:
            c_out[k] = 2.0 * s / (m - 1)
    c_out[0] /= 2.0
    c_out[-1] /= 2.0
    return c_out


def run():
    print("=" * 60)
    print("Estimating the noise level of a Chebyshev series")
    print("=" * 60)

    print("\nMethod: pad Chebyshev series with zeros, then transform forward/backward")
    print("Rounding errors in the round-trip reveal the noise floor.")
    print("extend1: vertical errors only")
    print("extend2: also accounts for horizontal (x-perturbation) errors")

    # Main example: cos(100*exp(x))
    print("\n1. f(x) = cos(100*exp(x))")
    f1 = lambda x: np.cos(100 * np.exp(x))

    # Compute a doublelength series
    n_full = 280
    c_full = compute_cheb_coeffs_fft(f1, n_full)
    print(f"   Full series length: {len(c_full)}")

    # Truncate to first 200 (some genuine, some noise)
    n_trunc = 180
    c_trunc = c_full[:n_trunc + 1].copy()

    # Apply extend1
    c_ext1 = extend1(c_trunc)
    noise1 = np.max(np.abs(c_ext1[n_trunc + 1:]))
    print(f"   extend1 noise estimate: {noise1:.2e}")

    # Apply extend2
    c_ext2 = extend2(c_trunc)
    noise2 = np.max(np.abs(c_ext2[n_trunc + 1:]))
    print(f"   extend2 noise estimate: {noise2:.2e}")

    # The noise plateau in the actual series
    coeff_magnitudes = np.abs(c_trunc)
    noise_plateau = np.mean(coeff_magnitudes[-20:])  # last 20 coefficients
    print(f"   Actual noise plateau (last 20 coeffs): {noise_plateau:.2e}")

    # Check: noise estimates should be above machine epsilon
    assert noise1 > 1e-16, "extend1 estimate too small"
    assert noise2 > 1e-16, "extend2 estimate too small"
    print("   PASS: noise estimates above machine epsilon")

    # Additional example: Runge function
    print("\n2. f(x) = 1/(1+25*x^2)")
    f2 = lambda x: 1.0 / (1.0 + 25 * x**2)
    n2 = 60
    c2 = compute_cheb_coeffs_fft(f2, n2)
    n2_trunc = int(0.65 * n2)
    c2_trunc = c2[:n2_trunc + 1].copy()

    c2_ext1 = extend1(c2_trunc)
    c2_ext2 = extend2(c2_trunc)
    print(f"   Length: {n2_trunc}, extend1: {np.max(np.abs(c2_ext1[n2_trunc+1:])):.2e}, "
          f"extend2: {np.max(np.abs(c2_ext2[n2_trunc+1:])):.2e}")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(2, 2)

    n_plot = np.arange(len(c_trunc))
    n_ext = np.arange(len(c_ext1))

    # Top left: original truncated series
    axes[0, 0].semilogy(n_plot, np.abs(c_trunc), '.b', markersize=4)
    axes[0, 0].set_title("cos(100exp(x)): truncated Chebyshev series", fontsize=10)
    axes[0, 0].set_xlabel("n"); axes[0, 0].set_ylabel("|a_n|")
    axes[0, 0].set_ylim([1e-18, 10]); axes[0, 0].grid(True, alpha=0.3)

    # Top right: extend1 vs original
    axes[0, 1].semilogy(n_ext, np.abs(c_ext1), '.r', markersize=2, label='extend1')
    axes[0, 1].semilogy(n_plot, np.abs(c_trunc), '.b', markersize=4, label='original')
    axes[0, 1].set_title("extend1: noise estimate (red)", fontsize=10)
    axes[0, 1].set_xlabel("n"); axes[0, 1].set_ylabel("|a_n|")
    axes[0, 1].set_ylim([1e-18, 10])
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

    # Bottom left: extend2 vs original
    axes[1, 0].semilogy(n_ext, np.abs(c_ext2), '.g', markersize=2, label='extend2')
    axes[1, 0].semilogy(n_plot, np.abs(c_trunc), '.b', markersize=4, label='original')
    axes[1, 0].set_title("extend2: improved noise estimate (green)", fontsize=10)
    axes[1, 0].set_xlabel("n"); axes[1, 0].set_ylabel("|a_n|")
    axes[1, 0].set_ylim([1e-18, 10])
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3)

    # Bottom right: Runge function
    n2_ext = np.arange(len(c2_ext1))
    n2_trunc_plot = np.arange(len(c2_trunc))
    axes[1, 1].semilogy(n2_trunc_plot, np.abs(c2_trunc), '.b', markersize=6)
    axes[1, 1].semilogy(n2_ext, np.abs(c2_ext1), '.r', markersize=2,
                        label='extend1', alpha=0.7)
    axes[1, 1].semilogy(n2_ext, np.abs(c2_ext2), '.g', markersize=2,
                        label='extend2', alpha=0.7)
    axes[1, 1].set_title("Runge 1/(1+25x²): noise estimates", fontsize=10)
    axes[1, 1].set_xlabel("n"); axes[1, 1].set_ylabel("|a_n|")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Noise level estimation in Chebyshev series", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "noise_level.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
