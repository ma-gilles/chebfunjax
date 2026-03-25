"""Central Limit Theorem via convolution.

Demonstrates the Central Limit Theorem by convolving a triangular
distribution and showing convergence to the normal distribution.
Translated from stats/CentralLimitTheorem.m.

Original: https://www.chebfun.org/examples/stats/CentralLimitTheorem.html
Authors: Nick Trefethen and Mohsin Javed, July 2012
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.signal import fftconvolve
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Triangular distribution: support [-4/3, 2/3], zero outside [-3,3]
    # PDF: (4/3 + x)/2 on [-4/3, 2/3]
    n_pts = 2000
    xs = np.linspace(-3, 3, n_pts)
    dx = xs[1] - xs[0]

    def tri_pdf(x):
        result = np.zeros_like(x, dtype=float)
        mask = (x >= -4/3) & (x <= 2/3)
        result[mask] = (4/3 + x[mask]) / 2
        return result

    X_pdf = tri_pdf(xs)

    # Mean and variance
    mean_X = np.trapezoid(xs * X_pdf, xs)
    var_X = np.trapezoid(xs**2 * X_pdf, xs) - mean_X**2
    sigma_X = np.sqrt(var_X)
    print(f"Mean of X: {mean_X:.6f}  (exact: 0)")
    print(f"Variance of X: {var_X:.6f}  (exact: 2/9 = {2/9:.6f})")

    # Normal with same mean and variance
    gauss = np.exp(-0.5 * ((xs - mean_X) / sigma_X)**2) / (sigma_X * np.sqrt(2 * np.pi))

    axes[0].plot(xs, X_pdf, 'b-', linewidth=2, label='Triangular X')
    axes[0].plot(xs, gauss, 'r--', linewidth=2, label='Normal fit')
    axes[0].set_title('Distribution of X', fontsize=11)
    axes[0].set_xlabel('x'); axes[0].legend(fontsize=9)
    axes[0].set_xlim(-3, 3); axes[0].set_ylim(-0.2, 1.2)
    axes[0].grid(True, alpha=0.3)

    # Convolve X with itself and renormalize
    X2 = fftconvolve(X_pdf, X_pdf, mode='full') * dx
    # The support doubles, need to rescale to compare
    n2 = len(X2)
    xs2 = np.linspace(-6, 6, n2)
    # Renormalize: scale by sqrt(2) so variance is again var_X
    # S2(x) = sqrt(2) * X2(sqrt(2)*x)  -- distribution of (X1+X2)/sqrt(2)
    xs2_rescaled = np.linspace(-3, 3, n_pts)
    # Interpolate X2 at sqrt(2)*x
    from scipy.interpolate import interp1d
    X2_interp = interp1d(xs2, X2, bounds_error=False, fill_value=0)
    S2 = np.sqrt(2) * X2_interp(np.sqrt(2) * xs2_rescaled)

    axes[1].plot(xs2_rescaled, S2, 'b-', linewidth=2, label='(X+X)/√2')
    axes[1].plot(xs, gauss, 'r--', linewidth=2, label='Normal fit')
    axes[1].set_title('Renorm. sum of 2', fontsize=11)
    axes[1].set_xlabel('x'); axes[1].legend(fontsize=9)
    axes[1].set_xlim(-3, 3); axes[1].set_ylim(-0.2, 1.2)
    axes[1].grid(True, alpha=0.3)

    # Convolve again for sum of 3
    X3 = fftconvolve(X2, X_pdf, mode='full') * dx
    n3 = len(X3)
    xs3 = np.linspace(-9, 9, n3)
    X3_interp = interp1d(xs3, X3, bounds_error=False, fill_value=0)
    S3 = np.sqrt(3) * X3_interp(np.sqrt(3) * xs2_rescaled)

    axes[2].plot(xs2_rescaled, S3, 'b-', linewidth=2, label='(X+X+X)/√3')
    axes[2].plot(xs, gauss, 'r--', linewidth=2, label='Normal fit')
    axes[2].set_title('Renorm. sum of 3', fontsize=11)
    axes[2].set_xlabel('x'); axes[2].legend(fontsize=9)
    axes[2].set_xlim(-3, 3); axes[2].set_ylim(-0.2, 1.2)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Central Limit Theorem', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'central_limit_theorem.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("central_limit_theorem: done")
    return True


if __name__ == "__main__":
    run()
