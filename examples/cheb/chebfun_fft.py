"""The FFT in Chebfun.

Explains how function values at Chebyshev points are converted to
Chebyshev coefficients via the FFT. Based on Chebfun example cheb/ChebfunFFT.m
by Mark Richardson (May 2011).

Original: https://www.chebfun.org/examples/cheb/ChebfunFFT.html
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
from chebfunjax.plotting import chebfun_style
chebfun_style()



def cheb2coeffs_via_fft(fvals):
    """Convert Chebyshev point values to coefficients via FFT.
    fvals should be values at Chebyshev-2 points in ascending order.
    """
    n = len(fvals)
    # Mirror to get values on full circle
    vals_circle = np.concatenate([fvals[::-1], fvals[1:-1]])
    # FFT
    c_full = np.real(np.fft.fft(vals_circle)) / (n - 1)
    # Extract coefficients (first half, scaling endpoints)
    c = c_full[:n].copy()
    c[0] /= 2
    c[-1] /= 2
    return c


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/cheb')
    os.makedirs(outdir, exist_ok=True)

    # --- Build a Chebfun and extract its coefficients --------------------
    fc = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(jnp.pi * x) + x)
    n = len(fc)
    print(f"Chebfun length: {n}")

    # Get the Chebyshev coefficients directly from chebfunjax
    coeffs_cj = np.array(fc.coeffs)
    print(f"Chebyshev coefficients (first 5): {coeffs_cj[:5]}")

    # Get values at Chebyshev points and convert via FFT
    # Chebyshev-2 points: x_k = cos(pi*k/(n-1)) for k=0..n-1 (descending)
    k = np.arange(n)
    cheb_pts = np.cos(np.pi * k / (n - 1))  # descending: [-1..1] reversed
    f_fn = lambda x: np.exp(x) * np.sin(np.pi * x) + x
    fvals_desc = f_fn(cheb_pts)
    # Ascending order (as chebfunjax stores them)
    fvals_asc = fvals_desc[::-1]

    coeffs_fft = cheb2coeffs_via_fft(fvals_asc)
    print(f"\nCoefficients via FFT (first 5): {coeffs_fft[:5]}")
    print(f"Max difference: {np.max(np.abs(coeffs_fft[:n] - coeffs_cj[:n])):.2e}")

    # --- Plot: values at Chebyshev points and the reconstructed function -
    fig, axes = plt.subplots(1, 2)

    xx = np.linspace(-1, 1, 600)
    fv = np.array(fc(jnp.array(xx)))
    axes[0].plot(xx, fv, 'b-', linewidth=1.5, label='$f$')
    axes[0].plot(cheb_pts, fvals_desc, '.r', markersize=7, label='Cheb-2 nodes')
    axes[0].set_title(r'$f(x) = e^x \sin(\pi x) + x$', fontsize=11)
    axes[0].set_xlabel('$x$')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Coefficient decay
    axes[1].semilogy(np.abs(coeffs_cj), 'b.-', markersize=6, linewidth=1.2,
                     label='Chebyshev coefficients')
    axes[1].semilogy(np.abs(coeffs_fft), 'r--', linewidth=1.0,
                     label='Via FFT', alpha=0.7)
    axes[1].set_title('Chebyshev coefficient decay', fontsize=11)
    axes[1].set_xlabel('Index $k$')
    axes[1].set_ylabel('$|a_k|$')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, which='both', alpha=0.3)
    axes[1].set_ylim(bottom=1e-18)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'chebfun_fft.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Verify FFT-based coefficients match chebfunjax
    assert np.max(np.abs(coeffs_fft - coeffs_cj)) < 1e-12, "FFT coefficients mismatch"
    print("\nFFT coefficients match chebfunjax coefficients to 1e-12.")

    print("chebfun_fft: done")
    return True


if __name__ == "__main__":
    run()
