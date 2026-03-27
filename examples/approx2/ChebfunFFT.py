"""The FFT in Chebfun.

Explains how the FFT (and IFFT) underlies the conversion between Chebyshev
point values and Chebyshev expansion coefficients.

Credit: Mark Richardson, May 2011.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/ChebfunFFT.html
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


_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def chebpts(n):
    """n Chebyshev points of second kind on [-1,1], ordered from x=1 down to x=-1."""
    k = np.arange(n)
    return np.cos(np.pi * k / (n - 1))


def vals_to_coeffs(fvals):
    """Convert function values at Chebyshev points (from x=1 to x=-1) to coefficients.

    Uses the DCT-I via FFT: extend values symmetrically, then take FFT.
    """
    n = len(fvals)
    # Extend: [f(1), f(cos(pi/n-1)), ..., f(-1), f(cos(pi(n-2)/n-1)), ..., f(cos(pi/n-1))]
    extended = np.concatenate([fvals, fvals[-2:0:-1]])
    F = np.real(np.fft.fft(extended)) / (n - 1)
    coeffs = F[:n].copy()
    coeffs[0] /= 2
    coeffs[-1] /= 2
    return coeffs


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Build a test chebfun
    def test_func(x): return jnp.exp(x) * jnp.sin(jnp.pi * x) + x
    fc = cj.chebfun(test_func)
    n = len(fc)

    # Get Chebyshev points and function values
    pts = chebpts(n)
    fvals = np.array([float(test_func(jnp.array(x))) for x in pts])

    # Compute coefficients manually via FFT
    manual_coeffs = vals_to_coeffs(fvals)
    chebfun_coeffs = np.array(fc.coeffs)

    # Compare
    err = np.max(np.abs(manual_coeffs - chebfun_coeffs[:len(manual_coeffs)]))
    print(f"ChebfunFFT: max difference in coefficients = {err:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    xx_plot = np.linspace(-1.0, 1.0, 400)
    f_vals_plot = np.array([float(fc(jnp.array(x))) for x in xx_plot])
    ax.plot(xx_plot, f_vals_plot, 'b', lw=1.8, label='f(x)')
    ax.plot(pts, fvals, '.r', ms=8, label='Chebyshev nodes')
    ax.set_title('f(x) with Chebyshev nodes', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    k_cheb = np.arange(len(chebfun_coeffs))
    ax2.semilogy(k_cheb, np.abs(chebfun_coeffs) + 1e-18, 'b.', ms=5,
                 label='Chebfun coeffs')
    ax2.semilogy(np.arange(len(manual_coeffs)),
                 np.abs(manual_coeffs) + 1e-18, 'r+', ms=8,
                 label='FFT coeffs')
    ax2.set_title('Chebyshev coefficients (FFT method)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_xlabel('degree k')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('The FFT in Chebfun: values ↔ coefficients', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'ChebfunFFT.png'), dpi=150)
    plt.close(fig)

    assert err < 1e-12, f"FFT coefficient mismatch: {err}"
    print("ChebfunFFT: all assertions passed.")
    return True


if __name__ == '__main__':
    run()
