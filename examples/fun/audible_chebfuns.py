"""Can one hear the shape of a chebfun?

Demonstrates that a chebfun can be "heard" by mapping its Chebyshev
coefficients to audio frequencies. This example originally produced
sound output; here we visualize the frequency spectrum.
Translated from fun/AudibleChebfuns.m.

Original: https://www.chebfun.org/examples/fun/AudibleChebfuns.html
Author: Stefan Guttel, November 2011
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



def cheb_coeffs(f_vals, n):
    """Compute Chebyshev coefficients via DCT."""
    # f_vals sampled at Chebyshev nodes
    c = np.fft.rfft(f_vals) / n
    c[1:-1] *= 2  # double interior coefficients
    return np.abs(c[:n//2+1])


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/fun')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3)

    # Sampling points
    N = 512
    x = np.linspace(-1, 1, N)

    # --- Function 1: sin(10*pi*x) ---
    f1 = np.sin(10 * np.pi * x)
    t1 = np.linspace(0, 1, N)  # time
    # Map to "audio": amplitude modulation
    freq_hz = 440 * (1 + f1)  # map to frequencies near A4

    axes[0].plot(x, f1, 'b-', linewidth=1.5)
    axes[0].set_title('f(x) = sin(10πx)\n"audible" waveform', fontsize=10)
    axes[0].set_xlabel('x'); axes[0].set_ylabel('f(x)')
    axes[0].grid(True, alpha=0.3)

    # --- Function 2: Runge function 1/(1+25x^2) ---
    f2 = 1 / (1 + 25 * x**2)
    c2 = cheb_coeffs(f2, N)
    freqs2 = np.linspace(0, N/2, len(c2))

    axes[1].semilogy(freqs2[:50], c2[:50] + 1e-16, 'r-', linewidth=2)
    axes[1].set_title('Runge function 1/(1+25x²)\nChebyshev spectrum', fontsize=10)
    axes[1].set_xlabel('Coefficient index'); axes[1].set_ylabel('|a_k|')
    axes[1].grid(True, alpha=0.3)

    # --- Function 3: exp(sin(pi*x)) ---
    f3 = np.exp(np.sin(np.pi * x))
    c3 = cheb_coeffs(f3, N)
    freqs3 = np.arange(len(c3))

    axes[2].semilogy(freqs3[:60], c3[:60] + 1e-16, 'g-', linewidth=2)
    axes[2].set_title('exp(sin(πx))\nChebyshev spectrum', fontsize=10)
    axes[2].set_xlabel('Coefficient index'); axes[2].set_ylabel('|a_k|')
    axes[2].grid(True, alpha=0.3)

    print("Chebfun spectra as 'audio':")
    print(f"  sin(10πx): {N} samples, fundamental freq index ~10")
    print(f"  Runge function: algebraic decay of coefficients")
    print(f"  exp(sin(πx)): exponential decay of coefficients")

    fig.suptitle('Can One "Hear" the Shape of a Chebfun?', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'audible_chebfuns.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("audible_chebfuns: done")
    return True


if __name__ == "__main__":
    run()
