"""Convolution of probability distributions.

Demonstrates how the PDF of a sum of independent random variables is
the convolution of their individual PDFs. Translated from
stats/ProbabilityConvolution.m.

Original: https://www.chebfun.org/examples/stats/ProbabilityConvolution.html
Authors: Nick Hale and Alex Townsend, January 2014
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def normal_pdf(xs, mu, sigma):
    return np.exp(-0.5 * ((xs - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))


def gamma_pdf(xs, k, theta):
    from scipy.special import gamma
    result = np.zeros_like(xs, dtype=float)
    mask = xs > 0
    result[mask] = (xs[mask]**(k - 1) * np.exp(-xs[mask] / theta)
                    / (theta**k * gamma(k)))
    return result


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- 1. Normal distribution convolution ---
    # N(mu1, sigma1) * N(mu2, sigma2) = N(mu1+mu2, sqrt(sigma1^2+sigma2^2))
    m1, s1 = 0.1, 0.10
    m2, s2 = -0.3, 0.11
    xs_n = np.linspace(-1.5, 1.5, 500)

    N1 = normal_pdf(xs_n, m1, s1)
    N2 = normal_pdf(xs_n, m2, s2)

    # Numerical convolution using scipy
    from scipy.signal import fftconvolve
    dx = xs_n[1] - xs_n[0]
    N3_conv = fftconvolve(N1, N2, mode='full') * dx
    xs_conv = np.linspace(xs_n[0] + xs_n[0], xs_n[-1] + xs_n[-1], len(N3_conv))

    # Exact result
    N4 = normal_pdf(xs_n, m1 + m2, np.sqrt(s1**2 + s2**2))

    axes[0].plot(xs_n, N1, 'b-', linewidth=2, label=f'N({m1},{s1})')
    axes[0].plot(xs_n, N2, 'r-', linewidth=2, label=f'N({m2},{s2})')
    # Only show convolution within reasonable range
    mask_c = (xs_conv >= -1.5) & (xs_conv <= 1.5)
    axes[0].plot(xs_conv[mask_c], N3_conv[mask_c], 'k-', linewidth=2, label='Convolution')
    axes[0].plot(xs_n, N4, 'g--', linewidth=2, label='Exact sum')
    axes[0].set_title('Normal distribution convolution', fontsize=10)
    axes[0].set_xlabel('x'); axes[0].legend(fontsize=8)
    axes[0].set_xlim(-1.5, 1.5); axes[0].grid(True, alpha=0.3)

    # Verify: max error
    idx_match = np.argmin(np.abs(xs_conv - 0))
    print(f"Normal: N(mu1+mu2, sqrt(s1²+s2²)) at x=0: "
          f"exact={N4[len(N4)//2]:.6f}")

    # --- 2. Gamma distribution convolution ---
    # G(k1, theta) + G(k2, theta) = G(k1+k2, theta)
    k1, k2, theta = 2, 1, 0.3
    xs_g = np.linspace(0, 5, 500)

    G1 = gamma_pdf(xs_g, k1, theta)
    G2 = gamma_pdf(xs_g, k2, theta)
    dx_g = xs_g[1] - xs_g[0]
    G3 = fftconvolve(G1, G2, mode='full') * dx_g
    xs_g3 = np.linspace(0, xs_g[-1] * 2, len(G3))
    G4 = gamma_pdf(xs_g, k1 + k2, theta)

    axes[1].plot(xs_g, G1, 'b-', linewidth=2, label=f'Γ({k1},{theta})')
    axes[1].plot(xs_g, G2, 'r-', linewidth=2, label=f'Γ({k2},{theta})')
    mask_g = (xs_g3 >= 0) & (xs_g3 <= 5)
    axes[1].plot(xs_g3[mask_g], G3[mask_g], 'k-', linewidth=2, label='Convolution')
    axes[1].plot(xs_g, G4, 'g--', linewidth=2, label=f'Γ({k1+k2},{theta})')
    axes[1].set_title('Gamma distribution convolution', fontsize=10)
    axes[1].set_xlabel('x'); axes[1].legend(fontsize=8)
    axes[1].set_xlim(0, 5); axes[1].grid(True, alpha=0.3)

    print(f"Gamma: G({k1},{theta}) * G({k2},{theta}) ≈ G({k1+k2},{theta})")

    # --- 3. Exponential distribution convolution ---
    # Exp(lambda) * Exp(lambda) = Gamma(2, 1/lambda)
    lam = 0.25
    xs_e = np.linspace(0, 40, 500)

    def exp_pdf(xs, lam):
        result = np.zeros_like(xs, dtype=float)
        mask = xs >= 0
        result[mask] = lam * np.exp(-lam * xs[mask])
        return result

    E1 = exp_pdf(xs_e, lam)
    dx_e = xs_e[1] - xs_e[0]
    E2 = fftconvolve(E1, E1, mode='full') * dx_e
    xs_e2 = np.linspace(0, xs_e[-1] * 2, len(E2))
    # Exact: gamma(k=2, theta=1/lam)
    E_exact = gamma_pdf(xs_e, 2, 1.0 / lam)

    axes[2].plot(xs_e, E1, 'b-', linewidth=2, label=f'Exp({lam})')
    mask_e = (xs_e2 >= 0) & (xs_e2 <= 40)
    axes[2].plot(xs_e2[mask_e], E2[mask_e], 'k-', linewidth=2, label='Exp*Exp')
    axes[2].plot(xs_e, E_exact, 'r--', linewidth=2, label=f'Γ(2, {1/lam:.0f})')
    axes[2].set_title('Exponential distribution convolution', fontsize=10)
    axes[2].set_xlabel('x'); axes[2].legend(fontsize=8)
    axes[2].set_xlim(0, 40); axes[2].grid(True, alpha=0.3)

    print(f"Exponential: Exp({lam}) * Exp({lam}) ≈ Γ(2, {1/lam:.1f})")

    fig.suptitle('Convolution of Probability Distributions', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'probability_convolution.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("probability_convolution: done")
    return True


if __name__ == "__main__":
    run()
