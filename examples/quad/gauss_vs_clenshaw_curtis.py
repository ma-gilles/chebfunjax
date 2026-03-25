"""Gauss and Clenshaw-Curtis quadrature.

Compares Gauss-Legendre and Clenshaw-Curtis quadrature convergence.
Based on Chebfun example quad/GaussClenCurt.m by Nick Trefethen (September 2010).

Original: https://www.chebfun.org/examples/quad/GaussClenCurt.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def clenshaw_curtis_nodes_weights(n):
    """Clenshaw-Curtis nodes (Chebyshev-2 points) and weights on [-1,1]."""
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    k = np.arange(n)
    x = -np.cos(np.pi * k / (n - 1))  # Chebyshev points of 2nd kind
    # Weights via FFT-based Clenshaw-Curtis formula
    c = np.zeros(n)
    c[0::2] = 2.0 / (1 - np.arange(0, n, 2)**2)
    # DCT
    w = np.real(np.fft.ifft(np.concatenate([c, c[-2:0:-1]])))[:n]
    w[0] /= 2
    w[-1] /= 2
    return x, w


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/quad')
    os.makedirs(outdir, exist_ok=True)

    # Wiggly analytic function
    f_fn = lambda x: x * np.sin(2 * np.exp(2 * np.sin(2 * np.exp(2 * x))))
    f_jax = lambda x: x * jnp.sin(2 * jnp.exp(2 * jnp.sin(2 * jnp.exp(2 * x))))

    # Chebfun "exact" integral
    fc = cj.chebfun(f_jax)
    I_exact = float(fc.sum())
    print(f"Reference integral (Chebfun) = {I_exact:.15f}")
    print(f"Chebfun degree = {len(fc)}")

    # Convergence comparison
    NN = list(range(10, 220, 10))
    err_gauss = []
    err_cc = []

    for n in NN:
        # Gauss-Legendre
        x_g, w_g = roots_legendre(n)
        I_g = np.dot(w_g, f_fn(x_g))
        err_gauss.append(abs(I_g - I_exact))

        # Clenshaw-Curtis
        try:
            x_c, w_c = clenshaw_curtis_nodes_weights(n)
            I_c = np.dot(w_c, f_fn(x_c))
            err_cc.append(abs(I_c - I_exact))
        except Exception:
            err_cc.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: function plot
    xx = np.linspace(-1, 1, 600)
    axes[0].plot(xx, f_fn(xx), 'b-', linewidth=1.5)
    axes[0].set_title(r'$f(x) = x\sin(2e^{2\sin(2e^{2x})})$', fontsize=11)
    axes[0].set_xlabel('$x$')
    axes[0].grid(True, alpha=0.3)

    # Right: convergence
    axes[1].semilogy(NN, err_gauss, 'b.-', markersize=8, linewidth=1.4,
                     label='Gauss-Legendre')
    axes[1].semilogy(NN, err_cc, 'r.-', markersize=8, linewidth=1.4,
                     label='Clenshaw-Curtis')
    axes[1].set_xlabel('Number of quadrature points')
    axes[1].set_ylabel('Error')
    axes[1].set_title('Convergence of quadrature rules', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, which='both', alpha=0.3)
    axes[1].set_ylim(bottom=1e-17)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'gauss_vs_clenshaw_curtis.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Gauss 100-pt error: {err_gauss[9]:.2e}")
    print(f"CC    100-pt error: {err_cc[9]:.2e}")
    print("gauss_vs_clenshaw_curtis: done")
    return True


if __name__ == "__main__":
    run()
