"""Chebyshev interpolation of oscillatory entire functions.

Demonstrates geometric (spectral) convergence of Chebyshev interpolation
for entire functions, following the Chebfun example approx/Entire.m by
Mark Richardson (October 2011).

Original: https://www.chebfun.org/examples/approx/Entire.html
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



def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/approx')
    os.makedirs(outdir, exist_ok=True)

    # --- Geometric convergence for sin(N*pi*x) ----------------------------
    # As N grows, more Chebyshev coefficients are needed.
    # The theoretical estimate gives n ~ N*pi/log(r) for the optimal r.
    fig, ax = plt.subplots(figsize=(7, 4))
    NN = [1, 2, 4, 8, 16, 32]
    degrees = []
    for N in NN:
        f = cj.chebfun(lambda x, N=N: jnp.sin(N * jnp.pi * x))
        degrees.append(len(f) - 1)

    ax.semilogy(NN, degrees, 'b.-', markersize=10, linewidth=1.5,
                label='Chebfun degree')
    theory = [int(np.ceil(N * np.pi / np.log(1.1))) for N in NN]
    ax.semilogy(NN, theory, 'r--', linewidth=1.5, label=r'$\approx N\pi/\log r$')
    ax.set_xlabel('Oscillation parameter $N$', fontsize=12)
    ax.set_ylabel('Polynomial degree', fontsize=12)
    ax.set_title(r'Degree of $\sin(N\pi x)$ Chebfun vs. $N$', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'polynomial_convergence.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Coefficient decay: analytic vs non-analytic ----------------------
    fig2, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Analytic: exp(x) — geometric decay
    f_exp = cj.chebfun(lambda x: jnp.exp(x))
    coeffs_exp = np.abs(np.array(f_exp.coeffs))
    axes[0].semilogy(coeffs_exp, 'b.-', markersize=6, linewidth=1.2)
    axes[0].set_title('Chebyshev coefficients of $e^x$ (geometric decay)',
                      fontsize=11)
    axes[0].set_xlabel('Coefficient index $k$')
    axes[0].set_ylabel('$|a_k|$')
    axes[0].grid(True, which='both', alpha=0.4)
    axes[0].set_ylim(bottom=1e-17)

    # Non-analytic: |x| — algebraic decay ~ 1/k^2
    f_abs = cj.chebfun(lambda x: jnp.abs(x), n=256)
    coeffs_abs = np.abs(np.array(f_abs.coeffs))
    kk = np.arange(1, len(coeffs_abs) + 1)
    axes[1].semilogy(coeffs_abs, 'b.-', markersize=4, linewidth=1.0,
                     label='$|a_k|$')
    axes[1].semilogy(kk[10:], 2.0 / kk[10:]**2, 'r--', linewidth=1.5,
                     label=r'$2/k^2$ envelope')
    axes[1].set_title(r'Chebyshev coefficients of $|x|$ (algebraic decay)',
                      fontsize=11)
    axes[1].set_xlabel('Coefficient index $k$')
    axes[1].set_ylabel('$|a_k|$')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, which='both', alpha=0.4)
    axes[1].set_ylim(bottom=1e-6)

    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, 'polynomial_convergence_coeffs.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)

    print("polynomial_convergence: done")
    return True


if __name__ == "__main__":
    run()
