"""Convergence bounds for entire functions.

For entire functions, Chebyshev coefficients decay faster than any geometric
rate. This example verifies the Bernstein ellipse bounds for exp(x).

Credit: Nick Trefethen, April 2016.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/EntireBound.html
"""

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

def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Exp(x): Chebyshev coefficients and Bernstein bounds
    f = cj.chebfun(jnp.exp)
    coeffs = np.abs(np.array(f.coeffs)) + 1e-18
    n_max = len(coeffs) - 1
    nvec = np.arange(len(coeffs))

    fig, ax = plt.subplots()
    ax.semilogy(nvec, coeffs, 'b.', ms=7, label='Chebfun coefficients')

    # Bernstein bounds for rho = 2, 4, 8, 16, 32
    colors = ['r', 'g', 'm', 'c', 'orange']
    for rho, col in zip([2.0, 4.0, 8.0, 16.0, 32.0], colors):
        M = np.exp((rho + 1.0 / rho) / 2)
        bound = 2 * M * rho**(-nvec)
        ax.semilogy(nvec, bound, '--', color=col, lw=1.2,
                    label=f'ρ={rho}')

    ax.set_title('Chebyshev coefficients of exp(x) and Bernstein bounds', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(1e-18, 10)

    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'EntireBound.png'), dpi=150)
    plt.close(fig)

    # Also show for 1/(1+x^2): poles at ±i, rho = 1+sqrt(2) ≈ 2.414
    f2 = cj.chebfun(lambda x: 1.0 / (1.0 + x**2))
    coeffs2 = np.abs(np.array(f2.coeffs)) + 1e-18
    rho_exact = 1.0 + np.sqrt(2)  # geometric rate for 1/(1+x^2)

    fig2, ax2 = plt.subplots()
    nvec2 = np.arange(len(coeffs2))
    ax2.semilogy(nvec2, coeffs2, 'b.', ms=7, label='Chebfun coefficients')
    M2 = 1.0 / abs(1 + (rho_exact**2 + rho_exact**(-2)) / 4 - 1.0)
    ax2.semilogy(nvec2, 2 * M2 * rho_exact**(-nvec2), 'r--', lw=1.5,
                 label=f'ρ={rho_exact:.3f} bound')
    ax2.set_title('Chebyshev coefficients of 1/(1+x²)', fontsize=11)
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(_OUTDIR, 'EntireBound_runge.png'), dpi=150)
    plt.close(fig2)

    print(f"EntireBound: len(exp chebfun) = {len(f)}")
    return True

if __name__ == '__main__':
    run()
