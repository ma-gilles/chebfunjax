"""Polynomial basis for Hermite interpolation.

Computes a polynomial basis for Hermite interpolation (matching function
values and derivatives at given nodes) using Chebfun arithmetic.

Credit: Pedro Gonnet, September 2010.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/HermiteBasis.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def hermite_basis(nodes):
    """Compute Hermite basis polynomials at given nodes."""
    n = len(nodes)
    # Build Lagrange basis first
    def lagrange_k(k, x):
        result = np.ones_like(np.asarray(x, dtype=float))
        for j in range(n):
            if j != k:
                result *= (x - nodes[j]) / (nodes[k] - nodes[j])
        return result

    def lagrange_k_deriv(k, x):
        result = np.zeros_like(np.asarray(x, dtype=float))
        for i in range(n):
            if i != k:
                term = np.ones_like(result) / (nodes[k] - nodes[i])
                for j in range(n):
                    if j != k and j != i:
                        term *= (x - nodes[j]) / (nodes[k] - nodes[j])
                result += term
        return result

    # Hermite basis: h_k(x) = (1 - 2*(x-x_k)*lk'(x_k)) * lk(x)^2
    def H_k(k, x):
        lk = lagrange_k(k, x)
        lkp = lagrange_k_deriv(k, x)
        return (1.0 - 2.0 * (x - nodes[k]) * lkp[0]) * lk**2

    # Hermite basis for derivative: hk(x) = (x - xk) * lk(x)^2
    def Hd_k(k, x):
        lk = lagrange_k(k, x)
        return (x - nodes[k]) * lk**2

    return H_k, Hd_k


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    nodes = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    n = len(nodes)
    H_k, Hd_k = hermite_basis(nodes)

    xx = np.linspace(-1.0, 1.0, 400)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    colors = ['b', 'r', 'g', 'm', 'orange']
    for k in range(n):
        hk_vals = H_k(k, xx)
        ax.plot(xx, hk_vals, color=colors[k], lw=1.5, label=f'H_{k}')
    ax.plot(nodes, np.ones_like(nodes), '.k', ms=10)
    ax.set_title('Hermite value basis polynomials H_k(x)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for k in range(n):
        hdk_vals = Hd_k(k, xx)
        ax2.plot(xx, hdk_vals, color=colors[k], lw=1.5, label=f'Hd_{k}')
    ax2.plot(nodes, np.zeros_like(nodes), '.k', ms=10)
    ax2.set_title('Hermite derivative basis polynomials Hd_k(x)', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Verify: interpolate sin(x) with Hermite interpolation
    f_vals = np.sin(nodes)
    fp_vals = np.cos(nodes)
    hermite_approx = sum(f_vals[k] * H_k(k, xx) + fp_vals[k] * Hd_k(k, xx)
                         for k in range(n))
    exact = np.sin(xx)

    fig2, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(xx, exact, 'k--', lw=1.5, label='sin(x)')
    ax3.plot(xx, hermite_approx, 'r', lw=1.5, label='Hermite interp')
    ax3.plot(nodes, f_vals, '.b', ms=10)
    err = np.max(np.abs(hermite_approx - exact))
    ax3.set_title(f'Hermite interpolation of sin(x), max err={err:.2e}', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(_OUTDIR, 'HermiteBasis_interp.png'), dpi=150)
    plt.close(fig2)

    fig.suptitle('Hermite interpolation basis polynomials', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'HermiteBasis.png'), dpi=150)
    plt.close(fig)

    print(f"HermiteBasis: Hermite interpolation error = {err:.2e}")
    return True


if __name__ == '__main__':
    run()
