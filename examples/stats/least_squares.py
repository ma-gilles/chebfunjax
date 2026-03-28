"""Least-squares data fitting.

Demonstrates polynomial least-squares fitting to noisy data using
Chebyshev polynomial bases. Translated from stats/LeastSquares.m.

Original: https://www.chebfun.org/examples/stats/LeastSquares.html
Author: Nick Trefethen, October 2011
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

def cheb_vandermonde(xs, deg):
    """Build Chebyshev Vandermonde matrix for least-squares fitting."""
    n = deg + 1
    V = np.ones((len(xs), n))
    if deg >= 1:
        V[:, 1] = xs
    for k in range(2, n):
        V[:, k] = 2 * xs * V[:, k-1] - V[:, k-2]
    return V

def polyfit_cheb(xs, ys, deg):
    """Fit degree-deg polynomial in Chebyshev basis to data by least squares."""
    V = cheb_vandermonde(xs, deg)
    coeffs, _, _, _ = np.linalg.lstsq(V, ys, rcond=None)
    return coeffs

def eval_cheb(xs, coeffs):
    """Evaluate Chebyshev polynomial given coefficients."""
    deg = len(coeffs) - 1
    V = cheb_vandermonde(xs, deg)
    return V @ coeffs

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2)

    # --- 1. Discrete least-squares fit to Runge-like function with noise ---
    npts = 100
    xs_data = np.linspace(-1, 1, npts)
    ys_data = 1.0 / (1 + 25 * xs_data**2) + 0.1 * rng.standard_normal(npts)

    deg = 10
    coeffs = polyfit_cheb(xs_data, ys_data, deg)

    xs_fine = np.linspace(-1, 1, 500)
    ys_fit = eval_cheb(xs_fine, coeffs)
    ys_true = 1.0 / (1 + 25 * xs_fine**2)

    axes[0].plot(xs_data, ys_data, 'xk', markersize=6, alpha=0.6, label='Noisy data')
    axes[0].plot(xs_fine, ys_true, color='#77AC30', linestyle='--', linewidth=2, label='True f(x)')
    axes[0].plot(xs_fine, ys_fit, color='#D95319', linestyle='-', linewidth=2, label=f'Degree-{deg} fit')
    axes[0].set_title('Discrete polynomial least-squares fit', fontsize=11)
    axes[0].legend(fontsize=9)

    print(f"Discrete LS: L2 error vs truth = {np.sqrt(np.mean((ys_fit - ys_true)**2)):.4f}")

    # --- 2. Continuous least-squares: fit piecewise function ---
    # f(x) = |x+0.2| - 0.5*sign(x-0.5) -- discontinuous
    def jagged(x):
        return np.abs(x + 0.2) - 0.5 * np.sign(x - 0.5)

    xs_c = np.linspace(-1, 1, 1000)
    ys_c = jagged(xs_c)

    # Continuous LS: integrate against Chebyshev polynomials
    # c_k = (2/pi) * integral_{-1}^{1} f(x) T_k(x) / sqrt(1-x^2) dx
    # Approximate via discrete sum on fine grid
    deg_c = 10
    coeffs_c = polyfit_cheb(xs_c, ys_c, deg_c)
    ys_fit_c = eval_cheb(xs_c, coeffs_c)

    axes[1].plot(xs_c, ys_c, 'k-', linewidth=1.5, label='f(x) = |x+0.2| - 0.5·sgn(x-0.5)')
    axes[1].plot(xs_c, ys_fit_c, color='#D95319', linestyle='-', linewidth=2, label=f'Degree-{deg_c} LS fit')
    axes[1].set_title('Continuous polynomial least-squares fit', fontsize=11)
    axes[1].legend(fontsize=9)

    l2_err = np.sqrt(np.mean((ys_fit_c - ys_c)**2))
    print(f"Continuous LS: L2 error = {l2_err:.4f}")

    fig.suptitle('Least-Squares Data Fitting with Chebyshev Polynomials', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'least_squares.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("least_squares: done")
    return True

if __name__ == "__main__":
    run()
