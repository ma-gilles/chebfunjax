"""Lebesgue functions and Lebesgue constants.

Compares Lebesgue constants for Chebyshev, equispaced, and random
interpolation points, showing the Runge phenomenon.

Credit: Nick Trefethen, November 2010.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/LebesgueConst.html
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


def lebesgue_function(nodes, x_eval):
    """Compute Lebesgue function L(x) = sum |l_k(x)| for interpolation nodes."""
    n = len(nodes)
    L = np.zeros(len(x_eval))
    for k in range(n):
        # Compute k-th Lagrange basis polynomial at x_eval
        lk = np.ones(len(x_eval))
        for j in range(n):
            if j != k:
                lk *= (x_eval - nodes[j]) / (nodes[k] - nodes[j] + 1e-300)
        L += np.abs(lk)
    return L


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    xx = np.linspace(-1.0, 1.0, 400)

    try:
        from chebfunjax.utils.lebesgue import lebesgue
        USE_CJ = True
    except ImportError:
        USE_CJ = False

    def compute_lebesgue(pts):
        if USE_CJ:
            try:
                L_fun, Lambda = lebesgue(pts)
                L_vals = np.array([float(L_fun(jnp.array(x))) for x in xx])
                return L_vals, Lambda
            except Exception:
                pass
        L_vals = lebesgue_function(pts, xx)
        return L_vals, np.max(L_vals)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # 10 Chebyshev points
    pts_cheb_10 = np.cos(np.pi * np.arange(10) / 9)
    L10c, lam10c = compute_lebesgue(pts_cheb_10)
    axes[0, 0].plot(xx, L10c, 'b', lw=1.5)
    axes[0, 0].set_title(f'10 Chebyshev pts  Λ={lam10c:.2f}', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # 10 equispaced points
    pts_eq_10 = np.linspace(-1, 1, 10)
    L10e, lam10e = compute_lebesgue(pts_eq_10)
    axes[0, 1].semilogy(xx, L10e + 1e-1, 'r', lw=1.5)
    axes[0, 1].set_title(f'10 equispaced pts  Λ={lam10e:.2f}', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # 40 Chebyshev points
    pts_cheb_40 = np.cos(np.pi * np.arange(40) / 39)
    L40c, lam40c = compute_lebesgue(pts_cheb_40)
    axes[1, 0].semilogy(xx, L40c + 1e-1, 'b', lw=1.5)
    axes[1, 0].set_title(f'40 Chebyshev pts  Λ={lam40c:.2f}', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # 40 equispaced points (large Lebesgue constant!)
    pts_eq_40 = np.linspace(-1, 1, 40)
    # Only compute on interior to avoid overflow
    xx_safe = np.linspace(-0.99, 0.99, 200)
    L40e = lebesgue_function(pts_eq_40, xx_safe)
    lam40e = np.max(L40e)
    axes[1, 1].semilogy(xx_safe, L40e + 1, 'r', lw=1.5)
    axes[1, 1].set_title(f'40 equispaced pts  Λ≈{lam40e:.2e}', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('Lebesgue functions: Chebyshev vs. equispaced', fontsize=12)
    for ax in axes.flat:
        ax.set_xlabel('x')

    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'LebesgueConst.png'), dpi=150)
    plt.close(fig)

    print(f"LebesgueConst: n=10 cheb Λ={lam10c:.2f}, eq Λ={lam10e:.2f}")
    return True


if __name__ == '__main__':
    run()
