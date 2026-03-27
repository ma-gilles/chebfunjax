"""Chebyshev coefficients and convergence.

Demonstrates Chebyshev coefficient decay, exact coefficients, and
convergence rates, following cheb/ChebyshevCoeffs.m by Nick Trefethen
(September 2010) and cheb/Convergence.m by Alex Townsend (October 2010).

Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
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

from chebfunjax.utils.transforms import vals2coeffs
from chebfunjax.utils.quadrature import chebpts


def run():
    print("=" * 60)
    print("Chebyshev coefficients and convergence")
    print("=" * 60)

    # --- Coefficient decay for analytic function ---
    f = cj.chebfun(jnp.sin)
    coeffs_sin = f.funs[0].tech.coeffs
    print(f"\nsin(x): {len(coeffs_sin)} Chebyshev coefficients")
    print("First 8 non-tiny coefficients:")
    for i, c in enumerate(coeffs_sin[:8]):
        print(f"  a_{i} = {float(c):.6e}")

    # sin(x) has only odd-index coefficients
    for i in range(0, min(8, len(coeffs_sin)), 2):
        assert abs(float(coeffs_sin[i])) < 1e-14, f"a_{i} should be zero for sin(x)"

    # --- Exact coefficients of exp(x): I_n(1) (modified Bessel) ---
    f_exp = cj.chebfun(jnp.exp)
    coeffs_exp = f_exp.funs[0].tech.coeffs
    print(f"\nexp(x): {len(coeffs_exp)} Chebyshev coefficients")
    print("First 6 (geometric decay):")
    for i, c in enumerate(coeffs_exp[:6]):
        print(f"  a_{i} = {float(c):.8e}")

    # Geometric decay: ratio ~ 1/e per step (roughly)
    ratios = [abs(float(coeffs_exp[i+1]) / float(coeffs_exp[i]))
              for i in range(1, min(8, len(coeffs_exp)-1))
              if abs(float(coeffs_exp[i])) > 1e-16]
    print(f"  Decay ratios (should be <1): {[f'{r:.3f}' for r in ratios[:5]]}")
    assert all(r < 1.0 for r in ratios), "Coefficients should decay"

    # --- Convergence rates ---
    # C^k function: algebraic rate
    # Analytic function: geometric rate
    print(f"\n1/cosh(x) (entire): convergence")
    f_sech = cj.chebfun(lambda x: 1.0 / jnp.cosh(x))
    print(f"  Length: {len(f_sech.funs[0].tech.coeffs)}")

    # --- Plot: coefficient magnitudes ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Chebyshev coefficients of sin(x), exp(x), abs(x)
    f_abs = cj.chebfun(jnp.abs, n=256)
    c_sin = np.abs(np.array(coeffs_sin))
    c_exp = np.abs(np.array(coeffs_exp))
    c_abs = np.abs(np.array(f_abs.funs[0].tech.coeffs[:256]))

    axes[0].semilogy(c_sin[:20] + 1e-17, 'b.-', label='sin(x)')
    axes[0].semilogy(c_exp[:20] + 1e-17, 'r.-', label='exp(x)')
    axes[0].set_title("Chebyshev coefficient magnitudes", fontsize=12)
    axes[0].set_xlabel("Index n"); axes[0].set_ylabel("|aₙ|")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(c_abs[:256] + 1e-17, 'g-', label='|x|')
    nn = np.arange(1, 257)
    axes[1].loglog(nn, 1.0/nn**2, 'k--', alpha=0.5, label='O(1/n²)')
    axes[1].set_title("Coefficient decay: |x| (algebraic)", fontsize=12)
    axes[1].set_xlabel("Index n"); axes[1].set_ylabel("|aₙ|")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Chebyshev coefficient decay rates", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "chebyshev_coefficients.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
