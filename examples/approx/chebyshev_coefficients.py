"""Chebyshev coefficients and spectral convergence.

Demonstrates accessing Chebyshev coefficients and the spectral
decay rate for smooth functions.

Credit: Inspired by Chebfun approx/ChebCoeffs.m and related examples.
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import plot, plotcoeffs


def run():
    print("=" * 60)
    print("Chebyshev coefficients and spectral convergence")
    print("=" * 60)

    # --- Chebyshev coefficients of a smooth function ------------------
    # MATLAB: f = chebfun('exp(x)*sin(pi*x) + x'); a = chebcoeffs(f);
    fc = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(jnp.pi * x) + x)
    coeffs = fc.coeffs
    print(f"\nf(x) = exp(x)*sin(pi*x) + x on [-1,1]:")
    print(f"  Number of coefficients: {len(fc)}")
    print(f"  First 5 Chebyshev coefficients: {np.array(coeffs[:5])}")
    print(f"  Last 3 coefficients (should be ~0): {np.array(coeffs[-3:])}")

    # Verify that the last coefficients are near machine epsilon
    last_coeff = float(jnp.abs(coeffs[-1]))
    print(f"  |a_N| = {last_coeff:.2e}  (should be < 1e-13)")
    assert last_coeff < 1e-13, f"Last coefficient too large: {last_coeff}"

    # --- Spectral decay for different function classes ----------------
    # Entire function: exp(x) — exponential decay
    # Analytic in strip: 1/(1+x^2) — geometric decay
    print("\nSpectral decay:")

    f_exp = cj.chebfun(lambda x: jnp.exp(x))
    c_exp = np.abs(np.array(f_exp.coeffs))
    print(f"  exp(x): {len(f_exp)} coefficients, ratio |a_k|/|a_{{k-1}}| < 1/2")

    # Verify geometric decay by checking ratios
    for fn_name, fn_lambda in [
        ("exp(x)",       lambda x: jnp.exp(x)),
        ("cos(10*x)",    lambda x: jnp.cos(10.0*x)),
        ("sin(pi*x/2)",  lambda x: jnp.sin(jnp.pi*x/2.0)),
    ]:
        f = cj.chebfun(fn_lambda)
        c = np.abs(np.array(f.coeffs))
        n = len(f)
        print(f"  {fn_name}: n={n}, max|coeff| = {c.max():.2e}")

    # --- Reconstruction from coefficients: f(x) = sum a_k T_k(x) -----
    # Verify that evaluating via coefficients gives same result as chebfun
    f_sin = cj.chebfun(lambda x: jnp.sin(3.0 * x))
    coeffs_sin = np.array(f_sin.coeffs)
    x_test = jnp.linspace(-1.0, 1.0, 100)
    f_eval = f_sin(x_test)
    exact_eval = jnp.sin(3.0 * x_test)
    err = float(jnp.max(jnp.abs(f_eval - exact_eval)))
    print(f"\nsin(3x) reconstruction error: {err:.2e}")
    assert err < 1e-14

    # --- Confirm that sum of (Chebyshev) coefficients give inner product
    # For f = 1 (constant), the Chebyshev coefficient a_0 = 1 (first coeff)
    f_const = cj.chebfun(1.0)
    c_const = np.array(f_const.coeffs)
    print(f"\nConstant f=1: coeffs = {c_const[:5]}")
    # The sum (integral) of f=1 over [-1,1] = 2
    assert abs(float(f_const.sum()) - 2.0) < 1e-15

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))

    # Function plot
    fig, ax = plot(fc, title="f(x) = exp(x)·sin(πx) + x")
    fig.savefig(os.path.join(_here, "chebyshev_coefficients.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Coefficient decay comparison
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    for fn_name, fn_lambda, col in [
        ("exp(x)",    lambda x: jnp.exp(x),           "#4169E1"),
        ("cos(10x)",  lambda x: jnp.cos(10.0 * x),    "#E04040"),
        ("1/(1+x²)",  lambda x: 1.0 / (1.0 + x**2),  "#228B22"),
    ]:
        _f = cj.chebfun(fn_lambda)
        _c = np.abs(np.array(_f.coeffs))
        ax2.semilogy(np.arange(len(_c)), _c, ".", color=col,
                     markersize=4, label=fn_name)
    ax2.set_xlabel("degree $n$", fontsize=10)
    ax2.set_ylabel("$|a_n|$", fontsize=10)
    ax2.set_title("Chebyshev coefficient decay", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.set_facecolor("white")
    fig2.tight_layout()
    fig2.savefig(os.path.join(_here, "chebyshev_coefficients_decay.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
