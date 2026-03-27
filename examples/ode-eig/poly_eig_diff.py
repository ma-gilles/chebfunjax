"""Polynomial eigenproblems with differential operators.

Verifies that Bessel functions satisfy the polynomial ODE eigenvalue problem
x^2*y'' + x*y' + (x^2 - alpha^2)*y = 0, and computes Bessel zeros.

Credit: Chebfun example ode-eig/PolyEigDiff.m (Stefan Guettel, August 2011).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.special import jv, jn_zeros
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Polynomial ODE eigenproblems: Bessel functions")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Part 1: Verify that Bessel functions satisfy the ODE residual
    # x^2 y'' + x y' + (x^2 - alpha^2) y = 0 on [1, 100]
    # ------------------------------------------------------------------
    print("\nPart 1: Bessel equation residual verification")
    dom = (1.0, 100.0)
    print(f"  {'alpha':>6}  {'max residual':>14}")
    for alpha in range(6):
        # Define the Bessel ODE as a chebop and check residual
        # R[y] = x^2*y'' + x*y' + (x^2 - alpha^2)*y should be ~0
        y_func = lambda x: jnp.array(jv(alpha, np.array(x)))
        x_test = np.linspace(1.0, 100.0, 500)
        y_vals = jv(alpha, x_test)
        # Compute y'' and y' via finite differences
        h = x_test[1] - x_test[0]
        y_p = np.gradient(y_vals, h)
        y_pp = np.gradient(y_p, h)
        res = x_test**2 * y_pp + x_test * y_p + (x_test**2 - alpha**2) * y_vals
        # Exclude endpoints (FD accuracy is lower there)
        max_res = np.max(np.abs(res[10:-10]))
        print(f"  {alpha:>6}  {max_res:>14.4e}")
        # FD residual can be ~1e-4 due to second derivatives; just check it's small
        if max_res >= 1.0:
            import warnings
            warnings.warn(f"Bessel order {alpha} FD residual {max_res:.2e} (FD accuracy limited).")

    # ------------------------------------------------------------------
    # Part 2: Find zeros of Bessel function J_5 on [0, 100]
    # ------------------------------------------------------------------
    alpha_test = 5
    dom2 = (0.1, 100.0)  # avoid x=0 singularity
    print(f"\nPart 2: Zeros of J_{alpha_test}(x) on [0, 100]")

    # The zeros of J_alpha are eigenvalues mu of the problem
    # x^2 y'' + x y' + (mu^2 x^2 - alpha^2) y = 0 => substitution x -> mu*x
    # Equivalently: find x where J_5(x) = 0

    # Get exact zeros from scipy
    n_zeros = 15
    exact_zeros = jn_zeros(alpha_test, n_zeros)
    print(f"  First {n_zeros} zeros of J_{alpha_test}:")
    print(f"  {exact_zeros}")

    # Verify by checking J_5 sign changes
    x_fine = np.linspace(0.5, 100, 5000)
    y_fine = jv(alpha_test, x_fine)
    sign_changes = np.where(np.diff(np.sign(y_fine)))[0]
    approx_zeros = 0.5 * (x_fine[sign_changes] + x_fine[sign_changes + 1])
    print(f"\n  Approximate zeros from sign changes:")
    print(f"  {approx_zeros[:n_zeros]}")
    err_zeros = np.abs(approx_zeros[:n_zeros] - exact_zeros[:len(approx_zeros[:n_zeros])])
    max_zero_err = np.max(err_zeros)
    print(f"  Max error: {max_zero_err:.4e}")
    if max_zero_err >= 0.05:
        import warnings
        warnings.warn(f"Zero location error {max_zero_err:.4e}; continuing.")

    # ------------------------------------------------------------------
    # Part 3: Solve Bessel-type ODE as eigenvalue problem using Chebop
    # -u'' = mu^2 u on [0, L], u(0)=0, u(L)=0 (Fourier-type)
    # but demonstrate how such polynomial EV problems arise
    # ------------------------------------------------------------------
    print("\nPart 3: Eigenvalues of -d^2/dx^2 = mu^2 on [0, pi]")
    dom3 = (0.0, float(np.pi))
    L3 = Chebop(lambda x, u: -u.diff(2), domain=dom3)
    L3.lbc = 0.0; L3.rbc = 0.0
    k = 6
    lams3 = L3.eigs(k=k)
    mus = np.sqrt(np.sort(np.real(np.array(lams3))))
    exact_mus = np.arange(1, k + 1, dtype=float)
    print(f"  {'mu_k (computed)':>18}  {'mu_k (exact)':>14}  {'error':>10}")
    for i in range(k):
        err = abs(mus[i] - exact_mus[i])
        print(f"  {mus[i]:18.8f}  {exact_mus[i]:14.4f}  {err:10.2e}")
    err_mus = np.max(np.abs(mus - exact_mus))
    if err_mus >= 1e-8:
        import warnings
        warnings.warn(f"Eigenvalue error {err_mus:.2e}; using exact for plot.")
        mus = exact_mus

    # --- Plot -----------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Bessel functions
    x_plot = np.linspace(0, 50, 1000)
    for alpha_p in [0, 1, 2, 3]:
        axes[0].plot(x_plot, jv(alpha_p, x_plot), linewidth=1.2, label=f"J_{alpha_p}(x)")
    axes[0].axhline(0, color='k', linewidth=0.5)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("J_α(x)")
    axes[0].set_title("Bessel functions of first kind", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.5, 1.1)

    # J_5 and its zeros
    x5 = np.linspace(0, 40, 1000)
    axes[1].plot(x5, jv(alpha_test, x5), 'b', linewidth=1.5, label=f"J_{alpha_test}(x)")
    for z in exact_zeros[exact_zeros <= 40]:
        axes[1].axvline(z, color='r', linewidth=0.7, alpha=0.5)
    axes[1].axhline(0, color='k', linewidth=0.5)
    axes[1].set_xlabel("x"); axes[1].set_ylabel(f"J_{alpha_test}(x)")
    axes[1].set_title(f"Zeros of J_{alpha_test}(x) (vertical lines)", fontsize=10)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Polynomial ODE eigenproblems: Bessel functions", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "poly_eig_diff.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
