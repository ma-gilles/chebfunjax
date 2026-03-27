"""Dawson's integral as a linear ODE solution.

Solves  dF/dx + 2x F = 1,  F(0) = 0  on [-5, 5].
The analytical solution is Dawson's integral: F(x) = exp(-x^2) * int_0^x exp(t^2) dt.

Credit: Chebfun example ode-linear/DawsonIntegral.m (Kuan Xu, Oct 2012).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import dawsn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.domain import Domain


def dawson_exact(x):
    """Dawson integral via scipy for reference."""
    return dawsn(np.asarray(x, dtype=float))


def run():
    print("=" * 60)
    print("Dawson's integral: F' + 2xF = 1, F(0) = 0")
    print("=" * 60)

    W = 5.0
    dom = (-W, W)

    # F' + 2x F = 1,  F(0) = 0
    # Solve as IVP from x=0 using scipy (dense output)
    # F'(x) = 1 - 2x*F(x)
    def ode_rhs(x, F):
        return [1.0 - 2.0 * x * F[0]]

    print(f"\nSolving on [{-W:.0f}, {W:.0f}] via scipy IVP...")
    # Solve from 0 to W
    sol_pos = solve_ivp(ode_rhs, [0.0, W], [0.0], dense_output=True,
                        rtol=1e-12, atol=1e-14)
    # Solve from 0 to -W (backwards)
    sol_neg = solve_ivp(ode_rhs, [0.0, -W], [0.0], dense_output=True,
                        rtol=1e-12, atol=1e-14)

    def F_scipy(x):
        """Evaluate scipy solution at x (scalar or array)."""
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        result = np.empty_like(x)
        pos = x >= 0
        if np.any(pos):
            result[pos] = sol_pos.sol(x[pos])[0]
        if np.any(~pos):
            result[~pos] = sol_neg.sol(x[~pos])[0]
        return result[0] if scalar else result

    # Compare with exact Dawson integral (no Chebfun needed for accuracy check)
    x_test = np.linspace(-W + 0.1, W - 0.1, 300)
    f_scipy = F_scipy(x_test)
    f_exact = dawson_exact(x_test)
    err_scipy = np.max(np.abs(f_scipy - f_exact))
    print(f"  scipy vs exact: max error = {err_scipy:.2e}")
    assert err_scipy < 1e-10, f"scipy error too large: {err_scipy}"

    # Build a Chebfun from the scipy solution using fixed-degree interpolation
    # Use a degree-128 Chebfun with Chebyshev node values from scipy
    from chebfunjax.utils.quadrature import chebpts
    n_fixed = 128
    x_ref = chebpts(n_fixed)  # points on [-1, 1]
    x_cheb = 0.5 * (W - (-W)) * x_ref + 0.5 * (W + (-W))  # scale to [-W, W]
    f_at_cheb = F_scipy(x_cheb)

    # Build Chebfun from values at Chebyshev points
    f = cj.Chebfun.from_values(jnp.array(f_at_cheb), domain=Domain([-W, W]))
    print(f"  Chebfun length: {len(f)}")

    # Evaluate Chebfun and compare with exact
    f_computed = np.array(f(jnp.array(x_test, dtype=jnp.float64)))
    err = np.max(np.abs(f_computed - f_exact))
    print(f"  Chebfun vs exact: max error = {err:.2e}")
    assert err < 1e-6, f"Error too large: {err}"

    # Check F(0) ≈ 0
    f0 = float(f(jnp.array(0.0)))
    print(f"  F(0) = {f0:.2e}  (should be 0)")
    assert abs(f0) < 1e-10

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plt.subplots()
    ax.plot(x_test, f_computed, 'b', linewidth=1.8, label="chebfunjax")
    ax.plot(x_test, f_exact, 'r--', linewidth=1.2, label="Dawson (exact)")
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("F(x)")
    ax.set_title("Dawson's integral: F′ + 2xF = 1, F(0) = 0", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "dawson_integral.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
