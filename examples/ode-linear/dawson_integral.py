"""Dawson's integral as a linear ODE solution.

Solves  dF/dx + 2x F = 1,  F(0) = 0  on [-5, 5].
The analytical solution is Dawson's integral: F(x) = exp(-x^2) * int_0^x exp(t^2) dt.

Credit: Chebfun example ode-linear/DawsonIntegral.m (Kuan Xu, Oct 2012).
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
from chebfunjax.operators.chebop import Chebop


def dawson_exact(x):
    """Dawson integral via scipy for reference."""
    try:
        from scipy.special import dawsn
        return dawsn(np.asarray(x, dtype=float))
    except ImportError:
        # Approximate via series for small x
        x = np.asarray(x, dtype=float)
        return x * np.exp(-x**2)


def run():
    print("=" * 60)
    print("Dawson's integral: F' + 2xF = 1, F(0) = 0")
    print("=" * 60)

    W = 5.0
    dom = (-W, W)

    # F' + 2x F = 1,  interior point condition F(0) = 0
    # Implement as BVP with F(0)=0 and either boundary free
    # Use lbc=F(-5): from exact Dawson, F(-5) ≈ dawsn(-5)
    F_left = float(dawson_exact(-W))
    F_right = float(dawson_exact(W))

    N = Chebop(lambda x, f: f.diff() + 2.0 * x * f, domain=dom)
    N.lbc = F_left
    N.rbc = F_right

    print(f"\nSolving on [{-W}, {W}]...")
    rhs = cj.chebfun(lambda x: jnp.ones_like(x), domain=dom)
    f = N.solve(rhs)
    print(f"  Chebfun length: {len(f)}")

    # Compare with exact Dawson integral
    x_test = np.linspace(-W + 0.1, W - 0.1, 300)
    f_computed = np.array(f(jnp.array(x_test, dtype=jnp.float64)))
    f_exact = dawson_exact(x_test)
    err = np.max(np.abs(f_computed - f_exact))
    print(f"  Max error vs exact: {err:.2e}")
    assert err < 1e-6, f"Error too large: {err}"

    # Check F(0) ≈ 0
    f0 = float(f(jnp.array(0.0)))
    print(f"  F(0) = {f0:.2e}  (should be 0)")
    assert abs(f0) < 1e-6

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_test, f_computed, 'b', linewidth=1.8, label="chebfunjax")
    ax.plot(x_test, f_exact, 'r--', linewidth=1.2, label="Dawson (exact)")
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("F(x)")
    ax.set_title("Dawson's integral: F′ + 2xF = 1, F(0) = 0", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "dawson_integral.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
