"""Lane-Emden equation (linear case n=0 and n=1).

The Lane-Emden equation is  x u'' + 2 u' + x u^n = 0,  u'(0)=0, u(0)=1.
For n=0 (linear): u = 1 - x^2/6 (exact).
For n=1 (linear in u): u = sin(x)/x.

Credit: Chebfun example ode-linear/LaneEmden.m (Alex Townsend, May 2011).
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
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.domain import Domain
from chebfunjax.utils.quadrature import chebpts


def run():
    print("=" * 60)
    print("Lane-Emden equation: x u'' + 2 u' + x u^n = 0")
    print("=" * 60)

    # Lane-Emden IVP: u(0)=1, u'(0)=0
    # Rewrite as first-order system: let v = u'
    # u' = v,  v' = -2v/x - u^n
    # Near x=0: v'(0) = -1/3 for n=0 and n=1
    # Initialize at small eps to avoid x=0 singularity
    eps_x = 1e-6

    # ----------------------------------------------------------------
    # n=0: u'' + (2/x) u' + 1 = 0,  exact u = 1 - x^2/6
    R0 = float(np.sqrt(6.0))  # first zero
    exact_n0 = lambda x: 1.0 - np.asarray(x)**2 / 6.0

    print(f"\nCase n=0 (exact: u = 1 - x^2/6), R = sqrt(6) = {R0:.4f}")

    def rhs_n0(x, y):
        u, v = y
        return [v, -2.0 * v / x - 1.0]

    y0_n0 = [1.0 - eps_x**2/6.0, -eps_x/3.0]
    sol_n0 = solve_ivp(rhs_n0, (eps_x, R0), y0_n0, dense_output=True,
                       rtol=1e-12, atol=1e-14)

    x_test0 = np.linspace(eps_x * 10, R0 - 0.01, 300)
    u_computed0 = sol_n0.sol(x_test0)[0]
    err0 = np.max(np.abs(u_computed0 - exact_n0(x_test0)))
    print(f"  Max error vs exact: {err0:.2e}")
    assert err0 < 1e-8

    # Build Chebfun via Chebyshev interpolation
    dom0 = (eps_x * 10, R0 - 0.01)
    n_ch = 64
    x_ref = chebpts(n_ch)
    x_ch0 = 0.5 * (dom0[1] - dom0[0]) * x_ref + 0.5 * (dom0[0] + dom0[1])
    u0_vals = sol_n0.sol(x_ch0)[0]
    u0 = cj.Chebfun.from_values(jnp.array(u0_vals), domain=Domain(list(dom0)))
    print(f"  Chebfun length: {len(u0)}")

    # ----------------------------------------------------------------
    # n=1: u'' + (2/x) u' + u = 0,  exact u = sin(x)/x
    R1 = float(np.pi)
    exact_n1 = lambda x: np.sin(np.asarray(x, dtype=float)) / np.asarray(x, dtype=float)

    print(f"\nCase n=1 (exact: u = sin(x)/x), R = pi = {R1:.4f}")

    def rhs_n1(x, y):
        u, v = y
        return [v, -2.0 * v / x - u]

    y0_n1 = [1.0 - eps_x**2/6.0, -eps_x/3.0]
    sol_n1 = solve_ivp(rhs_n1, (eps_x, R1), y0_n1, dense_output=True,
                       rtol=1e-12, atol=1e-14)

    x_test1 = np.linspace(eps_x * 10, R1 - 0.01, 300)
    u1_computed = sol_n1.sol(x_test1)[0]
    err1 = np.max(np.abs(u1_computed - exact_n1(x_test1)))
    print(f"  Max error vs exact: {err1:.2e}")
    assert err1 < 1e-8

    dom1 = (eps_x * 10, R1 - 0.01)
    x_ch1 = 0.5 * (dom1[1] - dom1[0]) * x_ref + 0.5 * (dom1[0] + dom1[1])
    u1_vals = sol_n1.sol(x_ch1)[0]
    u1 = cj.Chebfun.from_values(jnp.array(u1_vals), domain=Domain(list(dom1)))
    print(f"  Chebfun length: {len(u1)}")

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    x_plot0 = jnp.linspace(float(dom0[0]), R0, 300)
    axes[0].plot(x_plot0, u0(x_plot0), 'b', linewidth=1.8, label="chebfunjax")
    axes[0].plot(np.linspace(dom0[0], R0, 300),
                 exact_n0(np.linspace(dom0[0], R0, 300)),
                 'r--', linewidth=1.2, label="exact 1−x²/6")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("u(x)")
    axes[0].set_title("Lane-Emden n=0: x u″+2u′+x = 0", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    x_plot1 = jnp.linspace(float(dom1[0]), R1, 300)
    axes[1].plot(x_plot1, u1(x_plot1), 'b', linewidth=1.8, label="chebfunjax")
    axes[1].plot(np.linspace(dom1[0], R1, 300),
                 exact_n1(np.linspace(dom1[0], R1, 300)),
                 'r--', linewidth=1.2, label="exact sin(x)/x")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("u(x)")
    axes[1].set_title("Lane-Emden n=1: x u″+2u′+xu = 0", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Lane-Emden equation (linear cases n=0,1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "lane_emden_linear.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
