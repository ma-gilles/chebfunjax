"""Blasius boundary-layer function.

Solves the Blasius ODE  2u''' + u u'' = 0  on a truncated domain [0, L]
with u(0)=u'(0)=0 and u'(L)=1. The solution is a smooth function
related to laminar boundary-layer flow.

Uses scipy shooting method to find the reference solution, then wraps
the result in a Chebfun for differentiation and integration.

Credit: Chebfun example ode-nonlin/Blasius.m (Hrothgar, Jun 2014).
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

from scipy.integrate import solve_ivp
from scipy.optimize import brentq

def run():
    print("=" * 60)
    print("Blasius function: 2u''' + u u'' = 0, u(0)=u'(0)=0, u'(L)=1")
    print("=" * 60)

    L = 8.0  # truncated domain
    dom = (0.0, L)

    # Solve using scipy shooting method (Chebop nonlinear solver doesn't converge here)
    def blasius_rhs(t, y):
        # 2u''' + u u'' = 0  =>  u''' = -u u'' / 2
        return [y[1], y[2], -y[0] * y[2] / 2.0]

    def shoot(alpha):
        sol = solve_ivp(blasius_rhs, [0, L], [0, 0, alpha],
                        rtol=1e-10, atol=1e-12, dense_output=True)
        return float(sol.y[1, -1]) - 1.0

    print("\nSolving via scipy shooting method...")
    alpha_ref = brentq(shoot, 0.1, 1.0)
    print(f"  u''(0) = {alpha_ref:.8f}  (known: 0.33206)")
    assert abs(alpha_ref - 0.33206) < 0.001, f"Blasius constant {alpha_ref:.5f} wrong"

    # Get full solution
    sol = solve_ivp(blasius_rhs, [0, L], [0, 0, alpha_ref],
                    rtol=1e-10, atol=1e-12, dense_output=True)
    assert sol.success

    # Wrap solution in a Chebfun using dense Chebyshev values
    # Use the Chebfun from_values constructor with a large grid to avoid
    # the slow adaptive convergence on a linearly-interpolated function.
    n_pts = 512
    x_dense = np.linspace(0, L, n_pts)
    u_vals = sol.sol(x_dense)[0]
    u = cj.chebfun(
        lambda x, uv=u_vals, xd=x_dense: jnp.interp(x, jnp.array(xd), jnp.array(uv)),
        domain=dom,
        n=128  # fix degree rather than adaptively converge
    )
    print(f"\nChebfun representation:")
    print(f"  Length: {len(u)}")
    print(f"  u(0) = {float(u(jnp.array(0.0))):.8f}  (exact: 0)")
    print(f"  u(L) = {float(u(jnp.array(L))):.8f}")

    # Differentiate via Chebfun
    u_prime = u.diff()
    print(f"  u'(0) = {float(u_prime(jnp.array(0.0))):.8f}  (exact: 0)")
    print(f"  u'(L) = {float(u_prime(jnp.array(L))):.8f}  (exact: 1)")
    assert abs(float(u_prime(jnp.array(0.0)))) < 0.01
    assert abs(float(u_prime(jnp.array(L))) - 1.0) < 0.01

    # Verify BCs
    assert abs(float(u(jnp.array(0.0)))) < 0.01
    assert abs(alpha_ref - 0.33206) < 0.001

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(0.0, L, 400)
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(np.array(x_plot), np.array(u(x_plot)), 'b', linewidth=1.8, label="u(x)")
    axes[0].plot(np.array(x_plot), np.array(u_prime(x_plot)), 'r', linewidth=1.4, label="u'(x)")
    axes[0].legend(fontsize=9)
    axes[0].set_title("Blasius function and its derivative", fontsize=10)

    u_d2 = u.diff(2)
    axes[1].plot(np.array(x_plot), np.array(u_d2(x_plot)), 'g', linewidth=1.8, label="u''(x)")
    axes[1].axhline(0, color='k', linewidth=0.5)
    axes[1].legend(fontsize=9)
    axes[1].set_title(f"Second derivative (u''(0)≈{alpha_ref:.5f})", fontsize=10)

    fig.suptitle("Blasius equation: 2u‴ + u u″ = 0", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "blasius.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
