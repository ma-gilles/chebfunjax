"""Nonlinear ODE with variable coefficients.

Demonstrates solving  u' - u*cos(u) = cos(4x) using scipy for the
nonlinear ODE, then wrapping the solution in a Chebfun to demonstrate
Chebfun arithmetic on the ODE residual.

The original Chebfun example used a built-in nonlinear solver
chebfunjax's Chebop Newton iteration struggles with first-order
nonlinear ODEs, so we use scipy as the ODE solver and Chebfun for
post-processing.

Credit: Chebfun example ode-nonlin/FourierCollocationNonLin.m (Hadrien Montanelli, Dec 2014).
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

from chebfunjax.operators.chebop import Chebop

def run():
    print("=" * 60)
    print("Nonlinear ODE: u' - u cos(u) = cos(4x)")
    print("=" * 60)

    dom = (0.0, float(jnp.pi))

    # Solve with scipy IVP
    def rhs_scipy(x, y):
        return [np.cos(4*x) + y[0]*np.cos(y[0])]

    sol_ref = solve_ivp(rhs_scipy, [0, float(jnp.pi)], [0.5],
                        t_eval=np.linspace(0, float(jnp.pi), 500), rtol=1e-10)
    print(f"\nScipy solution: u(0)=0.5, u(pi) = {float(sol_ref.y[0,-1]):.6f}")

    # Wrap in Chebfun
    t_dense = sol_ref.t
    u_dense = sol_ref.y[0]
    u = cj.chebfun(
        lambda x: jnp.interp(x, jnp.array(t_dense), jnp.array(u_dense)),
        domain=dom, n=128
    )
    print(f"  Chebfun length: {len(u)}")

    # Verify ODE residual using Chebfun differentiation
    x_test = jnp.linspace(0.05, float(jnp.pi) - 0.05, 200)
    res = u.diff()(x_test) - u(x_test) * jnp.cos(u(x_test)) - jnp.cos(4.0 * x_test)
    max_res = float(jnp.max(jnp.abs(res)))
    print(f"  ODE residual via Chebfun diff: {max_res:.2e}")
    assert max_res < 0.1  # jnp.interp has limited smoothness at boundaries

    # Use cj.cos on Chebfun for the residual check
    x_mid = jnp.linspace(0.2, float(jnp.pi) - 0.2, 100)
    u_cheb = u(x_mid)
    cos_u_cheb = jnp.cos(u_cheb)  # jnp.cos on JAX array (not Chebfun) is fine
    res_mid = u.diff()(x_mid) - u_cheb * cos_u_cheb - jnp.cos(4.0 * x_mid)
    max_res_mid = float(jnp.max(jnp.abs(res_mid)))
    print(f"  ODE residual (interior): {max_res_mid:.2e}")
    assert max_res_mid < 0.05

    # Linear version: u' + u = cos(4x), exact solution
    # Particular: u_p = (cos(4x) + 4*sin(4x))/17
    # General: u = (cos(4x) + 4*sin(4x))/17 + C*exp(-x)
    # With u(0)=0: 1/17 + C = 0 => C = -1/17
    # => u(x) = (cos(4x) + 4*sin(4x) - exp(-x))/17
    print("\nLinear reference: u' + u = cos(4x)")
    exact_lin = lambda x: (jnp.cos(4*x) + 4*jnp.sin(4*x) - jnp.exp(-x)) / 17.0
    u_lin = cj.chebfun(exact_lin, domain=dom)
    # Verify ODE residual
    res_lin = u_lin.diff()(x_mid) + u_lin(x_mid) - jnp.cos(4.0 * x_mid)
    max_res_lin = float(jnp.max(jnp.abs(res_lin)))
    print(f"  ODE residual: {max_res_lin:.2e}")
    assert max_res_lin < 1e-10

    # Solve linear version with Chebop as BVP
    N2 = Chebop(lambda x, u: u.diff() + u, domain=dom)
    N2.lbc = float(exact_lin(jnp.array(0.0)))
    N2.rbc = float(exact_lin(jnp.array(float(jnp.pi))))
    rhs2 = cj.chebfun(lambda x: jnp.cos(4.0 * x), domain=dom)
    u2 = N2.solve(rhs2)
    err2 = float(jnp.max(jnp.abs(u2(x_mid) - u_lin(x_mid))))
    print(f"  Chebop solution length: {len(u2)}, error vs exact: {err2:.2e}")
    assert err2 < 1e-8

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(0.0, float(jnp.pi), 300)
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(x_plot, u(x_plot), 'b', linewidth=1.8, label="scipy+Chebfun")
    axes[0].plot(sol_ref.t, sol_ref.y[0], color='#D95319', linestyle='--', linewidth=1.2, label="scipy", alpha=0.7)
    axes[0].set_title("u′ − u cos(u) = cos(4x)", fontsize=10)
    axes[0].set_xticks([0, np.pi/2, np.pi])
    axes[0].set_xticklabels(["0", "π/2", "π"])
    axes[0].legend(fontsize=8)

    axes[1].plot(x_plot, u2(x_plot), 'b', linewidth=1.8, label="Chebop")
    axes[1].plot(x_plot, u_lin(x_plot), color='#D95319', linestyle='--', linewidth=1.2, label="exact", alpha=0.7)
    axes[1].set_title("u′ + u = cos(4x)  (linear, Chebop)", fontsize=10)
    axes[1].set_xticks([0, np.pi/2, np.pi])
    axes[1].set_xticklabels(["0", "π/2", "π"])
    axes[1].legend(fontsize=8)

    fig.suptitle("Nonlinear and linear first-order ODEs", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "fourier_nonlin.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
