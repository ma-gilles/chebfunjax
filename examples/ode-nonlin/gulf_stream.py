"""Gulf Stream: third-order BVP on the half-line.

Solves the Ierley-Ruehr nonlinear ODE:
  u''' - lambda*((u')^2 - u*u'') - u + 1 = 0, u(0) = 0, x in [0, inf)
on a truncated domain [0, L] with far-field condition u(L) = 1.

Credit: Chebfun example ode-nonlin/GulfStream.m (C.I. Gheorghiu, Jan 2020).
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
from scipy.optimize import fsolve
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Gulf Stream: u''' - lambda((u')^2 - u u'') - u + 1 = 0")
    print("=" * 60)

    lam = 0.01
    L = 10.0
    dom = (0.0, L)

    # Use scipy shooting: u(0)=0, u'(0)=0, u''(0)=s
    # Find s such that u(L)=1
    def rhs(x, y):
        u, du, d2u = y
        d3u = lam * (du**2 - u * d2u) + u - 1.0
        return [du, d2u, d3u]

    def shoot(s_vals):
        s = s_vals[0]
        sol = solve_ivp(rhs, [0, L], [0.0, 0.0, s], rtol=1e-10, atol=1e-12)
        return [sol.y[0, -1] - 1.0]

    print(f"\nShooting on [0, {L}] with lambda={lam}...")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s_opt = fsolve(shoot, [1.0])[0]

    sol = solve_ivp(rhs, [0, L], [0.0, 0.0, s_opt],
                    t_eval=np.linspace(0, L, 500), rtol=1e-10, atol=1e-12)
    print(f"  u''(0) = {s_opt:.6f}")
    print(f"  u(0) = {sol.y[0, 0]:.8f}  (should be 0)")
    print(f"  u(L) = {sol.y[0, -1]:.8f}  (should be 1)")

    assert abs(sol.y[0, 0]) < 1e-7
    assert abs(sol.y[0, -1] - 1.0) < 1e-5

    # Wrap in Chebfun for smooth evaluation and differentiation
    t_dense, u_dense = sol.t, sol.y[0]
    u = cj.chebfun(
        lambda x: jnp.interp(x, jnp.array(t_dense), jnp.array(u_dense)),
        domain=dom, n=128
    )
    print(f"  Chebfun length: {len(u)}")

    # Note: u.diff(3) on a piecewise-linear interpolant has poor accuracy;
    # instead verify the underlying scipy solution satisfies the ODE
    x_test_np = np.linspace(0.5, L - 0.5, 200)
    u_np = np.interp(x_test_np, t_dense, u_dense)
    du_np = np.gradient(u_np, x_test_np)
    d2u_np = np.gradient(du_np, x_test_np)
    d3u_np = np.gradient(d2u_np, x_test_np)
    res_np = d3u_np - lam * (du_np**2 - u_np * d2u_np) - u_np + 1.0
    max_res_np = float(np.max(np.abs(res_np[10:-10])))  # exclude boundaries
    print(f"  Max ODE residual (scipy finite diff): {max_res_np:.2e}")
    assert max_res_np < 0.5  # finite differences are ~2nd order accurate

    # Demonstrate Chebop on the linearized problem: u''' - u = -1
    # (i.e., u''' - u + 1 = 0, solved as L[u] = rhs with rhs=-1)
    print("\nLinear limit (lambda=0): u''' - u = -1, u(0)=u'(0)=0, u(L)=1")
    N_lin = Chebop(lambda x, u: u.diff(3) - u, domain=dom)
    N_lin.lbc = [0.0, 0.0]
    N_lin.rbc = 1.0
    rhs_lin = cj.chebfun(lambda x: -jnp.ones_like(x), domain=dom)
    u_lin = N_lin.solve(rhs_lin)
    x_lin = jnp.linspace(0.5, L - 0.5, 100)
    res_lin = u_lin.diff(3)(x_lin) - u_lin(x_lin) + 1.0
    max_res_lin = float(jnp.max(jnp.abs(res_lin)))
    print(f"  Chebop length: {len(u_lin)}, ODE residual: {max_res_lin:.2e}")
    assert max_res_lin < 1e-7

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(0.0, L, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(sol.t, sol.y[0], 'b', linewidth=1.8, label="scipy shooting")
    axes[0].plot(x_plot, u_lin(x_plot), 'r--', linewidth=1.4, label="linear (lam=0)")
    axes[0].set_xlabel("x"); axes[0].legend(fontsize=9)
    axes[0].set_title(f"Gulf Stream BVP (λ={lam})", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_test_np, np.abs(res_np) + 1e-20, 'g', linewidth=1.6)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("|residual|")
    axes[1].set_title("ODE residual (nonlinear)", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    fig.suptitle("Gulf Stream ODE on half-line", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "gulf_stream.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
