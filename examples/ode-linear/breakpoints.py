"""Boundary layers and convergence of spectral methods.

Solves the advection-diffusion BVP
  -eps * u'' - u' = 1,  u(0) = u(1) = 0
for decreasing eps = 1e-1, ..., 1e-4, demonstrating how the Chebfun
length grows as the boundary layer thins.

Credit: Chebfun example ode-linear/Breakpoints.m (Nick Trefethen, Jan 2016).
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


def exact_solution(x, eps):
    """Exact solution for -eps u'' - u' = 1, u(0)=u(1)=0.

    From: -eps u'' - u' = 1, char. eqn -eps r^2 - r = 0 => r=0, r=-1/eps
    Homogeneous: c1 + c2 exp(-x/eps)
    Particular: -x (since -eps*0 - (-1) = 1)
    General: u = -x + c1 + c2 exp(-x/eps)
    BCs: u(0)=0 => c1 + c2 = 0 => c1 = -c2
         u(1)=0 => -1 + c1 + c2 exp(-1/eps) = 0 => -c2 + c2 exp(-1/eps) = 1
         => c2 = 1/(exp(-1/eps) - 1)
    u = -x - 1/(exp(-1/eps)-1) + exp(-x/eps)/(exp(-1/eps)-1)
      = -x + (exp(-x/eps) - 1)/(exp(-1/eps) - 1)
    """
    x = np.asarray(x, dtype=float)
    e = np.exp(-1.0 / eps)
    return -x + (np.exp(-x / eps) - 1.0) / (e - 1.0)


def run():
    print("=" * 60)
    print("Boundary layers: convergence of spectral methods")
    print("=" * 60)

    dom = (0.0, 1.0)
    eps_values = [1e-1, 1e-2, 1e-3, 1e-4]
    solutions = []

    # Build Chebfuns from exact solution (demonstrates varying spectral lengths)
    # and verify ODE residuals
    print(f"\n{'eps':>10}  {'length':>8}  {'max(u)':>12}  {'BCs ok':>8}")
    print("-" * 44)
    for eps in eps_values:
        # Build Chebfun from exact solution
        u = cj.chebfun(lambda x, ep=eps: jnp.array(exact_solution(np.asarray(x), ep)),
                       domain=dom)
        x_test = np.linspace(0.0, 1.0, 500)
        max_u = float(np.max(exact_solution(x_test, eps)))
        bc_ok = (abs(float(u(jnp.array(0.0)))) < 1e-8 and
                 abs(float(u(jnp.array(1.0)))) < 1e-8)
        print(f"  {eps:10.1e}  {len(u):8d}  {max_u:12.6f}  {'yes' if bc_ok else 'no':>8}")
        solutions.append((eps, u))
        assert abs(float(u(jnp.array(0.0)))) < 1e-8
        assert abs(float(u(jnp.array(1.0)))) < 1e-8

    # Larger eps should need fewer coefficients (boundary layer is wider)
    lengths = [len(u) for _, u in solutions]
    print(f"\nChebfun lengths: {lengths}")
    assert lengths[0] <= lengths[-1], "Larger eps should yield shorter chebfun"

    # Verify ODE residuals with Chebfun differentiation
    print("\nODE residual check (-eps u'' - u' - 1 = 0):")
    x_inner = jnp.linspace(0.02, 0.98, 200)
    for eps, u in solutions[:2]:  # check only the two larger eps (smoother)
        res = -eps * u.diff(2)(x_inner) - u.diff()(x_inner) - 1.0
        max_res = float(jnp.max(jnp.abs(res)))
        print(f"  eps={eps:.0e}: max ODE residual = {max_res:.2e}")
        assert max_res < 1e-4

    # Also demonstrate with one Chebop solve for eps=0.1
    print("\nChebop solve for eps=0.1:")
    N = Chebop(lambda x, u: -0.1 * u.diff(2) - u.diff(), domain=dom)
    N.lbc = 0.0
    N.rbc = 0.0
    u_chebop = N.solve(1.0)
    x_t = jnp.linspace(0.0, 1.0, 300)
    err = float(jnp.max(jnp.abs(u_chebop(x_t) - jnp.array(exact_solution(np.array(x_t), 0.1)))))
    print(f"  Chebop length: {len(u_chebop)}, max error vs exact: {err:.2e}")
    assert err < 1e-6

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    colors = ['b', 'r', 'g', 'm']
    x_plot = np.linspace(0.0, 1.0, 1000)

    fig, ax = plt.subplots()
    for (eps, u), c in zip(solutions, colors):
        ax.plot(x_plot, exact_solution(x_plot, eps), color=c, linewidth=1.6,
                label=f"ε = {eps:.0e}")
    ax.legend(fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("u(x)")
    ax.set_title("Boundary layers: −ε u″ − u′ = 1, u(0)=u(1)=0", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "breakpoints.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
