"""1D Poisson equation BVP.

Demonstrates solving -u'' = f on [0, 1] with Dirichlet BCs for
several right-hand sides with known exact solutions.

Credit: Inspired by Chebfun ode-linear examples.
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
from chebfunjax.plotting import plot
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("1D Poisson equation BVP")
    print("=" * 60)

    # --- Example 1: -u'' = 1, u(0)=0, u(1)=0 ------------------------
    # Exact: u = x(1-x)/2
    print("\n--- Example 1: -u'' = 1, u(0) = u(1) = 0 ---")
    N1 = Chebop(lambda x, u: -u.diff(2), domain=(0.0, 1.0))
    N1.lbc = 0.0
    N1.rbc = 0.0
    u1 = N1.solve(1.0)
    x_test1 = jnp.linspace(0.0, 1.0, 200)
    exact1 = x_test1 * (1.0 - x_test1) / 2.0
    err1 = float(jnp.max(jnp.abs(u1(x_test1) - exact1)))
    print(f"  ||u - x(1-x)/2||_inf = {err1:.2e}")
    assert err1 < 1e-12
    # u(0.5) = 0.5 * 0.5 / 2 = 0.125
    assert abs(float(u1(jnp.array(0.5))) - 0.125) < 1e-13

    # --- Example 2: -u'' = pi^2 * sin(pi*x), u(0)=0, u(1)=0 --------
    # Exact: u = sin(pi*x)
    print("\n--- Example 2: -u'' = pi^2*sin(pi*x), u(0)=u(1)=0 ---")
    pi = float(jnp.pi)
    N2 = Chebop(lambda x, u: -u.diff(2), domain=(0.0, 1.0))
    N2.lbc = 0.0
    N2.rbc = 0.0
    rhs2 = cj.chebfun(lambda x: pi**2 * jnp.sin(pi * x), domain=(0.0, 1.0))
    u2 = N2.solve(rhs2)
    x_test2 = jnp.linspace(0.0, 1.0, 200)
    exact2 = jnp.sin(pi * x_test2)
    err2 = float(jnp.max(jnp.abs(u2(x_test2) - exact2)))
    print(f"  ||u - sin(pi*x)||_inf = {err2:.2e}")
    assert err2 < 1e-10

    # --- Example 3: -u'' = exp(x), u(0) = 0, u(1) = 2e-3+1-e -------
    # Particular solution: u_p = -exp(x)
    # BC: u_p(0) = -1 + C2 = 0 => C2 = 1
    # u_p(1) = -e + C1 + 1 = (2e-3) => C1 = 2e - 3
    # Exact: u = (2e-3)*x + 1 - exp(x)
    print("\n--- Example 3: -u'' = exp(x) ---")
    e = float(jnp.exp(jnp.array(1.0)))
    exact3 = lambda x: (2*e - 3)*x + 1.0 - jnp.exp(x)
    N3 = Chebop(lambda x, u: -u.diff(2), domain=(0.0, 1.0))
    N3.lbc = 0.0
    N3.rbc = float(exact3(jnp.array(1.0)))
    rhs3 = cj.chebfun(lambda x: jnp.exp(x), domain=(0.0, 1.0))
    u3 = N3.solve(rhs3)
    x_test3 = jnp.linspace(0.0, 1.0, 200)
    exact3_vals = exact3(x_test3)
    err3 = float(jnp.max(jnp.abs(u3(x_test3) - exact3_vals)))
    print(f"  ||u - exact||_inf = {err3:.2e}")
    assert err3 < 1e-10

    # --- Example 4: -u'' = 1 on [-1, 1], u(±1) = 0 ------------------
    # Exact: u = (1 - x^2) / 2; u(0) = 0.5
    print("\n--- Example 4: -u'' = 1 on [-1,1], u(±1)=0 ---")
    N4 = Chebop(lambda x, u: -u.diff(2), domain=(-1.0, 1.0))
    N4.lbc = 0.0
    N4.rbc = 0.0
    u4 = N4.solve(1.0)
    mid_val = float(u4(jnp.array(0.0)))
    print(f"  u(0) = {mid_val:.15f}  (exact: 0.5)")
    assert abs(mid_val - 0.5) < 1e-12

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(u1, title="Poisson equation: −u″ = f on [0,1]",
                   label="u₁ (f=1)")
    plot(u2, ax=ax, color="#E04040", label="u₂ (f=π²sin(πx))")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "poisson_equation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
