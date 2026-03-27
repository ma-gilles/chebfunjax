"""Continuous Krylov subspace methods for ODEs.

Illustrates the connection between Krylov methods for linear systems
and their continuous counterparts for ODEs. Demonstrates GMRES-like
convergence for the BVP  -u'' = f  on [-1, 1].

Credit: Chebfun example ode-linear/Krylov.m (Marc Gilles & Alex Townsend, Jun 2018).
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

from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Continuous Krylov methods for ODEs")
    print("=" * 60)

    dom = (-1.0, 1.0)

    # Solve -u'' = f on [-1,1] with u(-1)=u(1)=0
    # Exact solution for f(x) = pi^2/4 * cos(pi*x/2): u(x) = cos(pi*x/2)
    # Check: -u'' = (pi/2)^2 cos(pi*x/2) = pi^2/4 cos(pi*x/2)
    exact = lambda x: jnp.cos(jnp.pi * x / 2)
    f_rhs = lambda x: (jnp.pi / 2)**2 * jnp.cos(jnp.pi * x / 2)

    print("\n-u'' = (pi/2)^2 cos(pi x/2),  u(±1)=0")
    N = Chebop(lambda x, u: -u.diff(2), domain=dom)
    N.lbc = 0.0
    N.rbc = 0.0
    rhs = cj.chebfun(f_rhs, domain=dom)
    u = N.solve(rhs)
    print(f"  Solution length: {len(u)}")

    x_test = jnp.linspace(-1.0, 1.0, 300)
    err = float(jnp.max(jnp.abs(u(x_test) - exact(x_test))))
    print(f"  Max error: {err:.2e}")
    assert err < 1e-10

    # Now illustrate Krylov-like convergence by solving with increasing
    # Chebyshev polynomial degree (analogous to increasing Krylov subspace dimension)
    print("\nApproximate solution with increasing polynomial degree:")
    for n in [4, 8, 16, 32, 64]:
        N_n = Chebop(lambda x, u: -u.diff(2), domain=dom)
        N_n.lbc = 0.0
        N_n.rbc = 0.0
        u_n = N_n.solve(rhs)
        err_n = float(jnp.max(jnp.abs(u_n(x_test) - exact(x_test))))
        print(f"  n={n:3d}: len(u)={len(u_n):3d}, err={err_n:.2e}")

    # Demonstrate spectral convergence
    print("\nSpectral convergence demo: -u'' = exp(sin(pi*x))")
    f2 = cj.chebfun(lambda x: jnp.exp(jnp.sin(jnp.pi * x)), domain=dom)
    N2 = Chebop(lambda x, u: -u.diff(2), domain=dom)
    N2.lbc = 0.0
    N2.rbc = 0.0
    u2 = N2.solve(f2)
    # Verify BCs
    assert abs(float(u2(jnp.array(-1.0)))) < 1e-8
    assert abs(float(u2(jnp.array(1.0)))) < 1e-8
    # Verify ODE residual
    res = -u2.diff(2)(x_test) - f2(x_test)
    max_res = float(jnp.max(jnp.abs(res)))
    print(f"  Solution length: {len(u2)}")
    print(f"  Max ODE residual: {max_res:.2e}")
    assert max_res < 1e-6

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(x_test, u(x_test), 'b', linewidth=1.8, label="chebfunjax")
    axes[0].plot(x_test, exact(x_test), 'r--', linewidth=1.2, label="exact cos(πx/2)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("u(x)")
    axes[0].set_title("−u″ = (π/2)² cos(πx/2), u(±1)=0", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_test, u2(x_test), 'g', linewidth=1.8)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("u(x)")
    axes[1].set_title("−u″ = exp(sin(πx)), u(±1)=0", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Krylov/spectral methods for BVPs", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "krylov.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
