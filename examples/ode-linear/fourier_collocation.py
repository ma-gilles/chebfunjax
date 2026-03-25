"""Fourier spectral collocation for periodic ODEs.

Solves the periodic first-order ODE
  u' + a(x) u = f(x),  x in [0, 2pi], periodic BCs
where a(x) = 1 + sin(cos(10x)) and f(x) = exp(sin(x)).

Credit: Chebfun example ode-linear/FourierCollocation.m (Hadrien Montanelli, Dec 2014).
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


def run():
    print("=" * 60)
    print("Fourier collocation for periodic ODE: u' + a(x)u = f(x)")
    print("=" * 60)

    dom = (0.0, 2.0 * float(jnp.pi))

    # a(x) = 1 + sin(cos(10x)),  f(x) = exp(sin(x))
    a_fn = cj.chebfun(lambda x: 1.0 + jnp.sin(jnp.cos(10.0 * x)), domain=dom)
    f_fn = cj.chebfun(lambda x: jnp.exp(jnp.sin(x)), domain=dom)

    # Solve with Chebop (periodic BCs)
    N = Chebop(lambda x, u: u.diff() + (1.0 + jnp.sin(jnp.cos(10.0 * x))) * u,
               domain=dom)
    N.bc = "periodic"
    u = N.solve(f_fn)
    print(f"\nSolution length: {len(u)}")

    # Verify periodicity
    u_left = float(u(jnp.array(dom[0])))
    u_right = float(u(jnp.array(dom[1])))
    print(f"  u(0) = {u_left:.8f}")
    print(f"  u(2π)= {u_right:.8f}")
    err_periodic = abs(u_left - u_right)
    print(f"  Periodicity error: {err_periodic:.2e}")
    assert err_periodic < 1e-8

    # Verify ODE residual at interior points
    x_test = jnp.linspace(0.1, float(2 * jnp.pi) - 0.1, 200)
    res = u.diff()(x_test) + (1.0 + jnp.sin(jnp.cos(10.0 * x_test))) * u(x_test) - f_fn(x_test)
    max_res = float(jnp.max(jnp.abs(res)))
    print(f"  Max ODE residual: {max_res:.2e}")
    assert max_res < 1e-8

    # Also solve the constant-coefficient case u' + u = sin(x) with exact solution
    print("\nConstant-coefficient case: u' + u = sin(x), periodic on [0, 2pi]")
    # Exact: u = (sin(x) - cos(x)) / 2
    N2 = Chebop(lambda x, u: u.diff() + u, domain=dom)
    N2.bc = "periodic"
    rhs2 = cj.chebfun(lambda x: jnp.sin(x), domain=dom)
    u2 = N2.solve(rhs2)
    exact2 = cj.chebfun(lambda x: (jnp.sin(x) - jnp.cos(x)) / 2.0, domain=dom)
    x_test2 = jnp.linspace(0.0, float(2 * jnp.pi), 200)
    err2 = float(jnp.max(jnp.abs(u2(x_test2) - exact2(x_test2))))
    print(f"  Max error vs exact: {err2:.2e}")
    assert err2 < 1e-10

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(0.0, float(2 * jnp.pi), 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axes[0].plot(x_plot, u(x_plot), 'b', linewidth=1.6)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("u(x)")
    axes[0].set_title("u′ + (1+sin(cos(10x)))u = exp(sin(x))", fontsize=9)
    axes[0].set_xticks([0, np.pi, 2*np.pi])
    axes[0].set_xticklabels(["0", "π", "2π"])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_plot, u2(x_plot), 'r', linewidth=1.6, label="chebfunjax")
    axes[1].plot(x_plot, exact2(x_plot), 'k--', linewidth=1.2, label="exact")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("u(x)")
    axes[1].set_title("u′ + u = sin(x),  exact = (sin−cos)/2", fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].set_xticks([0, np.pi, 2*np.pi])
    axes[1].set_xticklabels(["0", "π", "2π"])
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Fourier collocation for periodic ODEs", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "fourier_collocation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
