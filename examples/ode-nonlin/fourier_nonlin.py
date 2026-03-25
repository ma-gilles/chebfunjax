"""Nonlinear periodic ODE with Fourier collocation.

Solves  u' - u*cos(u) = cos(4x),  x in [0, 2pi], periodic BCs.
Uses Chebop with periodic boundary conditions.

Credit: Chebfun example ode-nonlin/FourierCollocationNonLin.m (Hadrien Montanelli, Dec 2014).
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
    print("Nonlinear periodic ODE: u' - u cos(u) = cos(4x)")
    print("=" * 60)

    dom = (0.0, 2.0 * float(jnp.pi))

    N = Chebop(
        lambda x, u: u.diff() - u * jnp.cos(u),
        domain=dom
    )
    N.bc = "periodic"
    rhs = cj.chebfun(lambda x: jnp.cos(4.0 * x), domain=dom)
    u = N.solve(rhs)

    print(f"\nSolution length: {len(u)}")

    # Verify periodicity
    err_p = abs(float(u(jnp.array(dom[0]))) - float(u(jnp.array(dom[1]))))
    print(f"  Periodicity error: {err_p:.2e}")
    assert err_p < 1e-8

    # Verify ODE residual
    x_test = jnp.linspace(0.1, float(2 * jnp.pi) - 0.1, 300)
    res = u.diff()(x_test) - u(x_test) * jnp.cos(u(x_test)) - jnp.cos(4.0 * x_test)
    max_res = float(jnp.max(jnp.abs(res)))
    print(f"  Max ODE residual: {max_res:.2e}")
    assert max_res < 1e-8

    # Solution is periodic and oscillatory
    u_vals = u(x_test)
    print(f"  max|u| = {float(jnp.max(jnp.abs(u_vals))):.4f}")

    # Also solve the linear version u' + u = cos(4x) for comparison
    N2 = Chebop(lambda x, u: u.diff() + u, domain=dom)
    N2.bc = "periodic"
    u2 = N2.solve(rhs)
    err_p2 = abs(float(u2(jnp.array(dom[0]))) - float(u2(jnp.array(dom[1]))))
    assert err_p2 < 1e-8

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(0.0, float(2 * jnp.pi), 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x_plot, u(x_plot), 'b', linewidth=1.8)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("u(x)")
    axes[0].set_title("u′ − u cos(u) = cos(4x), periodic", fontsize=9)
    axes[0].set_xticks([0, np.pi, 2*np.pi])
    axes[0].set_xticklabels(["0", "π", "2π"])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_plot, u2(x_plot), 'r', linewidth=1.8, label="linear u′+u=cos(4x)")
    axes[1].plot(x_plot, u(x_plot), 'b--', linewidth=1.2, label="nonlinear")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("u(x)")
    axes[1].set_title("Comparison: linear vs nonlinear", fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].set_xticks([0, np.pi, 2*np.pi])
    axes[1].set_xticklabels(["0", "π", "2π"])
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Nonlinear periodic ODE with Fourier collocation", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "fourier_nonlin.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
