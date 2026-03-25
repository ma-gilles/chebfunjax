"""Adjoint of a linear operator.

Illustrates the concept of operator adjoints for simple first- and
second-order operators with Dirichlet boundary conditions.

Credit: Chebfun example ode-linear/Adjoints.m (Yuji Nakatsukasa, Feb 2017).
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
from chebfunjax.plotting import plot
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Adjoint of a linear operator")
    print("=" * 60)

    dom = (-1.0, 1.0)

    # L = d/dx with Dirichlet BC at left: u(-1) = 0
    # Adjoint: L* = -d/dx with Dirichlet BC at right: v(1) = 0
    # Verify (v, Lu) = (L*v, u) numerically
    print("\nVerifying adjoint identity (v, Lu) = (L*v, u)...")

    # Choose test functions
    u_fn = cj.chebfun(lambda x: jnp.sin(jnp.pi * x), domain=dom)
    v_fn = cj.chebfun(lambda x: jnp.exp(x) * (1 - x**2), domain=dom)

    Lu = u_fn.diff()
    inner1 = (v_fn * Lu).sum()   # (v, Lu)

    Lstv = -v_fn.diff()
    inner2 = (Lstv * u_fn).sum()  # (L*v, u)

    err = abs(float(inner1) - float(inner2))
    print(f"  (v, Lu)   = {float(inner1):.10f}")
    print(f"  (L*v, u)  = {float(inner2):.10f}")
    print(f"  Difference: {err:.2e}")
    assert err < 1e-10, f"Adjoint identity failed: diff={err}"

    # Second-order example: L = d^2/dx^2 with u(-1)=u(1)=0
    # L is self-adjoint in this case
    print("\nSelf-adjoint check for d^2/dx^2 with Dirichlet BCs...")
    u2 = cj.chebfun(lambda x: jnp.sin(jnp.pi * x), domain=dom)
    v2 = cj.chebfun(lambda x: jnp.cos(jnp.pi * x / 2) * (1 - x**2), domain=dom)

    Lu2 = u2.diff(2)
    Lv2 = v2.diff(2)
    inner_a = (v2 * Lu2).sum()
    inner_b = (u2 * Lv2).sum()
    err2 = abs(float(inner_a) - float(inner_b))
    print(f"  (v, L u)  = {float(inner_a):.10f}")
    print(f"  (u, L v)  = {float(inner_b):.10f}")
    print(f"  Difference: {err2:.2e}")
    assert err2 < 1e-10

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    x_plot = jnp.linspace(-1.0, 1.0, 300)
    axes[0].plot(x_plot, u_fn(x_plot), label="u(x) = sin(πx)")
    axes[0].plot(x_plot, v_fn(x_plot), label="v(x) = eˣ(1−x²)")
    axes[0].plot(x_plot, Lu(x_plot), '--', label="Lu = u'(x)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Functions and L u", fontsize=10)
    axes[0].set_xlabel("x")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_plot, Lstv(x_plot), label="L*v = −v'(x)")
    axes[1].plot(x_plot, u_fn(x_plot), '--', label="u(x)")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Adjoint: L*v = −v'", fontsize=10)
    axes[1].set_xlabel("x")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Operator adjoint: (v, Lu) = (L*v, u)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "adjoints.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
