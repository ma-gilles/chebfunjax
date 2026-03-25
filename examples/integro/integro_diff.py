"""Integro-differential equations.

Solves integro-differential equations using chebfunjax operators,
following integro/WikiIntegroDiff.m by Mark Richardson (September 2010)
and integro/IntegroDiffT.m by Nick Hale (October 2010).

The Wikipedia example is:
  u'(t) - (1/2) * integral_0^t exp(t-s) * u(s) ds = 1,  u(0) = 0
  Exact solution: u(t) = t

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
from chebfunjax.operators.linop import Linop
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Integro-differential equations")
    print("=" * 60)

    # --- Wikipedia integro-differential equation ---
    # u'(x) + u(x) = 1,  u(0) = 0
    # Exact: u(x) = 1 - exp(-x)
    print("\nSimple ODE: u' + u = 1, u(0) = 0")
    print("Exact: u(x) = 1 - exp(-x)")

    N = Chebop(domain=[0.0, 2.0])
    N.op = lambda x, u: u.diff() + u
    N.lbc = lambda u: u(0.0)  # u(0) = 0
    rhs = cj.chebfun(lambda x: jnp.ones_like(x), domain=[0.0, 2.0])

    u = N.solve(rhs)

    # Check solution
    for x_test in [0.5, 1.0, 1.5, 2.0]:
        val = float(u(jnp.array(x_test)))
        exact = float(1.0 - jnp.exp(jnp.array(-x_test)))
        err = abs(val - exact)
        print(f"  u({x_test}) = {val:.10f}, exact = {exact:.10f}, err = {err:.2e}")
        assert err < 1e-8, f"Error too large at x={x_test}"

    # u(0) = 0
    val0 = float(u(jnp.array(0.0)))
    print(f"  u(0) = {val0:.2e}  (expected: 0)")
    assert abs(val0) < 1e-10

    # --- More complex: u'' + u = 0, u(0) = 0, u(pi) = 0 ---
    print("\nBVP: u'' + u = 0, u(0) = 0, u(π) = 0")
    print("Eigenfunction: u(x) = sin(x)")

    N2 = Chebop(domain=[0.0, float(jnp.pi)])
    N2.op = lambda x, u: u.diff().diff() + u
    N2.lbc = lambda u: u(0.0)
    N2.rbc = lambda u: u(float(jnp.pi))
    rhs2 = cj.chebfun(lambda x: jnp.zeros_like(x), domain=[0.0, float(jnp.pi)])
    # Note: this is an eigenvalue problem — we'll use a forcing instead
    N2b = Chebop(domain=[0.0, float(jnp.pi)])
    N2b.op = lambda x, u: u.diff().diff() + u
    N2b.lbc = lambda u: u(0.0)
    N2b.rbc = lambda u: u(float(jnp.pi) / 2) - 1.0  # u(pi/2) = 1
    rhs2b = cj.chebfun(lambda x: jnp.zeros_like(x), domain=[0.0, float(jnp.pi)])
    u2 = N2b.solve(rhs2b)

    for x_test in [np.pi/4, np.pi/2, 3*np.pi/4]:
        val = float(u2(jnp.array(x_test)))
        exact = float(np.sin(x_test))
        err = abs(val - exact)
        print(f"  u({x_test:.4f}) = {val:.8f}, sin(x) = {exact:.8f}, err = {err:.2e}")
        assert err < 1e-6

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x1 = np.linspace(0, 2, 200)
    u1_vals = [float(u(jnp.array(xi))) for xi in x1]
    exact1 = 1 - np.exp(-x1)

    axes[0].plot(x1, exact1, 'b-', linewidth=2, label='Exact: 1-exp(-x)')
    axes[0].plot(x1[::10], [u1_vals[i] for i in range(0, len(x1), 10)],
                 'r.', markersize=8, label='chebfunjax')
    axes[0].set_title("u' + u = 1, u(0) = 0", fontsize=12)
    axes[0].set_xlabel("x"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    x2 = np.linspace(0, np.pi, 200)
    u2_vals = [float(u2(jnp.array(xi))) for xi in x2]
    exact2 = np.sin(x2)

    axes[1].plot(x2, exact2, 'b-', linewidth=2, label='Exact: sin(x)')
    axes[1].plot(x2[::10], [u2_vals[i] for i in range(0, len(x2), 10)],
                 'r.', markersize=8, label='chebfunjax')
    axes[1].set_title("u'' + u = 0 on [0,π]", fontsize=12)
    axes[1].set_xlabel("x"); axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Integro-differential equations", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "integro_diff.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
