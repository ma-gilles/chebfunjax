"""Parameter-dependent ODE with breakpoints.

Solves  (a(x,s) u')' = 1,  u(0)=u(1)=0  where  a(x,s)=1+4s(x^2-x)
for several values of s, comparing with exact  u(x,s) = log(a)/(8s).

Credit: Chebfun example ode-linear/ParameterODE.m (Asgeir Birkisson, Jan 2012).
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
    print("Parameter ODE: (a(x,s) u')' = 1, u(0)=u(1)=0")
    print("=" * 60)

    dom = (0.0, 1.0)
    s_vals = [0.1, 0.5, 0.9]

    def a(x, s):
        return 1.0 + 4.0 * s * (x**2 - x)

    def exact_u(x, s):
        """Exact solution: u = log(a(x,s)) / (8s)"""
        return jnp.log(a(x, s)) / (8.0 * s)

    solutions = []
    for s in s_vals:
        # Expand (a u')' = a' u' + a u'' = 1
        # a'(x,s) = 4s(2x - 1)
        da = lambda x: 4.0 * s * (2.0 * x - 1.0)
        a_fn = lambda x: 1.0 + 4.0 * s * (x**2 - x)

        N = Chebop(
            lambda x, u: a_fn(x) * u.diff(2) + da(x) * u.diff(),
            domain=dom
        )
        N.lbc = 0.0
        N.rbc = 0.0
        u = N.solve(cj.chebfun(lambda x: jnp.ones_like(x), domain=dom))

        x_test = jnp.linspace(0.05, 0.95, 200)
        err = float(jnp.max(jnp.abs(u(x_test) - exact_u(x_test, s))))
        print(f"\ns={s}: max error = {err:.2e}, length = {len(u)}")
        assert err < 1e-8, f"s={s}: error too large {err}"
        solutions.append((s, u))

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(0.0, 1.0, 400)
    colors = ['b', 'r', 'g']

    fig, axes = plt.subplots(1, 2)
    for (s, u), c in zip(solutions, colors):
        axes[0].plot(x_plot, u(x_plot), color=c, linewidth=1.6, label=f"s={s}")
        axes[0].plot(x_plot, exact_u(x_plot, s), '--', color=c, linewidth=1.0, alpha=0.5)

    axes[0].set_xlabel("x"); axes[0].set_ylabel("u(x)")
    axes[0].set_title("(a(x,s) u′)′ = 1, u(0)=u(1)=0", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Show a(x,s) profiles
    x_plot_np = np.linspace(0.0, 1.0, 400)
    for s, c in zip(s_vals, colors):
        a_vals = 1.0 + 4.0 * s * (x_plot_np**2 - x_plot_np)
        axes[1].plot(x_plot_np, a_vals, color=c, linewidth=1.6, label=f"s={s}")
    axes[1].axhline(0, color='k', linewidth=0.5, linestyle='--')
    axes[1].set_xlabel("x"); axes[1].set_ylabel("a(x, s)")
    axes[1].set_title("Coefficient a(x,s) = 1 + 4s(x²−x)", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Parameter-dependent BVP", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "parameter_ode.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
