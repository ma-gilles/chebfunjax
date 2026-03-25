"""Linear exp initial-value problem.

Solves  u' - lambda u = 0,  u(0) = 1,  lambda = -10000
on [0, 0.005].  The solution is exp(lambda x) = exp(-10000 x).

Credit: Chebfun example ode-linear/LinExpIVP.m (Tom Maerz, Oct 2010).
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
    print("Linear exp IVP: u' - lambda u = 0, u(0)=1")
    print("=" * 60)

    lam = -10000.0
    d = (0.0, 0.005)

    exact = lambda x: jnp.exp(lam * x)

    # Use BVP with exact right endpoint value
    rbc_val = float(exact(jnp.array(0.005)))
    N = Chebop(lambda x, u: u.diff() - lam * u, domain=d)
    N.lbc = 1.0
    N.rbc = rbc_val

    print(f"\nSolving on [0, 0.005] with lambda = {lam}")
    u = N.solve(0.0)
    print(f"  Solution length: {len(u)}")

    x_test = jnp.linspace(0.0, 0.005, 200)
    err = float(jnp.max(jnp.abs(u(x_test) - exact(x_test))))
    print(f"  Max error vs exp(lambda x): {err:.2e}")
    assert err < 1e-8, f"Error too large: {err}"

    u0 = float(u(jnp.array(0.0)))
    print(f"  u(0) = {u0:.15f}  (exact: 1.0)")
    assert abs(u0 - 1.0) < 1e-12

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(0.0, 0.005, 400)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x_plot, u(x_plot), 'b', linewidth=1.8, label="chebfunjax")
    axes[0].plot(x_plot, exact(x_plot), 'r--', linewidth=1.2, label="exp(λx)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("u(x)")
    axes[0].set_title(f"u′ − λu = 0,  λ={lam}", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(x_plot, jnp.abs(u(x_plot) - exact(x_plot)) + 1e-20, 'g', linewidth=1.6)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("|error|")
    axes[1].set_title("Pointwise error", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Linear exp IVP: solution = exp({lam}x)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "lin_exp_ivp.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
