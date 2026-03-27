"""Picard iteration for ODE existence proof.

Illustrates Picard-Lindelof successive approximations for the IVP
  u' = u,  u(0) = 1, whose solution is exp(x).

Credit: Chebfun example ode-nonlin/Picard.m (Nick Trefethen, Jan 2016).
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

def run():
    print("=" * 60)
    print("Picard iteration: u' = u, u(0) = 1  =>  u = exp(x)")
    print("=" * 60)

    dom = (0.0, 1.0)
    # Picard: u_{n+1}(x) = 1 + integral_0^x u_n(t) dt
    # Starting from u_0 = 1:
    # u_1 = 1 + x
    # u_2 = 1 + x + x^2/2
    # u_k = sum_{j=0}^{k} x^j/j!  (Taylor series of exp)

    u = cj.chebfun(lambda x: jnp.ones_like(x), domain=dom)  # u_0 = 1
    exact = cj.chebfun(lambda x: jnp.exp(x), domain=dom)

    print(f"\n{'iteration':>12}  {'max_error':>14}")
    print("-" * 28)
    errors = []
    for k in range(8):
        err = float(jnp.max(jnp.abs(u(jnp.linspace(0, 1, 300)) -
                                     exact(jnp.linspace(0, 1, 300)))))
        errors.append(err)
        print(f"  {k:12d}  {err:14.6e}")

        # Picard step: u_{k+1}(x) = 1 + integral_0^x u_k(t) dt
        # cumsum gives integral from left endpoint (a=0 here)
        antideriv = u.cumsum()
        u_new = cj.chebfun(lambda x, ad=antideriv: 1.0 + ad(x),
                           domain=dom)
        u = u_new

    # Verify convergence (Picard converges ~1 order of magnitude per iteration)
    assert errors[0] > errors[-1], "Picard should converge"
    assert errors[-1] < 1e-4, f"After 8 iterations: {errors[-1]:.2e}"
    print(f"\nConverged: {errors[-1]:.2e} after 7 iterations")

    # Also verify the final approximation
    x_test = jnp.linspace(0.0, 1.0, 200)
    final_err = float(jnp.max(jnp.abs(u(x_test) - jnp.exp(x_test))))
    print(f"Final max error vs exp(x): {final_err:.2e}")
    assert final_err < 1e-4

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    # Rebuild individual iterations for plotting
    u_iter = cj.chebfun(lambda x: jnp.ones_like(x), domain=dom)
    x_plot = jnp.linspace(0.0, 1.0, 300)
    colors = plt.cm.viridis(np.linspace(0, 0.9, 6))

    for k in range(6):
        axes[0].plot(x_plot, u_iter(x_plot), color=colors[k], linewidth=1.4,
                     label=f"u_{k}" if k < 4 else None, alpha=0.7)
        antideriv = u_iter.cumsum()
        u_iter = cj.chebfun(
            lambda x, ad=antideriv: 1.0 + ad(x),
            domain=dom
        )
    axes[0].plot(x_plot, jnp.exp(x_plot), 'k--', linewidth=1.4, label="exp(x)")
    axes[0].set_title("Picard iterations converging to exp(x)", fontsize=9)
    axes[0].legend(fontsize=7)

    axes[1].semilogy(range(len(errors)), errors, 'b.-', markersize=8, linewidth=1.4)
    axes[1].set_title("Convergence of Picard iteration", fontsize=10)

    fig.suptitle("Picard-Lindelöf iteration: u' = u, u(0)=1", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "picard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
