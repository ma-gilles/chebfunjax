"""Gulf Stream: third-order BVP on the half-line.

Solves the Ierley-Ruehr nonlinear ODE:
  u''' - lambda*((u')^2 - u*u'') - u + 1 = 0, u(0) = 0, x in [0, inf)
on a truncated domain [0, L] with far-field condition u(L) = 1.

Credit: Chebfun example ode-nonlin/GulfStream.m (C.I. Gheorghiu, Jan 2020).
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
    print("Gulf Stream: u''' - lambda((u')^2 - u u'') - u + 1 = 0")
    print("=" * 60)

    lam = 0.01
    L = 10.0
    dom = (0.0, L)

    # u(0) = 0, u'(0) = ? — use Neumann-type via lbc=lambda u:[u, u']
    # Far-field: u(L) ≈ 1 (steady state)
    # Additional BC: u'(L) = 0 (far field derivative)
    N = Chebop(
        lambda x, u: (u.diff(3)
                      - lam * (u.diff()**2 - u * u.diff(2))
                      - u + 1.0),
        domain=dom
    )
    N.lbc = [0.0, 0.0]   # u(0)=0, u'(0)=0
    N.rbc = 1.0          # u(L)=1

    print(f"\nSolving on [0, {L}] with lambda={lam}...")
    u = N.solve(cj.chebfun(lambda x: x / L, domain=dom))
    print(f"  Solution length: {len(u)}")
    print(f"  u(0) = {float(u(jnp.array(0.0))):.8f}  (should be 0)")
    print(f"  u(L) = {float(u(jnp.array(L))):.8f}  (should be 1)")

    assert abs(float(u(jnp.array(0.0)))) < 1e-7
    assert abs(float(u(jnp.array(L))) - 1.0) < 1e-7

    # ODE residual
    x_test = jnp.linspace(0.5, L - 0.5, 200)
    res = (u.diff(3)(x_test) - lam * (u.diff()(x_test)**2 - u(x_test) * u.diff(2)(x_test))
           - u(x_test) + 1.0)
    max_res = float(jnp.max(jnp.abs(res)))
    print(f"  Max ODE residual: {max_res:.2e}")
    assert max_res < 1e-7

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(0.0, L, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x_plot, u(x_plot), 'b', linewidth=1.8, label="u(x)")
    axes[0].plot(x_plot, u.diff()(x_plot), 'r', linewidth=1.4, label="u'(x)")
    axes[0].set_xlabel("x"); axes[0].legend(fontsize=9)
    axes[0].set_title(f"Gulf Stream BVP (λ={lam})", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(x_plot[1:], jnp.abs(res[:len(x_plot)-1]) + 1e-20, 'g', linewidth=1.6)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("|residual|")
    axes[1].set_title("ODE residual (log scale)", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Gulf Stream ODE on half-line", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "gulf_stream.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
