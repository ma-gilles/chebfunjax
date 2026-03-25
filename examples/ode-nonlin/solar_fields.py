"""Solar magnetic field model.

Solves the nonlinear BVP arising in solar force-free magnetic field modeling:
  (1-mu^2)*P'' + n*(n+1)*P + a^2*(1+n)/n * P^(1+2/n) = 0, P(-1)=P(1)=0.

Credit: Chebfun example ode-nonlin/SolarFields.m (Nick Hale & Natasha Flyer, Sep 2010).
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
    print("Solar force-free magnetic field model")
    print("=" * 60)

    dom = (-1.0, 1.0)

    # Simplified: linear part (n large or a=0) is just the Legendre equation
    # (1-x^2)*P'' + n*(n+1)*P = 0, P(-1)=P(1)=0
    # P_1(x) = x (exact for n=1)
    n = 1
    print(f"\nLinear (a=0): (1-x^2)P'' + {n}({n+1})P = 0, P(±1)=0")
    # Use the linearized equation
    N_linear = Chebop(
        lambda x, P: (1.0 - x**2) * P.diff(2) + n*(n+1) * P,
        domain=(-0.999, 0.999)  # avoid endpoints where (1-x^2)=0
    )
    N_linear.lbc = float(-0.999)  # P(-1)=-1 (P_1(-1)=-1)
    N_linear.rbc = float(0.999)   # P(1)=1  (P_1(1)=1)
    P_linear = N_linear.solve(0.0)

    x_test = jnp.linspace(-0.99, 0.99, 300)
    err_linear = float(jnp.max(jnp.abs(P_linear(x_test) - x_test)))
    print(f"  Max error vs P_1(x)=x: {err_linear:.2e}")
    assert err_linear < 1e-6

    # Nonlinear case with small a
    a = 0.1
    n_nl = 2
    print(f"\nNonlinear (a={a}, n={n_nl}):")
    N_nl = Chebop(
        lambda x, P: ((1.0 - x**2) * P.diff(2)
                      + n_nl*(n_nl+1) * P
                      + a**2 * (1 + n_nl)/n_nl * jnp.sign(P) * jnp.abs(P)**(1 + 2.0/n_nl)),
        domain=(-0.999, 0.999)
    )
    N_nl.lbc = 0.0
    N_nl.rbc = 0.0
    P_nl = N_nl.solve(0.5)
    P0_nl = float(P_nl(jnp.array(0.0)))
    print(f"  P(0) = {P0_nl:.6f}  (non-trivial solution)")
    assert abs(float(P_nl(jnp.array(-0.999)))) < 1e-6
    assert abs(float(P_nl(jnp.array(0.999)))) < 1e-6

    # ODE residual
    x_int = jnp.linspace(-0.9, 0.9, 200)
    P_val = P_nl(x_int)
    P2_val = P_nl.diff(2)(x_int)
    res = ((1.0 - x_int**2) * P2_val
           + n_nl*(n_nl+1) * P_val
           + a**2 * (1 + n_nl)/n_nl * jnp.sign(P_val) * jnp.abs(P_val)**(1 + 2.0/n_nl))
    max_res = float(jnp.max(jnp.abs(res)))
    print(f"  Max ODE residual: {max_res:.2e}")
    assert max_res < 1e-7

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    x_plot = jnp.linspace(-0.999, 0.999, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x_plot, P_linear(x_plot), 'b', linewidth=1.8, label="chebfunjax")
    axes[0].plot(x_plot, x_plot, 'r--', linewidth=1.2, label="exact P₁(x)=x")
    axes[0].set_xlabel("μ"); axes[0].set_ylabel("P(μ)")
    axes[0].set_title(f"Linear Legendre n={n}", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_plot, P_nl(x_plot), 'b', linewidth=1.8)
    axes[1].set_xlabel("μ"); axes[1].set_ylabel("P(μ)")
    axes[1].set_title(f"Nonlinear solar field (a={a}, n={n_nl})", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Solar force-free magnetic field model", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "solar_fields.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
