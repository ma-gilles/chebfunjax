"""Definite and indefinite integrals (sum/cumsum).

Demonstrates computing definite integrals with sum() and
indefinite integrals (antiderivatives) with cumsum().

Credit: Inspired by Chebfun examples calc/Intro.m.
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

from chebfunjax.plotting import plot

def run():
    print("=" * 60)
    print("Definite and indefinite integrals")
    print("=" * 60)

    # MATLAB: f = chebfun('2*cos(x)', [0 10])
    dom = (0.0, 10.0)
    f = cj.chebfun(lambda x: 2.0 * jnp.cos(x), domain=dom)

    # --- Definite integral: sum(f) -----------------------------------
    integral = float(f.sum())
    exact = 2.0 * float(jnp.sin(jnp.array(10.0)))
    print(f"\nf(x) = 2*cos(x) on [0, 10]")
    print(f"  sum(f) = {integral:.15f}")
    print(f"  exact  = 2*sin(10) = {exact:.15f}")
    print(f"  error  = {abs(integral - exact):.2e}")
    assert abs(integral - exact) < 1e-12

    # Integral over subinterval [3, 4] = 2*sin(4) - 2*sin(3)
    f_restricted = f.restrict(3.0, 4.0)
    integral_sub = float(f_restricted.sum())
    exact_sub = 2.0 * (float(jnp.sin(jnp.array(4.0))) -
                       float(jnp.sin(jnp.array(3.0))))
    print(f"\n  Integral over [3, 4] = {integral_sub:.15f}")
    print(f"  Exact               = {exact_sub:.15f}")
    print(f"  Error               = {abs(integral_sub - exact_sub):.2e}")
    assert abs(integral_sub - exact_sub) < 1e-12

    # --- Indefinite integral: cumsum(f) ------------------------------
    g = f.cumsum()  # antiderivative with g(0) = 0
    # g(x) = 2*sin(x) - 2*sin(0) = 2*sin(x)
    x_test = jnp.array([1.0, 3.0, 5.0, 9.0])
    g_vals = g(x_test)
    exact_g = 2.0 * jnp.sin(x_test)
    err_g = float(jnp.max(jnp.abs(g_vals - exact_g)))
    print(f"\n  cumsum(f) evaluated at test points:")
    print(f"  Max error vs 2*sin(x): {err_g:.2e}")
    assert err_g < 1e-12

    # g(4) - g(3) should equal integral over [3,4]
    g4_minus_g3 = float(g(jnp.array(4.0))) - float(g(jnp.array(3.0)))
    print(f"\n  g(4) - g(3) = {g4_minus_g3:.15f}")
    print(f"  Integral [3,4] = {integral_sub:.15f}")
    assert abs(g4_minus_g3 - integral_sub) < 1e-12

    # --- diff(cumsum(f)) == f  (to machine precision) ----------------
    norm_diff_cumsum = float((f.cumsum().diff() - f).norm())
    print(f"\n  ||diff(cumsum(f)) - f|| = {norm_diff_cumsum:.2e}  (should be ~0)")
    assert norm_diff_cumsum < 1e-12

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f, title="2·cos(x) and its antiderivative on [0, 10]",
                   label="2·cos(x)")
    plot(g, ax=ax, color="#E04040", label="cumsum (antiderivative)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "definite_indefinite_integrals.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
