"""Differentiation of Chebfuns.

Demonstrates first and second derivatives, the chain rule, and
the Fundamental Theorem of Calculus.

Credit: Inspired by Chebfun examples calc/Differentiation.m.
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
    print("Differentiation of Chebfuns")
    print("=" * 60)

    # --- First derivative: d/dx sin(x) = cos(x) ----------------------
    dom = (0.0, float(2.0 * jnp.pi))
    f = cj.chebfun(lambda x: jnp.sin(x), domain=dom)
    df = f.diff()
    print(f"\nf(x) = sin(x) on [0, 2*pi]:")
    x_test = jnp.linspace(0.0, float(2.0 * jnp.pi), 200)
    err1 = float(jnp.max(jnp.abs(df(x_test) - jnp.cos(x_test))))
    print(f"  ||diff(sin(x)) - cos(x)||_inf = {err1:.2e}")
    assert err1 < 1e-12

    # --- Second derivative: d^2/dx^2 exp(x) = exp(x) -----------------
    g = cj.chebfun(lambda x: jnp.exp(x))
    d2g = g.diff(2)
    x_test2 = jnp.linspace(-1.0, 1.0, 200)
    err2 = float(jnp.max(jnp.abs(d2g(x_test2) - jnp.exp(x_test2))))
    print(f"\ng(x) = exp(x) on [-1,1]:")
    print(f"  ||diff(exp(x),2) - exp(x)||_inf = {err2:.2e}")
    assert err2 < 1e-12

    # --- Chain rule: d/dx sin(x^2) = 2x*cos(x^2) --------------------
    h = cj.chebfun(lambda x: jnp.sin(x**2), domain=(0.0, 2.0))
    dh = h.diff()
    x_test3 = jnp.linspace(0.0, 2.0, 200)
    err3 = float(jnp.max(jnp.abs(dh(x_test3) - 2.0 * x_test3 * jnp.cos(x_test3**2))))
    print(f"\nh(x) = sin(x^2) on [0, 2]:")
    print(f"  ||diff(sin(x^2)) - 2x*cos(x^2)||_inf = {err3:.2e}")
    assert err3 < 1e-11

    # --- FTC: integral of f' over [a,b] = f(b) - f(a) ---------------
    f4 = cj.chebfun(lambda x: jnp.cos(x), domain=(0.0, float(jnp.pi)))
    df4 = f4.diff()
    # Integral of d/dx cos(x) = -sin(x) from 0 to pi
    # = cos(pi) - cos(0) = -1 - 1 = -2
    integral_df4 = float(df4.sum())
    exact_integral = float(jnp.cos(jnp.pi)) - float(jnp.cos(jnp.array(0.0)))
    print(f"\nf(x) = cos(x) on [0, pi]:")
    print(f"  Integral of f'(x) = f(pi) - f(0) = {exact_integral:.15f}")
    print(f"  Computed: {integral_df4:.15f}")
    assert abs(integral_df4 - exact_integral) < 1e-12

    # --- Higher derivatives ------------------------------------------
    f_cubic = cj.chebfun(lambda x: x**3)
    d3 = f_cubic.diff(3)  # should be 6 everywhere
    x_test4 = jnp.linspace(-1.0, 1.0, 100)
    err_d3 = float(jnp.max(jnp.abs(d3(x_test4) - 6.0)))
    print(f"\nd^3/dx^3 (x^3) error vs 6: {err_d3:.2e}")
    assert err_d3 < 1e-13

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f, title="sin(x) and its derivative", label="sin(x)")
    plot(df, ax=ax, color="#E04040", label="cos(x)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "differentiation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
