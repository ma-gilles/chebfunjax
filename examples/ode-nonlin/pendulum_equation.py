"""Nonlinear pendulum BVP.

Solves theta'' + (g/L)*sin(theta) = 0 for the nonlinear pendulum
with given boundary conditions.

Credit: Inspired by Chebfun ode-nonlin examples.
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
    print("Nonlinear pendulum: theta'' + (g/L)*sin(theta) = 0")
    print("=" * 60)

    g_over_L = 1.0  # g/L = 1 for simplicity

    # For small angles: theta'' + theta = 0 (simple harmonic motion)
    # Solution: theta(t) = theta0*cos(t) for theta'(0)=0, theta(0)=theta0
    # For BVP: theta(0) = theta0, theta(T_linear/4) = 0 (quarter period)
    # T_linear = 2*pi, quarter period = pi/2

    theta0 = 0.3  # small angle amplitude in radians
    T_quarter = float(jnp.pi) / 2.0
    dom = (0.0, T_quarter)

    # Nonlinear pendulum BVP
    # Note: inside Chebop, `theta` is a Chebfun so use theta.sin() not jnp.sin(theta)
    N = Chebop(
        lambda t, theta: theta.diff(2) + g_over_L * theta.sin(),
        domain=dom
    )
    N.lbc = theta0   # theta(0) = theta0
    N.rbc = 0.0      # theta(T/4) = 0

    theta = N.solve(0.0)
    print(f"\nSolved on [0, pi/2] with theta(0)={theta0}, theta(pi/2)=0:")
    print(f"  Chebfun length: {len(theta)}")

    # Verify ODE residual
    t_test = jnp.linspace(0.05, T_quarter - 0.05, 200)
    d2theta = theta.diff(2)
    residual = d2theta(t_test) + g_over_L * jnp.sin(theta(t_test))
    max_res = float(jnp.max(jnp.abs(residual)))
    print(f"  Max ODE residual: {max_res:.2e}")
    assert max_res < 1e-9, f"Residual too large: {max_res}"

    # Verify boundary conditions
    assert abs(float(theta(jnp.array(0.0))) - theta0) < 1e-11
    assert abs(float(theta(jnp.array(T_quarter)))) < 1e-11

    # Compare with small-angle approximation: theta(t) ~ theta0*cos(t)
    t_all = jnp.linspace(0.0, T_quarter, 200)
    theta_small_angle = theta0 * jnp.cos(t_all)
    err_small = float(jnp.max(jnp.abs(theta(t_all) - theta_small_angle)))
    print(f"\nDifference from small-angle (theta0={theta0} rad):")
    print(f"  Max |theta - theta0*cos(t)| = {err_small:.2e}")
    # For theta0=0.3 radians, nonlinear correction is ~O(theta0^3) = O(0.027)
    # Expect difference ~ 0.027, so err_small < 0.05
    assert err_small < 0.05, f"Too different from linear: {err_small}"

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(theta, title="Pendulum equation: θ″ + sin θ = 0",
                   ylabel="θ (rad)")
    fig.savefig(os.path.join(_here, "pendulum_equation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
