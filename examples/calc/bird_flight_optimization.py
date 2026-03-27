"""Bird flight optimization via calculus.

A bird can fly faster over land than over water. Given a coastal
landscape, find the optimal crossing point that minimizes total
flight time (energy).

Credit: Inspired by Chebfun examples calc/BirdFlight.m (Nick Trefethen).
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
    print("Optimizing a bird's flight path")
    print("=" * 60)

    # Source at origin, coast at x = 5 km, destination at (13, 0)
    # Water speed = 1, land speed = 1.4 (40% faster on land)
    # Cross coast at point (5, y): water distance = sqrt(25 + y^2),
    #   land distance = sqrt(64 + y^2) (from (5,y) to (13,0))
    # Total time T(y) = sqrt(25+y^2)/1 + sqrt(64+y^2)/1.4
    dom = (0.0, 13.0)
    x = cj.chebfun(lambda t: t, domain=dom)

    # Standard bird-flight: cross a river of width w=5, land dist=8
    # Energy = water_length * 1 + land_length * 0.714 (land faster)
    water_speed = 1.0
    land_speed = 1.4
    w = 5.0  # river/water width
    d = 13.0  # total horizontal distance

    # T(x) = sqrt(x^2 + w^2)/water_speed + sqrt((d-x)^2 + w^2)/land_speed
    T = cj.chebfun(
        lambda x_: jnp.sqrt(x_**2 + w**2) / water_speed +
                   jnp.sqrt((d - x_)**2 + w**2) / land_speed,
        domain=dom
    )

    x_min, T_min = T.min()
    print(f"\nOptimal crossing point x* = {x_min:.10f}")
    print(f"Minimum travel time T* = {T_min:.10f}")

    # Exact solution from calculus (Snell's law):
    # sin(theta_water)/v_water = sin(theta_land)/v_land
    # x / sqrt(x^2 + w^2) / v_water = (d-x) / sqrt((d-x)^2 + w^2) / v_land
    # For small water: x/sqrt(x^2+25) = 1/1.4 * (13-x)/sqrt((13-x)^2+25)
    # Solve numerically for reference:
    from scipy.optimize import brentq
    def snell(x_):
        lhs = x_ / np.sqrt(x_**2 + w**2) / water_speed
        rhs = (d - x_) / np.sqrt((d - x_)**2 + w**2) / land_speed
        return lhs - rhs

    x_exact = brentq(snell, 0.01, d - 0.01)
    print(f"Exact (Snell's law) x* = {x_exact:.10f}")
    print(f"Error: {abs(x_min - x_exact):.2e}")
    assert abs(x_min - x_exact) < 1e-9, f"x_min error: {abs(x_min - x_exact)}"

    # Verify T'(x*) = 0 (first-order condition)
    dT = T.diff()
    dT_at_min = float(dT(jnp.array(x_min)))
    print(f"T'(x*) = {dT_at_min:.2e}  (should be ~0)")
    assert abs(dT_at_min) < 1e-8

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(T, title="Bird-flight travel time T(x)", ylabel="T")
    ax.axvline(float(x_min), color="#E04040", linewidth=1.2,
               linestyle="--", label=f"x* = {float(x_min):.4f}")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "bird_flight_optimization.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
