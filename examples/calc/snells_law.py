"""Snell's law: optimal light path between two media.

Light travels from point A in medium 1 (speed c1) to point B in
medium 2 (speed c2), crossing a flat interface. Snell's law says
the optimal crossing point satisfies sin(theta1)/c1 = sin(theta2)/c2.

Credit: Inspired by Chebfun examples calc/SnellsLaw.m.
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
    print("Snell's law via Chebfun optimization")
    print("=" * 60)

    # Source at (0, h1), target at (d, -h2), interface at y=0
    h1 = 1.0   # depth of source above interface
    h2 = 2.0   # depth of target below interface
    d = 3.0    # horizontal distance
    c1 = 1.0   # speed in medium 1 (above interface)
    c2 = 1.5   # speed in medium 2 (below interface); faster

    # Travel time as a function of crossing point x in [0, d]
    # T(x) = sqrt(x^2 + h1^2)/c1 + sqrt((d-x)^2 + h2^2)/c2
    dom = (0.0, d)
    T = cj.chebfun(
        lambda x: jnp.sqrt(x**2 + h1**2) / c1 +
                  jnp.sqrt((d - x)**2 + h2**2) / c2,
        domain=dom
    )

    x_min, T_min = T.min()
    print(f"\nSource at (0, {h1}), target at ({d}, -{h2})")
    print(f"Speeds: c1={c1}, c2={c2}")
    print(f"  Optimal crossing x* = {x_min:.12f}")
    print(f"  Minimum travel time T* = {T_min:.12f}")

    # Exact: Snell's law says sin(theta1)/c1 = sin(theta2)/c2
    # sin(theta1) = x / sqrt(x^2 + h1^2), sin(theta2) = (d-x)/sqrt((d-x)^2+h2^2)
    from scipy.optimize import brentq
    def snell_eq(x_):
        return (x_ / np.sqrt(x_**2 + h1**2) / c1 -
                (d - x_) / np.sqrt((d - x_)**2 + h2**2) / c2)

    x_snell = brentq(snell_eq, 0.001, d - 0.001)
    print(f"  Exact (Snell) x* = {x_snell:.12f}")
    print(f"  Error = {abs(x_min - x_snell):.2e}")
    assert abs(x_min - x_snell) < 1e-9

    # Verify Snell's law at the optimum
    sin_theta1 = x_min / np.sqrt(x_min**2 + h1**2)
    sin_theta2 = (d - x_min) / np.sqrt((d - x_min)**2 + h2**2)
    snell_ratio1 = sin_theta1 / c1
    snell_ratio2 = sin_theta2 / c2
    print(f"\nSnell's law check:")
    print(f"  sin(theta1)/c1 = {snell_ratio1:.12f}")
    print(f"  sin(theta2)/c2 = {snell_ratio2:.12f}")
    print(f"  Difference = {abs(snell_ratio1 - snell_ratio2):.2e}")
    assert abs(snell_ratio1 - snell_ratio2) < 1e-9

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(T, title="Snell's law: travel time T(x)", ylabel="T")
    ax.axvline(float(x_min), color="#E04040", linewidth=1.2,
               linestyle="--", label=f"x* = {float(x_min):.4f}")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "snells_law.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
