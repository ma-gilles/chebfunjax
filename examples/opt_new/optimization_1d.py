"""Optimization of 1D chebfuns.

Finds extrema of complex 1D functions using chebfunjax root-finding and min/max,
following opt/ExtremeExtrema.m by Trefethen (September 2010) and
opt/GlobalMinimum.m by Alex Townsend (March 2013).

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
    print("Optimization of 1D chebfuns")
    print("=" * 60)

    # --- Extreme extrema ---
    # f(x) = sin(x) * sin(2x) * sin(3x) on [0, 10]
    print("\nExtrema of sin(x)*sin(2x)*sin(3x) on [0, 10]:")
    f = cj.chebfun(lambda x: jnp.sin(x) * jnp.sin(2*x) * jnp.sin(3*x),
                   domain=[0.0, 10.0])

    # Global min and max: returns (x_location, value)
    x_max, f_max_val = f.max()
    x_min, f_min_val = f.min()
    print(f"  Global max: {float(f_max_val):.8f} at x = {float(x_max):.8f}")
    print(f"  Global min: {float(f_min_val):.8f} at x = {float(x_min):.8f}")
    assert float(f_max_val) > 0
    assert float(f_min_val) < 0
    assert 0 <= float(x_max) <= 10
    assert 0 <= float(x_min) <= 10

    # Verify by checking derivative is zero at extrema
    fp = f.diff()
    fp_at_max = float(fp(jnp.array(float(x_max))))
    fp_at_min = float(fp(jnp.array(float(x_min))))
    print(f"  f'(x_max) = {fp_at_max:.2e}  (expected ~0)")
    print(f"  f'(x_min) = {fp_at_min:.2e}  (expected ~0)")
    assert abs(fp_at_max) < 1e-4
    assert abs(fp_at_min) < 1e-4

    # --- Rosenbrock function (1D slices) ---
    # Rosenbrock: (1-x)^2 + 100(y-x^2)^2
    # Min at x=y=1
    print("\nRosenbrock banana function (y = x^2 slice):")
    rosen_y_fixed = cj.chebfun(
        lambda x: (1 - x)**2 + 100 * (x**2 - x**2)**2,  # y = x^2 slice
        domain=[-2.0, 2.0]
    )
    rosen_x_min, rosen_min_val = rosen_y_fixed.min()
    print(f"  Min of (1-x)^2 on [-2,2]: {float(rosen_min_val):.8f} at x = {float(rosen_x_min):.8f}")
    assert abs(float(rosen_min_val)) < 1e-10
    assert abs(float(rosen_x_min) - 1.0) < 1e-6

    # --- Optimizing a parameterized integral ---
    # f(a) = integral_0^1 exp(-a*x^2) dx
    # Derivative at a=1: should be -integral_0^1 x^2 exp(-x^2) dx
    print("\nParameterized integral f(a) = ∫₀¹ exp(-a*x²) dx:")
    a_vals = np.linspace(0.1, 5.0, 50)
    # Compute via chebfun integration
    integrals = []
    for a in a_vals:
        g = cj.chebfun(lambda x, a=a: jnp.exp(-a * x**2), domain=[0.0, 1.0])
        integrals.append(float(g.sum()))

    # f is monotone decreasing in a
    assert all(integrals[i] > integrals[i+1] for i in range(len(integrals)-1))
    print(f"  f(0.1) = {integrals[0]:.6f}  (expected ~0.97)")
    print(f"  f(5.0) = {integrals[-1]:.6f}  (expected ~0.15-0.4)")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3)

    x1 = np.linspace(0, 10, 500)
    y1 = np.sin(x1) * np.sin(2*x1) * np.sin(3*x1)
    axes[0].plot(x1, y1, 'b-', linewidth=1)
    axes[0].axhline(float(f_max_val), color='r', linestyle='--', alpha=0.5)
    axes[0].axhline(float(f_min_val), color='g', linestyle='--', alpha=0.5)
    axes[0].plot(float(x_max), float(f_max_val), 'rv', markersize=10, zorder=5, label='max')
    axes[0].plot(float(x_min), float(f_min_val), 'g^', markersize=10, zorder=5, label='min')
    axes[0].set_title("Extreme extrema", fontsize=11)
    axes[0].legend()

    x2 = np.linspace(-2, 2, 200)
    y2 = (1 - x2)**2
    axes[1].plot(x2, y2, 'b-', linewidth=2)
    axes[1].plot(float(rosen_x_min), float(rosen_min_val), 'r*', markersize=15, label='min')
    axes[1].set_title("(1-x)² on [-2,2]", fontsize=11)
    axes[1].legend()
    axes[1].set_ylim(-0.1, 5)

    axes[2].plot(a_vals, integrals, 'b-', linewidth=2)
    axes[2].set_title("∫₀¹ exp(-ax²) dx vs. a", fontsize=11)

    fig.suptitle("1D optimization with chebfunjax", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "optimization_1d.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
