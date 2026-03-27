"""Global minimum of smooth 1D functions.

Demonstrates finding global minima of smooth 1D functions using
Chebfun's min() method, which is exact to machine precision.

Credit: Inspired by Chebfun opt/GlobalMinimum.m.
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
    print("Global minimum of smooth functions")
    print("=" * 60)

    # --- Unimodal: f(x) = x^2 + sin(x) on [-pi, pi] ----------------
    dom = (-float(jnp.pi), float(jnp.pi))
    f = cj.chebfun(lambda x: x**2 + jnp.sin(x), domain=dom)
    x_min, y_min = f.min()
    print(f"\nf(x) = x^2 + sin(x) on [-pi, pi]:")
    print(f"  min at x* = {x_min:.12f}")
    print(f"  min value = {y_min:.12f}")
    # f'(x) = 2x + cos(x) = 0 near x ~ -0.45
    df = f.diff()
    df_at_min = float(df(jnp.array(x_min)))
    print(f"  f'(x*) = {df_at_min:.2e}  (should be ~0)")
    assert abs(df_at_min) < 1e-9
    assert x_min < 0.0 and x_min > -1.0

    # --- Multimodal: f(x) = sin(8x) + sin(5x) + 0.3*x ---------------
    dom2 = (0.0, float(2.0 * jnp.pi))
    g = cj.chebfun(lambda x: jnp.sin(8.0*x) + jnp.sin(5.0*x) + 0.3*x, domain=dom2)
    x_min2, y_min2 = g.min()
    x_max2, y_max2 = g.max()
    print(f"\ng(x) = sin(8x) + sin(5x) + 0.3x on [0, 2*pi]:")
    print(f"  Global min at x* = {x_min2:.12f}, value = {y_min2:.12f}")
    print(f"  Global max at x* = {x_max2:.12f}, value = {y_max2:.12f}")
    # Verify these are actual global extrema by checking all critical points
    dg = g.diff()
    crit_pts = dg.roots()
    crit_arr = np.array(crit_pts)
    interior = crit_arr[(crit_arr > 1e-10) & (crit_arr < float(2.0*jnp.pi) - 1e-10)]
    all_vals = np.array([float(g(jnp.array(xi))) for xi in interior])
    all_vals = np.append(all_vals, [float(g(jnp.array(0.0))),
                                     float(g(jnp.array(float(2.0*jnp.pi))))])
    true_min = np.min(all_vals)
    true_max = np.max(all_vals)
    assert abs(y_min2 - true_min) < 1e-9
    assert abs(y_max2 - true_max) < 1e-9

    # --- Rosenbrock 1D: min over y of f(1, y) = (1-1)^2 + 100(y-1)^2 = 0 at y=1
    dom_y = (-1.0, 3.0)
    f_y1 = cj.chebfun(lambda y: 100.0 * (y - 1.0)**2, domain=dom_y)
    min_y_pt, min_y_val = f_y1.min()
    print(f"\nRosenbrock: min over y of 100*(y-1)^2:")
    print(f"  min at y* = {min_y_pt:.12f}  (exact: 1.0)")
    print(f"  min value = {min_y_val:.2e}  (exact: 0)")
    assert abs(min_y_pt - 1.0) < 1e-10
    assert abs(min_y_val) < 1e-14

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f, title="x² + sin(x): global minimum on [−π, π]")
    ax.plot(float(x_min), float(y_min), "r*", markersize=12, label="min")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "minimum_of_smooth_function.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plot(g, title="sin(8x)+sin(5x)+0.3x on [0, 2π]")
    fig2.savefig(os.path.join(_here, "minimum_of_smooth_function2.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
