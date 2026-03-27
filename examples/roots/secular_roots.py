"""Roots of a secular equation with poles.

A secular equation is a rational function with partial-fraction form
r(x) = 1 + sum a_j/(b_j - x). Such functions arise in divide-and-conquer
eigenvalue algorithms. Chebfun finds all real sign-change crossings.

Credit: Inspired by Chebfun example roots/SecularRoots.m
(Nick Trefethen, November 2010).
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
    print("Roots of a secular equation with poles")
    print("=" * 60)

    # r(x) = 1 + 1/(1-x) + 1/(2-x) + 1/(3-x) + 1/(4-x)
    # Poles at x = 1, 2, 3, 4; one root between each pair of adjacent poles
    # and one root to the right of x=4.
    poles = [1.0, 2.0, 3.0, 4.0]

    # We compute on each interval between (and around) the poles
    intervals = [(-5.0, 0.9), (1.1, 1.9), (2.1, 2.9), (3.1, 3.9), (4.1, 10.0)]

    def secular(x):
        return (1.0 +
                1.0 / (1.0 - x) +
                1.0 / (2.0 - x) +
                1.0 / (3.0 - x) +
                1.0 / (4.0 - x))

    all_roots = []
    for a, b in intervals:
        f = cj.chebfun(secular, domain=(a, b))
        r = f.roots()
        all_roots.extend([float(x) for x in np.asarray(r)])

    all_roots = sorted(all_roots)
    print(f"\nRoots of secular equation:")
    for i, r in enumerate(all_roots):
        print(f"  r[{i}] = {r:.14f}   f(r) = {secular(r):.2e}")

    # There should be 4 roots (one between each pair of poles and one beyond)
    # Actually the standard secular equation with N=4 poles has N roots
    assert len(all_roots) == 4, f"Expected 4 roots, got {len(all_roots)}"
    for r in all_roots:
        assert abs(secular(r)) < 1e-6, f"f(root) = {secular(r)}"

    print(f"\nNumber of roots found: {len(all_roots)}")
    print("All roots verified: f(r) ~ 0")

    # --- Wide view on [-5, 10] to show the rational function ---
    xs_plot = []
    ys_plot = []
    for a, b in intervals:
        xs_seg = np.linspace(a, b, 300)
        ys_seg = np.array([secular(float(x)) for x in xs_seg])
        xs_plot.append(xs_seg)
        ys_plot.append(ys_seg)

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plt.subplots()
    for xs_seg, ys_seg in zip(xs_plot, ys_plot):
        ax.plot(xs_seg, ys_seg, color="#1e77b4", linewidth=1.5)
    ax.axhline(0, color="k", linewidth=0.5)
    for p in poles:
        ax.axvline(p, color="#999999", linestyle="--", linewidth=0.8)
    ax.plot(all_roots, [0.0] * len(all_roots), "or", markersize=8, label="roots")
    ax.set_ylim(-8, 8)
    ax.set_xlim(-5, 10)
    ax.set_title("Secular equation: poles (grey dashed) and roots (red dots)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "secular_roots.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
