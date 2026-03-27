"""Chebyshev polynomials as plotted by Fornberg and Higham & Higham.

Creates 3D plots of Chebyshev polynomials T_k(x) for selected degrees k,
as appear in Fornberg (1996) p.159 and Higham & Higham (2005) p.259.
Also shows Legendre polynomials P_k(x) for comparison.

Following cheb/ChebPolysHigham.m by Nick Trefethen (December 2011).

Original MATLAB: https://www.chebfun.org/examples/cheb/ChebPolysHigham.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import chebyshev as C
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def cheb_poly(k, x):
    """Evaluate Chebyshev polynomial T_k(x) at x."""
    return np.cos(k * np.arccos(np.clip(x, -1, 1)))


def legendre_poly(k, x):
    """Evaluate Legendre polynomial P_k(x) at x using recurrence."""
    if k == 0:
        return np.ones_like(x)
    elif k == 1:
        return x.copy()
    p_prev = np.ones_like(x)
    p_curr = x.copy()
    for n in range(1, k):
        p_next = ((2*n + 1) * x * p_curr - n * p_prev) / (n + 1)
        p_prev = p_curr
        p_curr = p_next
    return p_curr


def run():
    print("=" * 60)
    print("Chebyshev polynomials in 3D (Fornberg/Higham style)")
    print("=" * 60)

    degrees = [0, 2, 4, 10, 20, 40, 60]
    x = np.linspace(-1, 1, 300)

    print(f"\nPlotting T_k(x) for k = {degrees}")

    # Verify a few Chebyshev polynomials
    print("\nChecks:")
    # T_2(x) = 2x^2 - 1
    T2 = cheb_poly(2, x)
    T2_expected = 2*x**2 - 1
    print(f"  T_2(0.5) = {cheb_poly(2, np.array([0.5]))[0]:.6f} "
          f"(expected {2*0.25-1:.6f})")
    assert np.max(np.abs(T2 - T2_expected)) < 1e-12, "T_2 formula incorrect"

    # T_k(1) = 1 for all k
    for k in degrees:
        val = cheb_poly(k, np.array([1.0]))[0]
        assert abs(val - 1.0) < 1e-10, f"T_{k}(1) = {val} != 1"
    print("  T_k(1) = 1 for all k: PASS")

    # T_k(-1) = (-1)^k
    for k in degrees:
        val = cheb_poly(k, np.array([-1.0]))[0]
        expected = (-1)**k
        assert abs(val - expected) < 1e-10, f"T_{k}(-1) = {val} != {expected}"
    print("  T_k(-1) = (-1)^k for all k: PASS")

    # Also check Legendre
    print(f"\n  P_0(0.7) = {legendre_poly(0, np.array([0.7]))[0]:.4f} (expected 1.0)")
    print(f"  P_1(0.7) = {legendre_poly(1, np.array([0.7]))[0]:.4f} (expected 0.7)")
    print(f"  P_2(0.5) = {legendre_poly(2, np.array([0.5]))[0]:.6f}"
          f" (expected {(3*0.25-1)/2:.6f})")

    print("\nPASS: all Chebyshev polynomial checks passed")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig = plt.figure()

    # Chebyshev polynomials
    ax1 = fig.add_subplot(121, projection='3d')
    colors = plt.cm.tab10(np.linspace(0, 1, len(degrees)))
    for j, (k, color) in enumerate(zip(degrees, colors)):
        y_vals = cheb_poly(k, x)
        ones = np.ones_like(x) * (j + 1)
        ax1.plot(ones, x, y_vals, color=color, linewidth=1.6)
    ax1.set_xlabel("k"); ax1.set_ylabel("x"); ax1.set_zlabel("T_k(x)")
    ax1.set_title("Chebyshev polynomials T_k(x)", fontsize=11)
    ax1.set_xticks(range(1, len(degrees) + 1))
    ax1.set_xticklabels(degrees)
    ax1.view_init(elev=28, azim=-72)

    # Legendre polynomials
    ax2 = fig.add_subplot(122, projection='3d')
    for j, (k, color) in enumerate(zip(degrees, colors)):
        y_vals = legendre_poly(k, x)
        ones = np.ones_like(x) * (j + 1)
        ax2.plot(ones, x, y_vals, color=color, linewidth=1.6)
    ax2.set_xlabel("k"); ax2.set_ylabel("x"); ax2.set_zlabel("P_k(x)")
    ax2.set_title("Legendre polynomials P_k(x)", fontsize=11)
    ax2.set_xticks(range(1, len(degrees) + 1))
    ax2.set_xticklabels(degrees)
    ax2.view_init(elev=28, azim=-72)

    fig.suptitle("Chebyshev and Legendre polynomials (Fornberg/Higham style)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "cheb_polys_higham.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
