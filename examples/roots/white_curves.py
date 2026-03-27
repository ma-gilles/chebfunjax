"""The white curves of Ortiz and Rivlin.

Ortiz and Rivlin (1983) observed that the graph of Chebyshev polynomials
on [-1,1] shows striking "white curve" patterns.  This example plots many
overlapping Chebyshev polynomials and highlights the root structure.

Credit: Inspired by Chebfun example roots/WhiteCurves.m
(Stefan Guettel, November 2011).
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
    print("White curves of Ortiz and Rivlin")
    print("=" * 60)

    xs = np.linspace(-1.0, 1.0, 1000)
    pi = float(jnp.pi)

    N = 30  # number of Chebyshev polynomials to plot

    print(f"\nOverlaying T_1 through T_{N} on [-1,1]...")

    # T_n(x) = cos(n * arccos(x))
    all_ys = np.zeros((N, len(xs)))
    for n in range(1, N + 1):
        # Use recurrence for stability
        T_n = np.cos(n * np.arccos(np.clip(xs, -1.0, 1.0)))
        all_ys[n - 1] = T_n

    # Verify a few Chebyshev polynomials via chebfunjax
    print("\nVerifying T_n via Chebfun:")
    for n in [5, 10, 20]:
        coeffs = jnp.zeros(n + 1).at[n].set(1.0)
        f = cj.Chebfun.from_coeffs(coeffs)
        test_pts = jnp.array([-0.7, 0.0, 0.5, 0.9])
        computed = np.array(f(test_pts))
        exact = np.cos(n * np.arccos(np.clip(np.array(test_pts), -1.0, 1.0)))
        err = np.max(np.abs(computed - exact))
        print(f"  T_{n}: max err = {err:.2e}")
        assert err < 1e-10, f"T_{n} error: {err}"

    # Count total roots across all T_n (T_n has n roots)
    total_expected = sum(range(1, N + 1))  # 1+2+...+N
    print(f"\nTotal roots of T_1..T_{N} = {total_expected}")

    # Collect all Chebyshev roots
    all_roots_x = []
    all_roots_n = []
    for n in range(1, N + 1):
        # Exact roots of T_n
        k = np.arange(1, n + 1)
        r_k = np.cos((2.0 * k - 1.0) * pi / (2.0 * n))
        all_roots_x.extend(r_k.tolist())
        all_roots_n.extend([n] * n)

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    # Left: overlaid Chebyshev polynomials
    for n in range(1, N + 1):
        alpha = 0.5 if n > 15 else 0.7
        axes[0].plot(xs, all_ys[n - 1], color="#1e77b4", linewidth=0.5, alpha=alpha)
    axes[0].axhline(0, color="k", linewidth=0.5)
    axes[0].set_xlim(-1, 1)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_title(f"$T_1$ through $T_{{{N}}}$ overlaid")
    axes[0].set_xlabel("x")

    # Right: root density (histogram)
    axes[1].hist(all_roots_x, bins=80, color="#1e77b4", alpha=0.7, edgecolor="none")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Root count")
    xs_arcos = np.linspace(-1 + 0.01, 1 - 0.01, 200)
    # The arcsine density ~ 1/sqrt(1-x^2)
    density_norm = (N * (N + 1) / 2) / 80
    axes[1].plot(xs_arcos, density_norm / np.sqrt(1 - xs_arcos**2) * 0.025,
                 'r-', linewidth=1.5, label="arcsine density")
    axes[1].set_title("Root distribution (arcsine law)")
    axes[1].legend(fontsize=9)

    fig.suptitle("White curves: overlaid Chebyshev polynomials", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "white_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
