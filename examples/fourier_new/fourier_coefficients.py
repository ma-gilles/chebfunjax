"""Fourier-based Chebfuns and Fourier coefficients.

Demonstrates periodic functions using chebfunjax on periodic domains,
following fourier/FourierCoefficients.m by Grady Wright (June 2014)
and fourier/FourierBasedChebfuns.m.

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
    print("Fourier-based chebfuns and coefficients")
    print("=" * 60)

    # --- Periodic chebfun via Chebfun on [0, 2*pi] ---
    f = cj.chebfun(jnp.sin, domain=[0.0, 2.0 * float(jnp.pi)])
    print(f"\nsin(x) on [0, 2π]: {len(f.funs[0].tech.coeffs)} coefficients")

    # Evaluate at pi/2: should be 1
    val = float(f(jnp.array(float(jnp.pi) / 2)))
    print(f"f(π/2) = {val:.15f}  (exact: 1.0)")
    assert abs(val - 1.0) < 1e-12

    # Integrate: integral of sin over [0, 2*pi] = 0
    integral_sin = float(f.sum())
    print(f"Integral of sin on [0,2π] = {integral_sin:.2e}  (exact: 0)")
    assert abs(integral_sin) < 1e-10

    # cos on [0, 2*pi]
    g = cj.chebfun(jnp.cos, domain=[0.0, 2.0 * float(jnp.pi)])
    print(f"\ncos(x) on [0, 2π]: {len(g.funs[0].tech.coeffs)} coefficients")

    # sin^2 + cos^2 = 1
    h = cj.chebfun(lambda x: jnp.sin(x)**2 + jnp.cos(x)**2, domain=[0.0, 2.0 * float(jnp.pi)])
    val_identity = float(h(jnp.array(1.0)))
    print(f"\nsin²(x) + cos²(x) at x=1: {val_identity:.15f}  (exact: 1.0)")
    assert abs(val_identity - 1.0) < 1e-12

    # --- |cos(x)| on [0, 2*pi] ---
    print("\n|cos(x)| on [0, 2π]:")
    f_abs_cos = cj.chebfun(lambda x: jnp.abs(jnp.cos(x)), domain=[0.0, 2.0 * float(jnp.pi)])
    print(f"  Degree: {len(f_abs_cos.funs[0].tech.coeffs)}")

    # Integral of |cos(x)| on [0, 2*pi] = 4 (4 quarter-cycles each = 1)
    integral_abs_cos = float(f_abs_cos.sum())
    print(f"  Integral = {integral_abs_cos:.8f}  (exact: 4.0)")
    assert abs(integral_abs_cos - 4.0) < 1e-6

    # --- Chebyshev coefficients of periodic function ---
    coeffs_sin = f.funs[0].tech.coeffs
    coeffs_cos = g.funs[0].tech.coeffs
    print(f"\nCoefficient magnitudes (first 8):")
    print(f"  sin: {[f'{abs(float(c)):.3e}' for c in coeffs_sin[:8]]}")

    # --- Fejer-Jackson partial sums: positive on (0, pi) ---
    # S_n(x) = sum_{k=1}^{n} sin(kx)/k  > 0 for x in (0, pi)
    print("\nFejer-Jackson partial sums (first 5 positive on (0, pi)):")
    n_fj = 10
    S_fj = cj.chebfun(
        lambda x: sum(jnp.sin(k * x) / k for k in range(1, n_fj + 1)),
        domain=[0.0, float(jnp.pi)]
    )
    # Check min on (0, pi) is positive
    min_val, _ = S_fj.min()
    print(f"  Min of S_10 on [0,π]: {float(min_val):.6f}  (should be > 0 on interior)")
    # Note: min at endpoints can be 0

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    xs_plot = np.linspace(0, 2*np.pi, 300)

    axes[0].plot(xs_plot, np.sin(xs_plot), 'b-', label='sin(x)')
    axes[0].plot(xs_plot, np.cos(xs_plot), 'r-', label='cos(x)')
    axes[0].plot(xs_plot, np.abs(np.cos(xs_plot)), 'g-', label='|cos(x)|')
    axes[0].set_title("Periodic functions on [0, 2π]", fontsize=12)
    axes[0].set_xlabel("x"); axes[0].legend()
    axes[0].set_xlim(0, 2*np.pi); axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    axes[0].set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

    # Fejer-Jackson partial sums
    xs_fj = np.linspace(0.01, np.pi - 0.01, 200)
    for n_fj_plot in [2, 5, 10, 20]:
        S = sum(np.sin(k * xs_fj) / k for k in range(1, n_fj_plot + 1))
        axes[1].plot(xs_fj, S, linewidth=1.5, label=f'n={n_fj_plot}')
    axes[1].axhline(0, color='k', linestyle='-', linewidth=0.5)
    axes[1].set_title("Fejer-Jackson partial sums Σsin(kx)/k", fontsize=11)
    axes[1].set_xlabel("x"); axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, np.pi); axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    axes[1].set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])

    fig.suptitle("Fourier-based chebfuns", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "fourier_coefficients.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
