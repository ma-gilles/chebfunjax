"""Order stars for Runge-Kutta methods.

Plots order stars |e^{-z} R(z)| = 1 for Pade approximants to e^z,
illustrating the theory that resolves stability of ODE methods.

Credit: Chebfun example ode-linear/OrderStars.m (Nick Trefethen, Jul 2014).
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



def pade_approx(m, n, x):
    """Evaluate (m,n) Pade approximant to e^z at complex z values."""
    # Numerator and denominator via Pade formula
    # P_m(z) / Q_n(z) where Q_n is reverse of P_n with sign alternation
    from math import factorial, comb
    c_m = [1.0]
    P = np.zeros(m + 1, dtype=complex)
    Q = np.zeros(n + 1, dtype=complex)
    for k in range(m + 1):
        P[k] = factorial(m + n - k) * factorial(m) / (factorial(m + n) * factorial(k) * factorial(m - k))
    for k in range(n + 1):
        Q[k] = factorial(m + n - k) * factorial(n) / (factorial(m + n) * factorial(k) * factorial(n - k))
        if k % 2 == 1:
            Q[k] *= -1

    z = np.asarray(x, dtype=complex)
    Pz = sum(P[k] * z**k for k in range(m + 1))
    Qz = sum(Q[k] * z**k for k in range(n + 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(Qz) > 1e-15, Pz / Qz, np.nan)
    return result


def run():
    print("=" * 60)
    print("Order stars for ODE stability analysis")
    print("=" * 60)

    # Create grid in complex plane
    N_grid = 200
    x_range = np.linspace(-4, 4, N_grid)
    y_range = np.linspace(-4, 4, N_grid)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X + 1j * Y

    cases = [(1, 1), (2, 2), (1, 3), (3, 3)]

    print("\nOrder stars for Pade approximants R_{m,n}(z):")
    for m, n in cases:
        R = pade_approx(m, n, Z)
        ratio = np.abs(np.exp(-Z) * R)
        order_star_region = ratio > 1.0
        n_in = int(np.sum(order_star_region))
        print(f"  R_{m},{n}: {n_in}/{N_grid**2} grid points in order star (|e^{{-z}}R|>1)")
        assert n_in > 0

    # Verify that (1,1) Pade is e^z + O(z^3)
    z_small = np.array([0.01, 0.05, 0.1])
    R11 = pade_approx(1, 1, z_small)
    ez = np.exp(z_small)
    err11 = np.max(np.abs(R11 - ez))
    print(f"\n(1,1) Pade error at small z: {err11:.2e}")
    assert err11 < 0.01  # 2nd order approximant

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(2, 2)
    axes = axes.ravel()

    for idx, (m, n) in enumerate(cases):
        R = pade_approx(m, n, Z)
        ratio = np.abs(np.exp(-Z) * R)
        ax = axes[idx]
        ax.contourf(X, Y, ratio, levels=[0, 1], colors=['lightblue'], alpha=0.7)
        ax.contour(X, Y, ratio, levels=[1.0], colors='k', linewidths=0.8)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_title(f"Order star R_({m},{n})", fontsize=10)
        ax.set_xlabel("Re(z)"); ax.set_ylabel("Im(z)")
        ax.set_aspect('equal')

    fig.suptitle("Order stars |e⁻ᶻ R(z)| = 1 (blue = |e⁻ᶻ R| < 1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "order_stars.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
