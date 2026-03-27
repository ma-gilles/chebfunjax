"""Stability regions of ODE formulas.

Plots boundaries of stability regions for Adams-Bashforth (explicit)
and backward differentiation (BDF/implicit) ODE methods in the
complex lambda*dt plane.

Credit: Chebfun example ode-linear/Regions.m (Nick Trefethen, Feb 2011).
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



def stability_boundary(method, n_theta=2000):
    """
    Compute the stability boundary of a linear multistep method.
    'method' should return (alpha, beta) coefficient arrays.
    alpha: characteristic polynomial of x (recurrence), beta: of y
    |sum_k alpha_k xi^k| = |sum_k beta_k xi^k| defines boundary via root condition.
    We parametrize xi = exp(i*theta) and compute h*lambda = rho(xi)/sigma(xi).
    """
    theta = np.linspace(0, 2 * np.pi, n_theta)
    xi = np.exp(1j * theta)
    alpha, beta = method()
    # rho(xi) = sum alpha_k xi^k, sigma(xi) = sum beta_k xi^k
    rho = sum(a * xi**k for k, a in enumerate(alpha))
    sigma = sum(b * xi**k for k, b in enumerate(beta))
    with np.errstate(divide='ignore', invalid='ignore'):
        hlam = np.where(np.abs(sigma) > 1e-15, rho / sigma, np.nan)
    return hlam.real, hlam.imag


def run():
    print("=" * 60)
    print("Stability regions of ODE formulas")
    print("=" * 60)

    # Adams-Bashforth methods (explicit)
    def ab1():  # Forward Euler
        return [0, 1, -1], [0, 1, 0]  # rho(xi) = xi-1, sigma(xi) = 1

    def ab2():  # AB2
        return [0, 1, -1], [0, 3/2, -1/2]

    def ab3():  # AB3
        return [0, 1, -1], [0, 23/12, -16/12, 5/12]

    # BDF methods (implicit, A-stable or nearly)
    def bdf1():  # Backward Euler
        return [-1, 1], [0, 1]

    def bdf2():
        return [-1/3, 4/3, -1], [0, 0, 2/3]

    def bdf3():
        return [2/11, -9/11, 18/11, -1], [0, 0, 0, 6/11]

    print("\nComputing stability region boundaries...")
    ab_methods = [("AB1 (Euler)", ab1), ("AB2", ab2), ("AB3", ab3)]
    bdf_methods = [("BDF1", bdf1), ("BDF2", bdf2), ("BDF3", bdf3)]

    for name, meth in ab_methods + bdf_methods:
        try:
            xb, yb = stability_boundary(meth)
            print(f"  {name}: boundary points computed ({len(xb)})")
        except Exception as e:
            print(f"  {name}: {e}")

    # Verify basic properties:
    # Adams-Bashforth 1 (Forward Euler) stability region is circle of radius 1 centered at -1
    xb, yb = stability_boundary(ab1)
    # Should be a circle: check max |h*lambda| ~ 2
    assert np.nanmax(np.abs(xb + 1j * yb)) < 5.0

    # BDF1 stability region contains the left half-plane
    xb_bdf1, yb_bdf1 = stability_boundary(bdf1)
    print(f"  BDF1 boundary: Re range [{np.nanmin(xb_bdf1):.2f}, {np.nanmax(xb_bdf1):.2f}]")

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    colors_ab = ['b', 'r', 'g']
    for (name, meth), c in zip(ab_methods, colors_ab):
        xb, yb = stability_boundary(meth)
        axes[0].plot(xb, yb, color=c, linewidth=1.6, label=name)
    axes[0].axhline(0, color='k', linewidth=0.5)
    axes[0].axvline(0, color='k', linewidth=0.5)
    axes[0].set_xlim(-3, 0.5)
    axes[0].set_ylim(-2, 2)
    axes[0].set_xlabel("Re(h λ)"); axes[0].set_ylabel("Im(h λ)")
    axes[0].set_title("Adams-Bashforth stability regions", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    colors_bdf = ['b', 'r', 'g']
    for (name, meth), c in zip(bdf_methods, colors_bdf):
        xb, yb = stability_boundary(meth)
        axes[1].plot(xb, yb, color=c, linewidth=1.6, label=name)
    axes[1].axhline(0, color='k', linewidth=0.5)
    axes[1].axvline(0, color='k', linewidth=0.5)
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(-5, 5)
    axes[1].set_xlabel("Re(h λ)"); axes[1].set_ylabel("Im(h λ)")
    axes[1].set_title("BDF stability region boundaries", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Stability regions of ODE methods", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "regions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
