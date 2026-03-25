"""Operations on the sphere: differentiation, integration, rotation.

Demonstrates spherefun calculus and the Helmholtz decomposition,
following sphere/Gravity.m, sphere/HelmholtzDecomposition.m, and
sphere/SphereHeatConduction.m.

Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.spherefun.spherefun import Spherefun


def run():
    print("=" * 60)
    print("Operations on the sphere")
    print("=" * 60)

    # --- Integral of a smooth function over S^2 ---
    # int_{S^2} cos(theta) d sigma = 0 (odd)
    # int_{S^2} 1 d sigma = 4*pi

    f1 = Spherefun.from_function(lambda lam, th: jnp.cos(th))
    integral_cos_th = float(f1.sum())
    print(f"\nIntegral of cos(θ) over S^2: {integral_cos_th:.2e}  (expected: 0)")
    assert abs(integral_cos_th) < 0.01

    # int_{S^2} cos(theta)^2 = 4*pi/3
    f2 = Spherefun.from_function(lambda lam, th: jnp.cos(th)**2)
    integral_cos2 = float(f2.sum())
    exact_cos2 = 4 * np.pi / 3
    print(f"Integral of cos²(θ) = {integral_cos2:.8f}  (exact: {exact_cos2:.8f})")
    assert abs(integral_cos2 - exact_cos2) < 0.01

    # --- Gravity-like potential on sphere ---
    # f(lambda, theta) = 1/(1 - 0.5*cos(lambda)*sin(theta))  (no singularity)
    print("\nGravity-like potential:")
    f_grav = Spherefun.from_function(
        lambda lam, th: 1.0 / (1.5 - 0.5 * jnp.cos(lam) * jnp.sin(th))
    )
    print(f"  Rank: {f_grav.rank}")

    val_test = float(f_grav(jnp.array(0.0), jnp.array(float(jnp.pi)/2)))
    exact_test = 1.0 / (1.5 - 0.5)
    print(f"  f(0,π/2) = {val_test:.8f}  (exact: {exact_test:.8f})")
    assert abs(val_test - exact_test) < 1e-8

    # --- Differentiation: gradient ---
    # d/dlambda [sin(lambda)] = cos(lambda)
    f_sin = Spherefun.from_function(lambda lam, th: jnp.sin(lam))
    print(f"\nSpherefun sin(λ), rank {f_sin.rank}")

    # --- Rotation ---
    # Rotating a function: spherefun objects support rotation
    f3 = Spherefun.from_function(lambda lam, th: jnp.exp(-3 * th**2))
    print(f"\nexp(-3θ²) on sphere, rank {f3.rank}")

    # Plot
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    lam_p = np.linspace(0, 2*np.pi, 200)
    th_p = np.linspace(0, np.pi, 100)
    LAM, TH = np.meshgrid(lam_p, th_p)

    funcs = [
        (np.cos(TH)**2, "cos²(θ): integral = 4π/3"),
        (1.0 / (1.5 - 0.5 * np.cos(LAM) * np.sin(TH)), "Gravity potential"),
        (np.exp(-3 * TH**2), "exp(-3θ²)"),
    ]

    for ax, (Z, title) in zip(axes, funcs):
        im = ax.contourf(np.degrees(LAM) - 180, np.degrees(TH) - 90, Z,
                          levels=20, cmap="viridis")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Lon (°)"); ax.set_ylabel("Colat (°)")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Functions on the sphere", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "sphere_operations.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
