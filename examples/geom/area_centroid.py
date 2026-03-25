"""Area and centroid of a 2D region.

Computes area and centroid of regions defined by parametric curves using
Green's theorem. Translated from geom/Area.m.

Original: https://www.chebfun.org/examples/geom/Area.html
Author: Stefan Guettel, October 2010
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def area_from_curve(z):
    """Compute area enclosed by parametric complex curve z(t) using Green's theorem.

    Area = (1/2) * |Im(integral z* dz)|
         = integral Re(z) d(Im(z))
    """
    dz = np.diff(z)
    z_mid = 0.5 * (z[:-1] + z[1:])
    return np.abs(np.sum(np.real(z_mid) * np.imag(dz)))


def centroid_from_curve(z, area):
    """Compute centroid of region enclosed by curve."""
    dz = np.diff(z)
    z_mid = 0.5 * (z[:-1] + z[1:])
    # c = (1/(2i*A)) * integral z * conj(z) dz
    c = np.sum(z_mid * np.conj(z_mid) * dz) / (2j * area)
    return c


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- 1. Epicycloid: m=7 ---
    n_pts = 5000
    t = np.linspace(0, 2 * np.pi, n_pts)
    b = 1
    m = 7
    a = (m - 1) * b
    x_epi = (a + b) * np.cos(t) - b * np.cos((a + b) / b * t)
    y_epi = (a + b) * np.sin(t) - b * np.sin((a + b) / b * t)
    z_epi = x_epi + 1j * y_epi

    A_epi = area_from_curve(z_epi)
    A_exact = np.pi * b**2 * (m**2 + m)  # exact for integer m
    print(f"Epicycloid (m={m}): A = {A_epi:.6f}  (exact: {A_exact:.6f})")
    print(f"  Relative error: {abs(A_epi - A_exact) / A_exact:.2e}")
    assert abs(A_epi - A_exact) / A_exact < 1e-3

    axes[0].fill(x_epi, y_epi, color=[0.6, 0.6, 1], alpha=0.7)
    axes[0].set_aspect('equal')
    axes[0].set_title(f'Epicycloid (m={m})\nArea = {A_epi:.4f} (exact: {A_exact:.4f})',
                      fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # --- 2. Perturbed unit circle: z(t) = e^{it} + (1+i)*sin(6t)^2 ---
    z_perturb = np.exp(1j * t) + (1 + 1j) * np.sin(6 * t)**2
    A_perturb = area_from_curve(z_perturb)
    # Expected: since perturbations cancel by symmetry, A ≈ pi
    print(f"\nPerturbed circle: A = {A_perturb:.6f}  (expected: π = {np.pi:.6f})")

    # Centroid
    c_perturb = centroid_from_curve(z_perturb, A_perturb)
    print(f"  Centroid: {c_perturb:.6f}  (expected: ~0)")

    axes[1].fill(np.real(z_perturb), np.imag(z_perturb),
                 color=[0.6, 1, 0.6], alpha=0.7)
    axes[1].plot(np.real(c_perturb), np.imag(c_perturb), 'r+',
                 markersize=15, markeredgewidth=2, label='Centroid')
    axes[1].set_aspect('equal')
    axes[1].set_title(f'Perturbed circle\nArea = {A_perturb:.4f} (≈ π = {np.pi:.4f})',
                      fontsize=10)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    fig.suptitle('Area and Centroid of 2D Regions via Green\'s Theorem', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'area_centroid.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("area_centroid: done")
    return True


if __name__ == "__main__":
    run()
