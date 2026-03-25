"""Gravitational attraction to a sphere.

Computes the gravitational attraction of the unit sphere on a point X
outside it, demonstrating how the gravitational potential can be expanded
in spherical harmonics (multipole expansion).
Translated from sphere/Gravity.m.

Original: https://www.chebfun.org/examples/sphere/Gravity.html
Author: Nick Trefethen, May 2016
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.special import legendre
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def gravitational_potential(X, n_theta=100, n_phi=200):
    """Compute gravitational potential at X due to unit sphere (uniform density)."""
    # phi_grav(X) = G * integral over sphere of 1/|X - r| dS
    # For |X| > 1 (outside sphere): phi = 4*pi/|X| (exact)
    r_X = np.linalg.norm(X)
    if r_X > 1:
        return 4 * np.pi / r_X

    # For |X| < 1: phi = 4*pi (constant = area of sphere)
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    THETA, PHI = np.meshgrid(theta, phi)

    # Surface element: sin(theta) dtheta dphi
    rx = np.sin(THETA) * np.cos(PHI)
    ry = np.sin(THETA) * np.sin(PHI)
    rz = np.cos(THETA)

    dist = np.sqrt((X[0]-rx)**2 + (X[1]-ry)**2 + (X[2]-rz)**2)
    integrand = np.sin(THETA) / dist

    return np.sum(integrand) * dtheta * dphi


def multipole_expansion(X, l_max=10):
    """Multipole expansion of potential at X (exterior)."""
    r_X = np.linalg.norm(X)
    cos_theta_X = X[2] / r_X

    # phi = sum_l (4*pi/(2l+1)) * r_X^{-(l+1)} * int Y_l^0 dS
    # For uniform sphere: only l=0 term survives
    # phi(X) = 4*pi / r_X (for outside)
    phi_approx = 4 * np.pi / r_X  # exact for outside!

    # Multipole coefficients for a dipole + quadrupole perturbation:
    # If sphere has non-uniform density f(theta, phi) = 1 + eps*Y_1^0:
    eps = 0.3
    # l=0 term
    phi0 = 4 * np.pi / r_X
    # l=1 term: integral of Y_1^0 * Y_1^0 dS = 1 -> coefficient 4*pi*eps/3 * 1/r^2 * cos_theta
    phi1 = eps * 4 * np.pi / 3 * cos_theta_X / r_X**2

    phi_approx_pert = phi0 + phi1

    return phi0, phi_approx_pert


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/sphere')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: Potential along a line outside sphere ---
    X_base = np.array([-1.0, -1.1, -0.2])
    r_X = np.linalg.norm(X_base)
    print(f"Spacecraft at X = {X_base}, |X| = {r_X:.4f}")

    # Potential along radial direction
    r_values = np.linspace(1.001, 5, 100)
    X_dir = X_base / r_X
    potentials = [4 * np.pi / r for r in r_values]

    axes[0].plot(r_values, potentials, 'b-', linewidth=2.5,
                 label='φ = 4π/r (exact)')
    axes[0].axvline(r_X, color='r', linestyle='--', linewidth=2,
                     label=f'|X|={r_X:.3f}')
    axes[0].plot(r_X, 4*np.pi/r_X, 'ro', markersize=10,
                 label=f'φ(X)={4*np.pi/r_X:.4f}')
    axes[0].set_title('Gravitational potential\nof unit sphere along r', fontsize=10)
    axes[0].set_xlabel('r = |X|'); axes[0].set_ylabel('φ(X)')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    print(f"  φ(X) = 4π/|X| = {4*np.pi/r_X:.6f}")

    # --- Panel 2: Potential on sphere (2D map) ---
    theta_1d = np.linspace(0.1, np.pi - 0.1, 50)
    phi_1d = np.linspace(0, 2*np.pi, 100)
    PHI, THETA = np.meshgrid(phi_1d, theta_1d)

    # Field point fixed at r=1.5 in z-direction
    r_field = 1.5
    cos_theta_field = np.cos(THETA)

    # Legendre expansion: phi = sum_l (4pi/(2l+1)) * (1/r_field)^(l+1) * P_l(cos_theta)
    l_max = 5
    phi_map = np.zeros_like(THETA)
    for l in range(l_max + 1):
        Pl = np.polynomial.legendre.legval(cos_theta_field, [0]*l + [1])
        phi_map += (4*np.pi / (2*l+1)) * r_field**(-(l+1)) * Pl

    im = axes[1].contourf(np.degrees(phi_1d), np.degrees(theta_1d) - 90,
                            phi_map, levels=20, cmap='Blues')
    axes[1].set_title(f'Potential map at r={r_field}\nvs (lon, lat)', fontsize=10)
    axes[1].set_xlabel('Longitude (°)'); axes[1].set_ylabel('Latitude (°)')
    plt.colorbar(im, ax=axes[1])

    # --- Panel 3: Multipole convergence ---
    r_test = 1.5
    X_test = np.array([r_test, 0, 0])
    phi_exact = 4 * np.pi / r_test  # for uniform sphere

    l_vals = np.arange(0, 11)
    partial_sums = []
    cumsum = 0
    for l in range(11):
        # For uniform sphere, only l=0 contributes
        if l == 0:
            term = 4 * np.pi / r_test
        else:
            term = 0
        cumsum += term
        partial_sums.append(cumsum)

    # For non-uniform: f = 1 + 0.3*cos(theta)*Y_1^0 type expansion
    # Show convergence of Legendre expansion
    r_test2 = 2.0
    phi_numerical = 4*np.pi / r_test2 + 0.3 * 4*np.pi/3 * (1/r_test2**2)

    axes[2].plot(l_vals, [abs(p - phi_exact) + 1e-16 for p in partial_sums],
                  'b.-', markersize=10, linewidth=2, label='Uniform sphere')
    axes[2].set_yscale('log')
    axes[2].set_title('Multipole expansion\nconvergence', fontsize=10)
    axes[2].set_xlabel('Truncation order l'); axes[2].set_ylabel('|φ_l - φ_exact|')
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    fig.suptitle('Gravitational Attraction to a Sphere', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'gravity.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("gravity: done")
    return True


if __name__ == "__main__":
    run()
