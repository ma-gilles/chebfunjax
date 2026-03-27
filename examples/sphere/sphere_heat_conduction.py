"""Solving the heat equation on the unit sphere.

Uses the eigenfunction expansion of the Laplace-Beltrami operator to solve
the spherical heat equation u_t = kappa * Delta_S u, where Delta_S is the
Laplace-Beltrami operator on S^2.
Translated from sphere/SphereHeatConduction.m.

Original: https://www.chebfun.org/examples/sphere/SphereHeatConduction.html
Authors: Alex Townsend and Grady Wright, May 2016
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.special import sph_harm_y
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def spherical_harmonic_real(l, m, theta, phi):
    """Real spherical harmonic."""
    Ylm = sph_harm_y(l, abs(m), theta, phi)
    if m > 0:
        return np.sqrt(2) * (-1)**m * np.real(Ylm)
    elif m < 0:
        return np.sqrt(2) * (-1)**m * np.imag(Ylm)
    else:
        return np.real(Ylm)


def sh_coefficient(f, theta, phi, l, m):
    """Compute spherical harmonic coefficient of f."""
    Y = spherical_harmonic_real(l, m, theta, phi)
    dtheta = theta[1, 0] - theta[0, 0]
    dphi = phi[0, 1] - phi[0, 0]
    return np.sum(f * Y * np.sin(theta)) * dtheta * dphi


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/sphere')
    os.makedirs(outdir, exist_ok=True)

    # Grid
    n_theta, n_phi = 80, 160
    theta_1d = np.linspace(0.02, np.pi - 0.02, n_theta)
    phi_1d = np.linspace(0, 2*np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta_1d, phi_1d, indexing='ij')

    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    # Initial condition: localized bump at north pole
    sigma = 0.3
    u0 = np.exp(-((THETA - 0.3)**2 + (PHI - np.pi/2)**2) / sigma**2)

    # Diffusivity
    kappa = 0.1

    # Compute SH coefficients of u0
    l_max = 8
    coeffs = {}
    for l in range(l_max + 1):
        for m in range(-l, l+1):
            c = sh_coefficient(u0, THETA, PHI, l, m)
            if abs(c) > 1e-8:
                coeffs[(l, m)] = c

    print(f"Heat equation on sphere (kappa={kappa}):")
    print(f"  Initial condition: localized bump")
    print(f"  Non-zero SH coefficients (|c|>1e-8): {len(coeffs)}")

    def heat_solution(t):
        """Compute heat equation solution at time t via eigenfunction expansion."""
        u = np.zeros_like(u0)
        for (l, m), c in coeffs.items():
            # Eigenvalue of Laplace-Beltrami: -l*(l+1)
            decay = np.exp(-kappa * l * (l+1) * t)
            Y = spherical_harmonic_real(l, m, THETA, PHI)
            u += c * decay * Y
        return u

    fig = plt.figure(figsize=(15, 5))

    times = [0, 0.5, 2.0]
    ax_list = [fig.add_subplot(1, 3, i+1, projection='3d') for i in range(3)]

    vmin, vmax = -0.2, u0.max()

    for ax, t in zip(ax_list, times):
        u_t = heat_solution(t)
        u_norm = (u_t - vmin) / (vmax - vmin + 1e-14)
        u_norm = np.clip(u_norm, 0, 1)
        ax.plot_surface(X, Y, Z, facecolors=plt.cm.hot(u_norm),
                         alpha=0.9, linewidth=0)
        ax.set_title(f't = {t}', fontsize=11)
        ax.set_axis_off()

        print(f"  t={t}: max u = {u_t.max():.4f}, total = {np.sum(u_t*np.sin(THETA)):.4f}")

    fig.suptitle('Heat Equation on the Unit Sphere\nu_t = κ·Δ_S u, κ=0.1', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'sphere_heat_conduction.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("sphere_heat_conduction: done")
    return True


if __name__ == "__main__":
    run()
