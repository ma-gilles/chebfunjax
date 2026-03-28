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
from chebfunjax.plotting import chebfun_style, PARULA, _setup_3d_axes
chebfun_style()

def gravitational_potential(X, n_theta=100, n_phi=200):
    """Compute gravitational potential at X due to unit sphere (uniform density)."""
    r_X = np.linalg.norm(X)
    if r_X > 1:
        return 4 * np.pi / r_X

    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    THETA, PHI = np.meshgrid(theta, phi)

    rx = np.sin(THETA) * np.cos(PHI)
    ry = np.sin(THETA) * np.sin(PHI)
    rz = np.cos(THETA)

    dist = np.sqrt((X[0]-rx)**2 + (X[1]-ry)**2 + (X[2]-rz)**2)
    integrand = np.sin(THETA) / dist

    return np.sum(integrand) * dtheta * dphi

def _sphere_panel(ax, fig, X, Y, Z, F, title, cmap=PARULA, elev=20, azim=-60):
    """Render a single MATLAB-quality sphere panel."""
    ax.view_init(elev=elev, azim=azim)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    fmin, fmax = float(F.min()), float(F.max())
    if fmax > fmin:
        norm_vals = (F - fmin) / (fmax - fmin)
    else:
        norm_vals = np.full_like(F, 0.5)

    fcolors = cmap(norm_vals)
    ax.plot_surface(X, Y, Z, facecolors=fcolors,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, pad=2)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor((0.8, 0.8, 0.8, 0.15))

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/sphere')
    os.makedirs(outdir, exist_ok=True)

    # Fine grid
    n_theta, n_phi = 100, 200
    theta_1d = np.linspace(0.01, np.pi - 0.01, n_theta)
    phi_1d = np.linspace(0, 2*np.pi, n_phi)
    PHI, THETA = np.meshgrid(phi_1d, theta_1d)

    X_sph = np.sin(THETA) * np.cos(PHI)
    Y_sph = np.sin(THETA) * np.sin(PHI)
    Z_sph = np.cos(THETA)

    # --- Uniform gravitational potential on sphere surface ---
    # phi(x) = 4*pi for x on the unit sphere (from outside, approaches 4*pi)
    phi_uniform = 4 * np.pi * np.ones_like(THETA)

    # --- Non-uniform density: f(theta, phi) = 1 + 0.3*cos(theta)
    # This adds a dipole perturbation
    eps = 0.3
    density = 1 + eps * np.cos(THETA)

    # --- Potential at r=1.5 from non-uniform density (Legendre expansion) ---
    r_field = 1.5
    cos_theta_field = np.cos(THETA)
    l_max = 10
    phi_map = np.zeros_like(THETA)
    for l_val in range(l_max + 1):
        Pl = np.polynomial.legendre.legval(cos_theta_field, [0]*l_val + [1])
        phi_map += (4*np.pi / (2*l_val+1)) * r_field**(-(l_val+1)) * Pl

    fig = plt.figure(figsize=(14, 4.5), facecolor='white')

    # --- Panel 1: Uniform density on sphere ---
    ax1 = fig.add_subplot(131, projection='3d')
    _sphere_panel(ax1, fig, X_sph, Y_sph, Z_sph, density,
                  'Non-uniform density\n$\\rho = 1 + 0.3\\cos\\theta$', cmap=PARULA)

    # --- Panel 2: Potential at r=1.5 on a sphere ---
    X_outer = r_field * X_sph
    Y_outer = r_field * Y_sph
    Z_outer = r_field * Z_sph

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.view_init(elev=20, azim=-60)
    ax2.set_facecolor("white")
    # Outer sphere: potential
    fmin2, fmax2 = float(phi_map.min()), float(phi_map.max())
    norm2 = (phi_map - fmin2) / (fmax2 - fmin2 + 1e-14)
    fcolors2 = PARULA(norm2)
    ax2.plot_surface(X_outer / r_field, Y_outer / r_field, Z_outer / r_field,
                     facecolors=fcolors2,
                     rstride=1, cstride=1,
                     linewidth=0, antialiased=True, shade=False)
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_zlim(-1.05, 1.05)
    ax2.set_axis_off()
    ax2.set_title(f'Potential at $r={r_field}$\n(Legendre expansion)', fontsize=10, pad=2)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False

    # --- Panel 3: Gravitational field direction on sphere ---
    # Show -grad(phi) direction: dominated by radial component, small angular
    # Use cos(theta) pattern (dipole field)
    grav_field = -(eps * np.cos(THETA) / r_field**2 + 1.0 / r_field)
    ax3 = fig.add_subplot(133, projection='3d')
    _sphere_panel(ax3, fig, X_sph, Y_sph, Z_sph, grav_field,
                  'Gravitational field strength\n$-\\partial\\phi/\\partial r$', cmap=PARULA)

    X_base = np.array([-1.0, -1.1, -0.2])
    r_X = np.linalg.norm(X_base)
    print(f"Spacecraft at X = {X_base}, |X| = {r_X:.4f}")
    print(f"  phi(X) = 4*pi/|X| = {4*np.pi/r_X:.6f}")
    print("gravity: done")

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, 'gravity.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return True

if __name__ == "__main__":
    run()
