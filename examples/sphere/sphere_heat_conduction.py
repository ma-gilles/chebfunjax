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
from chebfunjax.plotting import chebfun_style, PARULA, _setup_3d_axes
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

def _sphere_panel(ax, fig, X, Y, Z, F, title, cmap=PARULA, elev=20, azim=-60,
                  vmin=None, vmax=None):
    """Render a single MATLAB-quality sphere panel with optional global colour limits."""
    ax.view_init(elev=elev, azim=azim)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    if vmin is None:
        vmin = float(F.min())
    if vmax is None:
        vmax = float(F.max())
    if vmax > vmin:
        norm_vals = np.clip((F - vmin) / (vmax - vmin), 0, 1)
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
            decay = np.exp(-kappa * l * (l+1) * t)
            Y_sh = spherical_harmonic_real(l, m, THETA, PHI)
            u += c * decay * Y_sh
        return u

    # Use consistent colour limits across all panels
    times = [0, 0.5, 2.0]
    solutions = [heat_solution(t) for t in times]
    vmin = min(s.min() for s in solutions)
    vmax = max(s.max() for s in solutions)

    fig = plt.figure(figsize=(14, 4.5), facecolor='white')

    for i, (t, u_t) in enumerate(zip(times, solutions)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        _sphere_panel(ax, fig, X, Y, Z, u_t, f'$t = {t}$', cmap=PARULA,
                      vmin=vmin, vmax=vmax)
        print(f"  t={t}: max u = {u_t.max():.4f}, total = {np.sum(u_t*np.sin(THETA)):.4f}")

    fig.suptitle('Heat Equation on the Unit Sphere: $u_t = \\kappa \\cdot \\Delta_S u$',
                 fontsize=12, y=1.02)
    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, 'sphere_heat_conduction.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print("sphere_heat_conduction: done")
    return True

if __name__ == "__main__":
    run()
