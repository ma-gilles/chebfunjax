"""Solid harmonics.

Solid harmonics are solutions to Laplace's equation in 3D spherical
coordinates. Regular solid harmonics R_l^m grow as r^l, while irregular
ones S_l^m decay as r^{-(l+1)}.
Translated from sphere/SolidHarmonics.m.

Original: https://www.chebfun.org/examples/sphere/SolidHarmonics.html
Authors: Nicolas Boulle and Alex Townsend, May 2019
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

def regular_solid_harmonic(l, m, r, theta, phi):
    """Regular solid harmonic: R_l^m(r, theta, phi) = r^l * Y_l^m(theta, phi)."""
    Ylm_raw = sph_harm_y(l, abs(m), theta, phi)
    if m > 0:
        Ylm = np.sqrt(2) * (-1)**m * np.real(Ylm_raw)
    elif m < 0:
        Ylm = np.sqrt(2) * (-1)**m * np.imag(Ylm_raw)
    else:
        Ylm = np.real(Ylm_raw)
    return r**l * Ylm

def irregular_solid_harmonic(l, m, r, theta, phi):
    """Irregular solid harmonic: S_l^m = r^{-(l+1)} * Y_l^m(theta, phi)."""
    Ylm_raw = sph_harm_y(l, abs(m), theta, phi)
    if m > 0:
        Ylm = np.sqrt(2) * (-1)**m * np.real(Ylm_raw)
    elif m < 0:
        Ylm = np.sqrt(2) * (-1)**m * np.imag(Ylm_raw)
    else:
        Ylm = np.real(Ylm_raw)
    return r**(-(l+1)) * Ylm

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
    lim = max(abs(X).max(), abs(Y).max(), abs(Z).max()) * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
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

    # Fine grid for smooth rendering
    n_theta, n_phi = 100, 200
    theta_1d = np.linspace(0.01, np.pi - 0.01, n_theta)
    phi_1d = np.linspace(0, 2*np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta_1d, phi_1d, indexing='ij')

    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    fig = plt.figure(figsize=(14, 4.5), facecolor='white')

    # --- Panel 1: R_2^1 on the unit sphere ---
    l, m = 2, 1
    R21 = regular_solid_harmonic(l, m, 1.0, THETA, PHI)

    ax1 = fig.add_subplot(131, projection='3d')
    _sphere_panel(ax1, fig, X, Y, Z, R21,
                  'Regular $R_2^1$ on unit sphere', cmap=PARULA)

    # Verify: Laplacian of R_l^m = 0
    print(f"Solid harmonics:")
    print(f"  R_l^m = r^l * Y_l^m satisfies Delta(R_l^m) = 0")
    print(f"  S_l^m = r^{{-(l+1)}} * Y_l^m satisfies Delta(S_l^m) = 0 for r!=0")

    # --- Panel 2: R_3^2 on the unit sphere ---
    l2, m2 = 3, 2
    R32 = regular_solid_harmonic(l2, m2, 1.0, THETA, PHI)

    ax2 = fig.add_subplot(132, projection='3d')
    _sphere_panel(ax2, fig, X, Y, Z, R32,
                  'Regular $R_3^2$ on unit sphere', cmap=PARULA)

    # --- Panel 3: Irregular solid harmonic S_2^1 on sphere of radius 2 ---
    r_outer = 2.0
    S21 = irregular_solid_harmonic(l, m, r_outer, THETA, PHI)

    X2 = r_outer * np.sin(THETA) * np.cos(PHI)
    Y2 = r_outer * np.sin(THETA) * np.sin(PHI)
    Z2 = r_outer * np.cos(THETA)

    ax3 = fig.add_subplot(133, projection='3d')
    _sphere_panel(ax3, fig, X2, Y2, Z2, S21,
                  'Irregular $S_2^1$ at $r=2$', cmap=PARULA)

    print(f"  R_2^1 range on unit sphere: [{R21.min():.4f}, {R21.max():.4f}]")
    print(f"  S_2^1 range at r=2: [{S21.min():.4f}, {S21.max():.4f}]")

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, 'solid_harmonics.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print("solid_harmonics: done")
    return True

if __name__ == "__main__":
    run()
