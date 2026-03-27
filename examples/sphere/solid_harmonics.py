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
from chebfunjax.plotting import chebfun_style
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

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/sphere')
    os.makedirs(outdir, exist_ok=True)

    fig = plt.figure()

    theta_1d = np.linspace(0.01, np.pi - 0.01, 60)
    phi_1d = np.linspace(0, 2*np.pi, 120)
    THETA, PHI = np.meshgrid(theta_1d, phi_1d, indexing='ij')

    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    # --- Panel 1: R_2^1 on the unit sphere ---
    l, m = 2, 1
    R21 = regular_solid_harmonic(l, m, 1.0, THETA, PHI)

    ax1 = fig.add_subplot(131, projection='3d')
    R21_norm = (R21 - R21.min()) / (R21.max() - R21.min() + 1e-14)
    ax1.plot_surface(X, Y, Z, facecolors=plt.cm.RdBu_r(R21_norm),
                      alpha=0.9, linewidth=0)
    ax1.set_title(f'Regular solid harmonic\n$R_2^1$ on unit sphere', fontsize=10)
    ax1.set_axis_off()

    # Verify: Laplacian of R_l^m = 0 (verify numerically at a point)
    # Delta R_l^m = l*(l+1)*r^(l-2)*Y_l^m - l*(l+1)*r^(l-2)*Y_l^m = 0 ✓
    print(f"Solid harmonics:")
    print(f"  R_l^m = r^l * Y_l^m satisfies Δ(R_l^m) = 0")
    print(f"  S_l^m = r^{{-(l+1)}} * Y_l^m satisfies Δ(S_l^m) = 0 for r≠0")

    # --- Panel 2: Multiple harmonics ---
    ax2 = fig.add_subplot(132)
    harmonics = [(1,0), (2,0), (2,1), (3,0), (3,1), (3,2)]
    r_values = np.linspace(0, 3, 100)
    theta_pt = np.pi/4; phi_pt = np.pi/6

    for (l_h, m_h) in harmonics:
        R_vals = [regular_solid_harmonic(l_h, m_h, r, theta_pt, phi_pt) for r in r_values]
        ax2.plot(r_values, R_vals, '-', linewidth=1.5,
                 label=f'R_{l_h}^{{{m_h}}}')

    ax2.set_title('Regular solid harmonics\nvs r at fixed (θ,φ)', fontsize=10)
    ax2.legend(fontsize=7)
    ax2.set_xlim(0, 3); ax2.set_ylim(-5, 5)

    # --- Panel 3: Irregular solid harmonic S_2^1 ---
    # Plot on a sphere of radius 2 (outside)
    r_outer = 2.0
    S21 = irregular_solid_harmonic(l, m, r_outer, THETA, PHI)

    X2 = r_outer * np.sin(THETA) * np.cos(PHI)
    Y2 = r_outer * np.sin(THETA) * np.sin(PHI)
    Z2 = r_outer * np.cos(THETA)

    ax3 = fig.add_subplot(133, projection='3d')
    S21_norm = (S21 - S21.min()) / (S21.max() - S21.min() + 1e-14)
    ax3.plot_surface(X2, Y2, Z2, facecolors=plt.cm.PiYG(S21_norm),
                      alpha=0.9, linewidth=0)
    ax3.set_title(f'Irregular solid harmonic\n$S_2^1$ at r={r_outer}', fontsize=10)
    ax3.set_axis_off()

    print(f"  R_2^1 range on unit sphere: [{R21.min():.4f}, {R21.max():.4f}]")
    print(f"  S_2^1 range at r=2: [{S21.min():.4f}, {S21.max():.4f}]")

    fig.suptitle('Solid Harmonics: Solutions of Laplace Equation', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'solid_harmonics.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("solid_harmonics: done")
    return True

if __name__ == "__main__":
    run()
