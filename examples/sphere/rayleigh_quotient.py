"""Rayleigh quotient and eigenvalues on the sphere.

The Rayleigh quotient r(u) = <u, L u> / <u, u> for the Laplace-Beltrami
operator L on the sphere. The minimum over unit-norm functions is the
smallest eigenvalue -l*(l+1) = -2 (for l=1), achieved by spherical harmonics.
Translated from sphere/RayleighQuotientExample.m.

Original: https://www.chebfun.org/examples/sphere/RayleighQuotientExample.html
Author: Grady Wright, February 2017
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
    Ylm = sph_harm_y(l, abs(m), theta, phi)
    if m > 0:
        return np.sqrt(2) * (-1)**m * np.real(Ylm)
    elif m < 0:
        return np.sqrt(2) * (-1)**m * np.imag(Ylm)
    else:
        return np.real(Ylm)

def rayleigh_quotient(f_coeffs):
    """Compute Rayleigh quotient r(f) = <f, L f> / <f, f> from SH coefficients."""
    norm2 = 0.0
    Lf_dot_f = 0.0
    for (l, m), c in f_coeffs.items():
        norm2 += c**2
        Lf_dot_f += c * (-l*(l+1)) * c
    if norm2 < 1e-14:
        return 0.0
    return Lf_dot_f / norm2

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
    THETA, PHI = np.meshgrid(theta_1d, phi_1d, indexing='ij')

    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    # Pure Y_1^0 (should give R = -2)
    c_Y10 = {(1, 0): 1.0}
    rq_Y10 = rayleigh_quotient(c_Y10)

    # Pure Y_2^1 (should give R = -6)
    c_Y21 = {(2, 1): 1.0}
    rq_Y21 = rayleigh_quotient(c_Y21)

    test_functions = [
        (r'$Y_1^0$', c_Y10, rq_Y10, -1*2),
        (r'$Y_2^1$', c_Y21, rq_Y21, -2*3),
        (r'$Y_3^0$', {(3, 0): 1.0}, rayleigh_quotient({(3,0): 1.0}), -3*4),
        (r'$Y_1^0 + Y_2^0$', {(1,0): 1/np.sqrt(2), (2,0): 1/np.sqrt(2)},
         rayleigh_quotient({(1,0): 1/np.sqrt(2), (2,0): 1/np.sqrt(2)}), None),
        (r'$Y_1^0 + Y_3^0$', {(1,0): 1/np.sqrt(2), (3,0): 1/np.sqrt(2)},
         rayleigh_quotient({(1,0): 1/np.sqrt(2), (3,0): 1/np.sqrt(2)}), None),
    ]

    for name, _, rq, _ in test_functions:
        print(f"  {name}: R = {rq:.4f}")

    fig = plt.figure(figsize=(14, 4.5), facecolor='white')

    # --- Panel 1: Y_1^0 (eigenfunction with lambda = -2) ---
    F_Y10 = spherical_harmonic_real(1, 0, THETA, PHI)
    ax1 = fig.add_subplot(131, projection='3d')
    _sphere_panel(ax1, fig, X, Y, Z, F_Y10,
                  '$Y_1^0$: $R = -2$', cmap=PARULA)

    # --- Panel 2: Y_2^1 (eigenfunction with lambda = -6) ---
    F_Y21 = spherical_harmonic_real(2, 1, THETA, PHI)
    ax2 = fig.add_subplot(132, projection='3d')
    _sphere_panel(ax2, fig, X, Y, Z, F_Y21,
                  '$Y_2^1$: $R = -6$', cmap=PARULA)

    # --- Panel 3: Mixed mode Y_1^0 + Y_3^0 ---
    F_mix = (spherical_harmonic_real(1, 0, THETA, PHI) +
             spherical_harmonic_real(3, 0, THETA, PHI)) / np.sqrt(2)
    rq_mix = rayleigh_quotient({(1,0): 1/np.sqrt(2), (3,0): 1/np.sqrt(2)})
    ax3 = fig.add_subplot(133, projection='3d')
    _sphere_panel(ax3, fig, X, Y, Z, F_mix,
                  f'$Y_1^0 + Y_3^0$: $R = {rq_mix:.1f}$', cmap=PARULA)

    # Assertions
    assert abs(rq_Y10 - (-2)) < 0.01, f"Y_1^0 Rayleigh = {rq_Y10}, expected -2"
    assert abs(rq_Y21 - (-6)) < 0.01, f"Y_2^1 Rayleigh = {rq_Y21}, expected -6"
    print(f"Assertions passed: R(Y_1^0)={rq_Y10:.4f}~=-2, R(Y_2^1)={rq_Y21:.4f}~=-6")

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, 'rayleigh_quotient.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print("rayleigh_quotient: done")
    return True

if __name__ == "__main__":
    run()
