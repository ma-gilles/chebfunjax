"""The Laplace equation on the unit ball.

Solves Delta(u) = 0 in the unit ball with given Dirichlet boundary data
on the unit sphere, using the Poisson integral formula.
Translated from sphere/LaplaceBall.m.

Original: https://www.chebfun.org/examples/sphere/LaplaceBall.html
Author: Nick Trefethen, June 2019
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

def sh_coeff(f, theta, phi):
    """Compute SH coefficients up to l_max=6."""
    l_max = 6
    coeffs = {}
    dtheta = theta[1, 0] - theta[0, 0]
    dphi = phi[0, 1] - phi[0, 0]
    for l in range(l_max + 1):
        for m in range(-l, l+1):
            Y = spherical_harmonic_real(l, m, theta, phi)
            coeffs[(l, m)] = np.sum(f * Y * np.sin(theta)) * dtheta * dphi
    return coeffs

def _sphere_panel(ax, fig, X, Y, Z, F, title, cmap=PARULA, elev=20, azim=-60,
                  vmin=None, vmax=None, alpha=1.0):
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
    # Apply alpha to RGBA
    if alpha < 1.0:
        fcolors[..., 3] = alpha
    ax.plot_surface(X, Y, Z, facecolors=fcolors,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/sphere')
    os.makedirs(outdir, exist_ok=True)

    # Fine grid on unit sphere
    n_theta, n_phi = 100, 200
    theta_1d = np.linspace(0.01, np.pi - 0.01, n_theta)
    phi_1d = np.linspace(0, 2*np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta_1d, phi_1d, indexing='ij')

    # Smooth boundary data
    np.random.seed(1)
    lam = 0.2
    h = np.cos(np.pi * THETA / lam) * np.cos(np.pi * PHI / lam)
    h = h * np.exp(-((THETA - np.pi/2)**2 + (PHI - np.pi)**2) / 0.5)

    # Compute SH expansion of boundary data
    coeffs = sh_coeff(h, THETA, PHI)

    def laplace_solution(r, theta, phi):
        u = np.zeros_like(theta)
        for (l, m), c in coeffs.items():
            Y = spherical_harmonic_real(l, m, theta, phi)
            u += c * r**l * Y
        return u

    X_sph = np.sin(THETA) * np.cos(PHI)
    Y_sph = np.sin(THETA) * np.sin(PHI)
    Z_sph = np.cos(THETA)

    # Consistent colour limits
    h_vmin, h_vmax = float(h.min()), float(h.max())

    r_mid = 0.5
    u_mid = laplace_solution(r_mid, THETA, PHI)
    X_m = r_mid * X_sph
    Y_m = r_mid * Y_sph
    Z_m = r_mid * Z_sph

    # Residual
    u_boundary = laplace_solution(1.0, THETA, PHI)
    residual = h - u_boundary

    fig = plt.figure(figsize=(14, 4.5), facecolor='white')

    # --- Panel 1: Boundary data h(theta, phi) ---
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.view_init(elev=20, azim=-60)
    ax1.set_facecolor("white")
    _sphere_panel(ax1, fig, X_sph, Y_sph, Z_sph, h,
                  'Boundary data $h(\\theta,\\phi)$', cmap=PARULA,
                  vmin=h_vmin, vmax=h_vmax)
    ax1.set_xlim(-1.05, 1.05)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_zlim(-1.05, 1.05)
    ax1.set_axis_off()
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    # --- Panel 2: Interior + boundary (nested spheres) ---
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.view_init(elev=20, azim=-60)
    ax2.set_facecolor("white")
    # Inner sphere: solution at r=0.5
    _sphere_panel(ax2, fig, X_m, Y_m, Z_m, u_mid,
                  f'Solution at $r={r_mid}$ + boundary', cmap=PARULA,
                  vmin=h_vmin, vmax=h_vmax, alpha=1.0)
    # Outer sphere: boundary (transparent)
    _sphere_panel(ax2, fig, X_sph, Y_sph, Z_sph, h,
                  '', cmap=PARULA,
                  vmin=h_vmin, vmax=h_vmax, alpha=0.2)
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_zlim(-1.05, 1.05)
    ax2.set_axis_off()
    ax2.set_title(f'Solution at $r={r_mid}$ + boundary', fontsize=10, pad=2)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False

    # --- Panel 3: SH approximation residual on boundary ---
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.view_init(elev=20, azim=-60)
    ax3.set_facecolor("white")
    fmin3, fmax3 = float(residual.min()), float(residual.max())
    if fmax3 > fmin3:
        norm3 = (residual - fmin3) / (fmax3 - fmin3)
    else:
        norm3 = np.full_like(residual, 0.5)
    fcolors3 = PARULA(norm3)
    ax3.plot_surface(X_sph, Y_sph, Z_sph, facecolors=fcolors3,
                     rstride=1, cstride=1,
                     linewidth=0, antialiased=True, shade=False)
    ax3.set_xlim(-1.05, 1.05)
    ax3.set_ylim(-1.05, 1.05)
    ax3.set_zlim(-1.05, 1.05)
    ax3.set_axis_off()
    ax3.set_title('SH approximation residual\n$h - u(1,\\theta,\\phi)$', fontsize=10, pad=2)
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False

    # Mean value property check
    theta_pt, phi_pt = np.pi/3, np.pi/4
    dtheta = theta_1d[1] - theta_1d[0]
    dphi_v = phi_1d[1] - phi_1d[0]
    mean_h = np.sum(h * np.sin(THETA)) * dtheta * dphi_v / (4 * np.pi)
    u0 = laplace_solution(0, np.array([[theta_pt]]), np.array([[phi_pt]]))[0,0]

    print(f"Laplace equation in ball:")
    print(f"  u(0) = {u0:.6f} (should equal mean of boundary data)")
    print(f"  Mean(h) = {mean_h:.6f}")
    print(f"  Difference: {abs(u0-mean_h):.4e}")

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, 'laplace_ball.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print("laplace_ball: done")
    return True

if __name__ == "__main__":
    run()
