"""Advection-diffusion in the unit ball.

Solves the advection-diffusion equation on the unit ball using spectral methods,
demonstrating how to handle the 3D Laplacian in spherical coordinates.
Translated from sphere/AdvectionDiffusion.m.

Original: https://www.chebfun.org/examples/sphere/AdvectionDiffusion.html
Author: Nicolas Boulle, July 2019
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import solve_ivp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, PARULA, _setup_3d_axes
chebfun_style()

def spherical_harmonic(l, m, theta, phi):
    """Real spherical harmonic Y_l^m(theta, phi)."""
    from scipy.special import sph_harm_y
    Ylm = sph_harm_y(l, abs(m), theta, phi)
    if m > 0:
        return np.sqrt(2) * (-1)**m * np.real(Ylm)
    elif m < 0:
        return np.sqrt(2) * (-1)**m * np.imag(Ylm)
    else:
        return np.real(Ylm)

def _sphere_panel(ax, fig, X, Y, Z, F, title, cmap=PARULA, elev=20, azim=-60,
                  vmin=None, vmax=None):
    """Render a single MATLAB-quality sphere panel with optional global color limits."""
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

    # Fine grid for smooth rendering
    n_theta, n_phi = 100, 200
    theta_1d = np.linspace(0, np.pi, n_theta)
    phi_1d = np.linspace(0, 2*np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta_1d, phi_1d, indexing='ij')

    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    # --- Initial condition on unit sphere ---
    # f(x,y,z) = Y_2^1(theta, phi) (spherical harmonic)
    F0 = spherical_harmonic(2, 1, THETA, PHI)

    # --- Solution after diffusion (analytic) ---
    kappa = 0.1  # diffusion coefficient
    t_final = 1.0
    l = 2
    decay = np.exp(-l*(l+1) * kappa * t_final)
    F_diff = decay * F0

    # --- Advection term effect ---
    omega = 0.5; alpha_adv = 0.3
    t_adv = 0.8
    THETA_adv = THETA + omega * t_adv
    PHI_adv = PHI + alpha_adv * t_adv
    F_adv = decay * spherical_harmonic(2, 1, THETA_adv, PHI_adv)

    # Use consistent color limits across all panels
    vmin = min(F0.min(), F_diff.min(), F_adv.min())
    vmax = max(F0.max(), F_diff.max(), F_adv.max())

    fig = plt.figure(figsize=(14, 4.5), facecolor='white')

    panels = [
        (F0, f'Initial: $Y_2^1(\\theta, \\phi)$'),
        (F_diff, f'After diffusion $t={t_final}$\n(decayed by {decay:.4f})'),
        (F_adv, f'With advection $t={t_adv}$\n(rotated + decayed)'),
    ]

    for i, (F, title) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        _sphere_panel(ax, fig, X, Y, Z, F, title, cmap=PARULA,
                      vmin=vmin, vmax=vmax)

    print(f"Advection-diffusion on sphere:")
    print(f"  Y_2^1 eigenvalue of Laplace-Beltrami: -l(l+1) = {-l*(l+1)}")
    print(f"  Decay factor exp(-{l*(l+1)}*{kappa}*{t_final}) = {decay:.6f}")

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, 'advection_diffusion.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print("advection_diffusion: done")
    return True

if __name__ == "__main__":
    run()
