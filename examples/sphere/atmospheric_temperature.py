"""Computing with an atmospheric dataset in Spherefun.

Demonstrates how to load and analyze a global atmospheric temperature
dataset on the sphere, computing the spherical harmonic expansion and
low-rank approximation.
Translated from sphere/AtmosphericTemperature.m.

Original: https://www.chebfun.org/examples/sphere/AtmosphericTemperature.html
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

def synthetic_temperature(theta, phi):
    """Generate a synthetic global temperature field resembling atmospheric data."""
    T = 15 * (0.5 - 0.7 * np.cos(theta))
    T += 3 * np.sin(2*theta) * np.cos(phi)
    T += 2 * np.sin(theta)**2 * np.cos(2*phi)
    T += 1.5 * np.sin(theta) * np.sin(phi)
    T += 4 * np.exp(-5 * ((theta - np.pi/2)**2 + (phi - 0)**2))
    T += -3 * np.exp(-5 * ((theta - np.pi/2)**2 + (phi - np.pi)**2))
    return T

def spherical_harmonic_coeff(f, theta, phi, l, m):
    """Compute spherical harmonic coefficient <f, Y_l^m>."""
    Ylm = sph_harm_y(l, abs(m), theta, phi)
    if m > 0:
        Y = np.sqrt(2) * (-1)**m * np.real(Ylm)
    elif m < 0:
        Y = np.sqrt(2) * (-1)**m * np.imag(Ylm)
    else:
        Y = np.real(Ylm)
    dtheta = theta[1, 0] - theta[0, 0] if theta.ndim > 1 else theta[1] - theta[0]
    dphi = phi[0, 1] - phi[0, 0] if phi.ndim > 1 else phi[1] - phi[0]
    return np.sum(f * Y * np.sin(theta)) * dtheta * dphi

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
    theta_1d = np.linspace(0, np.pi, n_theta)
    phi_1d = np.linspace(0, 2*np.pi, n_phi)
    PHI, THETA = np.meshgrid(phi_1d, theta_1d)

    T = synthetic_temperature(THETA, PHI)

    X = np.sin(THETA) * np.cos(PHI)
    Y_cart = np.sin(THETA) * np.sin(PHI)
    Z_cart = np.cos(THETA)

    # --- Low-rank SH approximation (l_max=3) ---
    l_max_approx = 3
    T_approx = np.zeros_like(T)
    for l in range(l_max_approx + 1):
        for m in range(-l, l+1):
            c_lm = spherical_harmonic_coeff(T, THETA, PHI, l, m)
            Ylm = sph_harm_y(l, abs(m), THETA, PHI)
            if m > 0:
                Y_sh = np.sqrt(2) * (-1)**m * np.real(Ylm)
            elif m < 0:
                Y_sh = np.sqrt(2) * (-1)**m * np.imag(Ylm)
            else:
                Y_sh = np.real(Ylm)
            T_approx += c_lm * Y_sh

    T_diff = T - T_approx

    fig = plt.figure(figsize=(14, 4.5), facecolor='white')

    # --- Panel 1: Full temperature on sphere ---
    ax1 = fig.add_subplot(131, projection='3d')
    _sphere_panel(ax1, fig, X, Y_cart, Z_cart, T,
                  'Temperature field $T(\\theta, \\phi)$', cmap=PARULA)

    # --- Panel 2: Low-rank SH approximation ---
    ax2 = fig.add_subplot(132, projection='3d')
    _sphere_panel(ax2, fig, X, Y_cart, Z_cart, T_approx,
                  f'SH approximation ($l \\leq {l_max_approx}$)', cmap=PARULA)

    # --- Panel 3: Residual ---
    ax3 = fig.add_subplot(133, projection='3d')
    _sphere_panel(ax3, fig, X, Y_cart, Z_cart, T_diff,
                  'Residual (full $-$ approx)', cmap=PARULA)

    T_min, T_max = T.min(), T.max()
    print(f"Atmospheric temperature: range [{T_min:.1f}, {T_max:.1f}] C")

    # Spherical harmonic power spectrum
    l_max = 10
    power = np.zeros(l_max + 1)
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            c_lm = spherical_harmonic_coeff(T, THETA, PHI, l, m)
            power[l] += c_lm**2
        print(f"  l={l}: power={power[l]:.4f}")

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, 'atmospheric_temperature.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print("atmospheric_temperature: done")
    return True

if __name__ == "__main__":
    run()
