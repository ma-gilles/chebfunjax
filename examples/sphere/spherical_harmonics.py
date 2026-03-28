"""Spherical harmonics and Spherefun.

Demonstrates spherical harmonic approximation and spherefun operations,
following sphere/SphericalHarmonics.m and sphere/AtmosphericTemperature.m.

Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, PARULA, _setup_3d_axes
chebfun_style()

from chebfunjax.spherefun.spherefun import Spherefun

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
    print("=" * 60)
    print("Spherical harmonics and Spherefun")
    print("=" * 60)

    # --- Spherefun approximation ---
    # Constant function: f = 1
    f_const = Spherefun.from_function(lambda lam, th: jnp.ones_like(lam + th))
    print(f"\nSpherefun f=1:")
    print(f"  Rank: {f_const.rank}")

    # Integral of 1 over S^2 = 4*pi
    integral_const = float(f_const.sum())
    print(f"  Integral = {integral_const:.8f}  (exact: {4*np.pi:.8f})")
    assert abs(integral_const - 4*np.pi) < 0.01

    # cos(lambda) * sin(theta): first spherical harmonic Y_1^1
    f_Y11 = Spherefun.from_function(lambda lam, th: jnp.cos(lam) * jnp.sin(th))
    print(f"\nSpherefun cos(lam)sin(th) [~= Y_1^1]:")
    print(f"  Rank: {f_Y11.rank}")

    # Integral should be 0
    integral_Y11 = float(f_Y11.sum())
    print(f"  Integral = {integral_Y11:.2e}  (expected: ~0)")
    assert abs(integral_Y11) < 0.1

    # Evaluate on equator: f(lambda=0, theta=pi/2) = cos(0)*sin(pi/2) = 1
    val_eq = float(f_Y11(jnp.array(0.0), jnp.array(float(jnp.pi)/2)))
    exact_eq = float(jnp.cos(jnp.array(0.0)) * jnp.sin(jnp.array(float(jnp.pi)/2)))
    err_eq = abs(val_eq - exact_eq)
    print(f"  f at lam=0, th=pi/2 = {val_eq:.8f}  (exact: {exact_eq:.8f})")
    assert err_eq < 1e-8

    # --- Plot: three spherical harmonics as 3D coloured spheres ---
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

    fig = plt.figure(figsize=(14, 4.5), facecolor='white')

    # Y_1^1 ~ cos(lam)*sin(th)
    F1 = np.cos(PHI) * np.sin(THETA)
    ax1 = fig.add_subplot(131, projection='3d')
    _sphere_panel(ax1, fig, X, Y, Z, F1,
                  '$Y_1^1 \\approx \\cos(\\lambda)\\sin(\\theta)$', cmap=PARULA)

    # Y_2^0 ~ (3*cos^2(th) - 1)/2
    F2 = (3 * np.cos(THETA)**2 - 1) / 2
    ax2 = fig.add_subplot(132, projection='3d')
    _sphere_panel(ax2, fig, X, Y, Z, F2,
                  '$Y_2^0 = (3\\cos^2\\theta - 1)/2$', cmap=PARULA)

    # Y_2^2 ~ sin^2(th)*cos(2*lam)
    F3 = np.sin(THETA)**2 * np.cos(2 * PHI)
    ax3 = fig.add_subplot(133, projection='3d')
    _sphere_panel(ax3, fig, X, Y, Z, F3,
                  '$Y_2^2 \\approx \\sin^2(\\theta)\\cos(2\\lambda)$', cmap=PARULA)

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, "spherical_harmonics.png"),
                dpi=150, bbox_inches="tight", facecolor='white')
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
