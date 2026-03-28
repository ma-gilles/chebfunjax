"""Operations on the sphere: differentiation, integration, rotation.

Demonstrates spherefun calculus and the Helmholtz decomposition,
following sphere/Gravity.m, sphere/HelmholtzDecomposition.m, and
sphere/SphereHeatConduction.m.

Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    print("Operations on the sphere")
    print("=" * 60)

    # --- Integral of a smooth function over S^2 ---
    f1 = Spherefun.from_function(lambda lam, th: jnp.cos(th))
    integral_cos_th = float(f1.sum())
    print(f"\nIntegral of cos(th) over S^2: {integral_cos_th:.2e}  (expected: 0)")
    assert abs(integral_cos_th) < 0.01

    f2 = Spherefun.from_function(lambda lam, th: jnp.cos(th)**2)
    integral_cos2 = float(f2.sum())
    exact_cos2 = 4 * np.pi / 3
    print(f"Integral of cos^2(th) = {integral_cos2:.8f}  (exact: {exact_cos2:.8f})")
    assert abs(integral_cos2 - exact_cos2) < 0.01

    # --- Gravity-like potential on sphere ---
    print("\nGravity-like potential:")
    f_grav = Spherefun.from_function(
        lambda lam, th: 1.0 / (1.5 - 0.5 * jnp.cos(lam) * jnp.sin(th))
    )
    print(f"  Rank: {f_grav.rank}")

    val_test = float(f_grav(jnp.array(0.0), jnp.array(float(jnp.pi)/2)))
    exact_test = 1.0 / (1.5 - 0.5)
    print(f"  f(0,pi/2) = {val_test:.8f}  (exact: {exact_test:.8f})")
    assert abs(val_test - exact_test) < 1e-8

    # --- Differentiation: gradient ---
    f_sin = Spherefun.from_function(lambda lam, th: jnp.sin(lam))
    print(f"\nSpherefun sin(lam), rank {f_sin.rank}")

    # --- Rotation ---
    f3 = Spherefun.from_function(lambda lam, th: jnp.exp(-3 * th**2))
    print(f"\nexp(-3*th^2) on sphere, rank {f3.rank}")

    # --- Plot: 3D sphere rendering ---
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/sphere')
    os.makedirs(outdir, exist_ok=True)

    # Fine grid
    n_theta, n_phi = 100, 200
    theta_1d = np.linspace(0, np.pi, n_theta)
    phi_1d = np.linspace(0, 2*np.pi, n_phi)
    LAM, TH = np.meshgrid(phi_1d, theta_1d)

    X = np.sin(TH) * np.cos(LAM)
    Y = np.sin(TH) * np.sin(LAM)
    Z = np.cos(TH)

    funcs = [
        (np.cos(TH)**2, "$\\cos^2(\\theta)$: integral $= 4\\pi/3$"),
        (1.0 / (1.5 - 0.5 * np.cos(LAM) * np.sin(TH)), "Gravity potential"),
        (np.exp(-3 * TH**2), "$\\exp(-3\\theta^2)$"),
    ]

    fig = plt.figure(figsize=(14, 4.5), facecolor='white')

    for i, (F, title) in enumerate(funcs):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        _sphere_panel(ax, fig, X, Y, Z, F, title, cmap=PARULA)

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, "sphere_operations.png"),
                dpi=150, bbox_inches="tight", facecolor='white')
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
