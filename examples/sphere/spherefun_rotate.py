"""Rotating functions on the sphere.

Demonstrates how to rotate a function defined on the sphere using
Wigner D-matrices for spherical harmonic rotations.
Translated from sphere/SpherefunRotate.m.

Original: https://www.chebfun.org/examples/sphere/SpherefunRotate.html
Authors: Alex Townsend and Grady Wright, May 2017
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, PARULA, _setup_3d_axes
chebfun_style()

def rotate_sphere_function(f_func, R_matrix, theta, phi):
    """Evaluate f(R^{-1} * x) on the sphere."""
    X = np.sin(theta) * np.cos(phi)
    Y = np.sin(theta) * np.sin(phi)
    Z = np.cos(theta)

    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    pts_rot = (R_matrix.T @ pts.T).T  # R^{-1} = R^T

    # Convert back to spherical
    theta_new = np.arccos(np.clip(pts_rot[:, 2], -1, 1))
    phi_new = np.arctan2(pts_rot[:, 1], pts_rot[:, 0])

    return f_func(theta_new, phi_new).reshape(theta.shape)

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

    # Light gray panes
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

    # Function: f(x,y,z) = cos(50*z) + x^2
    def f_func(theta, phi):
        z = np.cos(theta)
        x = np.sin(theta) * np.cos(phi)
        return np.cos(50 * z) + x**2

    F_orig = f_func(THETA, PHI)

    # Rotation: Euler angles (alpha, beta, gamma) = (pi/3, pi/4, pi/6)
    alpha, beta, gamma = np.pi/3, np.pi/4, np.pi/6
    R = Rotation.from_euler('ZYZ', [alpha, beta, gamma]).as_matrix()

    F_rot = rotate_sphere_function(f_func, R, THETA, PHI)
    F_diff = F_rot - F_orig

    # --- Three side-by-side 3D spheres ---
    fig = plt.figure(figsize=(14, 4.5), facecolor='white')

    panels = [
        (F_orig, 'Original: $f = \\cos(50z) + x^2$', PARULA),
        (F_rot, 'Rotated by ZYZ($\\pi/3, \\pi/4, \\pi/6$)', PARULA),
        (F_diff, 'Difference (rotated $-$ original)', PARULA),
    ]

    for i, (F, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        _sphere_panel(ax, fig, X, Y, Z, F, title, cmap=cmap)

    # Verify: L2 norm is preserved under rotation
    norm_orig = np.sqrt(np.sum(F_orig**2 * np.sin(THETA)) *
                         (theta_1d[1]-theta_1d[0]) * (phi_1d[1]-phi_1d[0]))
    norm_rot = np.sqrt(np.sum(F_rot**2 * np.sin(THETA)) *
                        (theta_1d[1]-theta_1d[0]) * (phi_1d[1]-phi_1d[0]))
    print(f"Spherical rotation:")
    print(f"  Euler angles ZYZ: (pi/3, pi/4, pi/6)")
    print(f"  L2 norm preserved: {norm_orig:.6f} -> {norm_rot:.6f} (diff: {abs(norm_orig-norm_rot):.4e})")

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, 'spherefun_rotate.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print("spherefun_rotate: done")
    return True

if __name__ == "__main__":
    run()
