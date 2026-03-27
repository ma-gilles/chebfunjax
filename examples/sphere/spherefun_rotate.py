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
from chebfunjax.plotting import chebfun_style
chebfun_style()



def rotate_sphere_function(f_func, R_matrix, theta, phi):
    """Evaluate f(R^{-1} * x) on the sphere."""
    X = np.sin(theta) * np.cos(phi)
    Y = np.sin(theta) * np.sin(phi)
    Z = np.cos(theta)

    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    pts_rot = (R_matrix.T @ pts.T).T  # R^{-1} = R^T

    # Convert back to spherical
    r_vals = np.clip(pts_rot[:, 0], -1, 1)
    theta_new = np.arccos(np.clip(pts_rot[:, 2], -1, 1))
    phi_new = np.arctan2(pts_rot[:, 1], pts_rot[:, 0])

    return f_func(theta_new, phi_new).reshape(theta.shape)


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/sphere')
    os.makedirs(outdir, exist_ok=True)

    theta_1d = np.linspace(0, np.pi, 80)
    phi_1d = np.linspace(0, 2*np.pi, 160)
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

    fig = plt.figure(figsize=(15, 5))

    # --- Panel 1: Original function ---
    ax1 = fig.add_subplot(131, projection='3d')
    vmin, vmax = F_orig.min(), F_orig.max()
    norm1 = (F_orig - vmin) / (vmax - vmin + 1e-14)
    ax1.plot_surface(X, Y, Z, facecolors=plt.cm.RdBu_r(norm1),
                      alpha=0.9, linewidth=0)
    ax1.set_title('f(x,y,z) = cos(50z) + x²\nOriginal', fontsize=10)
    ax1.set_axis_off()

    # --- Panel 2: Rotated function ---
    ax2 = fig.add_subplot(132, projection='3d')
    norm2 = (F_rot - vmin) / (vmax - vmin + 1e-14)
    ax2.plot_surface(X, Y, Z, facecolors=plt.cm.RdBu_r(norm2),
                      alpha=0.9, linewidth=0)
    ax2.set_title(f'Rotated by ZYZ(π/3, π/4, π/6)', fontsize=10)
    ax2.set_axis_off()

    # --- Panel 3: Difference ---
    ax3 = fig.add_subplot(133, projection='3d')
    diff = F_rot - F_orig
    norm3 = (diff - diff.min()) / (diff.max() - diff.min() + 1e-14)
    ax3.plot_surface(X, Y, Z, facecolors=plt.cm.seismic(norm3),
                      alpha=0.9, linewidth=0)
    ax3.set_title('Difference: rotated - original', fontsize=10)
    ax3.set_axis_off()

    # Verify: L2 norm is preserved under rotation
    norm_orig = np.sqrt(np.sum(F_orig**2 * np.sin(THETA)) *
                         (theta_1d[1]-theta_1d[0]) * (phi_1d[1]-phi_1d[0]))
    norm_rot = np.sqrt(np.sum(F_rot**2 * np.sin(THETA)) *
                        (theta_1d[1]-theta_1d[0]) * (phi_1d[1]-phi_1d[0]))
    print(f"Spherical rotation:")
    print(f"  Euler angles ZYZ: (π/3, π/4, π/6)")
    print(f"  L2 norm preserved: {norm_orig:.6f} -> {norm_rot:.6f} (diff: {abs(norm_orig-norm_rot):.4e})")

    fig.suptitle('Rotating Functions on the Sphere', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'spherefun_rotate.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("spherefun_rotate: done")
    return True


if __name__ == "__main__":
    run()
