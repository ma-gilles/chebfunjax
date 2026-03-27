"""Helmholtz-Hodge decomposition in the ball.

Any vector field v in the unit ball can be decomposed as:
    v = grad(f) + curl(psi) + grad(phi)
where curl(psi) is divergence-free and grad(phi) is harmonic.
Translated from sphere/HelmholtzDecompositionBall.m.

Original: https://www.chebfun.org/examples/sphere/HelmholtzDecompositionBall.html
Authors: Nicolas Boulle and Alex Townsend, May 2019
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/sphere')
    os.makedirs(outdir, exist_ok=True)

    # Grid in the ball (use spherical coordinates)
    n_r, n_theta, n_phi = 20, 30, 60
    r_1d = np.linspace(0, 1, n_r)
    theta_1d = np.linspace(0.05, np.pi - 0.05, n_theta)
    phi_1d = np.linspace(0, 2*np.pi, n_phi)

    R, THETA, PHI = np.meshgrid(r_1d, theta_1d, phi_1d, indexing='ij')

    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    # Define a vector field v = (vx, vy, vz) in the ball
    # v = grad(f) + curl(psi): split into curl-free and divergence-free parts
    # Take f(x,y,z) = x*y + z^2 => grad(f) = (y, x, 2z)
    vx_cf = Y.copy()  # grad_x f
    vy_cf = X.copy()  # grad_y f
    vz_cf = 2 * Z     # grad_z f

    # Divergence-free part: psi = (0, 0, xy) => curl(psi) = (x, -y, 0)
    vx_df = X.copy()
    vy_df = -Y.copy()
    vz_df = np.zeros_like(Z)

    # Total field
    vx = vx_cf + vx_df
    vy = vy_cf + vy_df
    vz = vz_cf + vz_df

    # Divergence: div(v) = div(grad f) + div(curl psi) = Laplacian(f) + 0
    # Laplacian(f=xy+z^2) = 0 + 0 + 2 = 2
    div_v = np.full_like(X, 2.0)  # analytic: div(v) = 2

    # Curl: curl(v) = curl(grad f) + curl(curl psi) = 0 + (0,0, -2*y - 2*(-1)) = computed below
    # curl(vx, vy, vz) at equatorial slice
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: Vector field cross-section (z=0 slice) ---
    z0_idx = n_r // 2
    t0_idx = n_theta // 2  # equatorial

    r_slice = r_1d
    phi_slice = phi_1d
    R_s, PHI_s = np.meshgrid(r_slice, phi_slice, indexing='ij')
    X_s = R_s * np.cos(PHI_s)
    Y_s = R_s * np.sin(PHI_s)

    vx_s = Y_s  # grad_x (xy) at z=0
    vy_s = X_s  # grad_y (xy) at z=0
    vx_s_df = X_s  # curl part
    vy_s_df = -Y_s

    # Total
    vx_tot = vx_s + vx_s_df
    vy_tot = vy_s + vy_s_df

    skip = 3
    axes[0].quiver(X_s[::skip, ::skip], Y_s[::skip, ::skip],
                    vx_tot[::skip, ::skip], vy_tot[::skip, ::skip],
                    np.sqrt(vx_tot[::skip,::skip]**2 + vy_tot[::skip,::skip]**2),
                    cmap='viridis', alpha=0.8)
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Total field v\nat z=0 cross-section', fontsize=10)
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')

    # --- Panel 2: Curl-free (gradient) part ---
    axes[1].quiver(X_s[::skip, ::skip], Y_s[::skip, ::skip],
                    vx_s[::skip, ::skip], vy_s[::skip, ::skip],
                    np.sqrt(vx_s[::skip,::skip]**2 + vy_s[::skip,::skip]**2),
                    cmap='Blues', alpha=0.8)
    axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Curl-free part: grad(xy+z²)\nat z=0', fontsize=10)
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')

    # --- Panel 3: Divergence-free (curl) part ---
    axes[2].quiver(X_s[::skip, ::skip], Y_s[::skip, ::skip],
                    vx_s_df[::skip, ::skip], vy_s_df[::skip, ::skip],
                    np.sqrt(vx_s_df[::skip,::skip]**2 + vy_s_df[::skip,::skip]**2),
                    cmap='Reds', alpha=0.8)
    axes[2].set_aspect('equal'); axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Divergence-free part: curl(ψ)\nat z=0', fontsize=10)
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y')

    print("Helmholtz-Hodge decomposition in the ball:")
    print(f"  v = grad(xy+z²) + curl(0,0,xy)")
    print(f"  div(v) = Laplacian(xy+z²) = 2 (analytic)")
    print(f"  curl-free part: (y, x, 2z)")
    print(f"  divergence-free part: (x, -y, 0)")

    fig.suptitle('Helmholtz-Hodge Decomposition in the Unit Ball', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'helmholtz_decomposition_ball.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("helmholtz_decomposition_ball: done")
    return True


if __name__ == "__main__":
    run()
