"""Helmholtz-Hodge decomposition of a vector field on the sphere.

Any tangent vector field on the sphere can be uniquely decomposed into
a divergence-free part and a curl-free part:
    V = curl(psi) + grad(phi)
Translated from sphere/HelmholtzDecomposition.m.

Original: https://www.chebfun.org/examples/sphere/HelmholtzDecomposition.html
Authors: Alex Townsend and Grady Wright, May 2016
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



def surface_curl(psi, theta, phi):
    """Compute surface curl of scalar potential psi on sphere.
    curl_S(psi) = (1/sin(theta)) * dpsi/dphi * e_theta - dpsi/dtheta * e_phi
    """
    dtheta = theta[0, 1] - theta[0, 0] if theta.ndim > 1 else theta[1] - theta[0]
    dphi = phi[1, 0] - phi[0, 0] if phi.ndim > 1 else phi[1] - phi[0]

    dpsi_dtheta = np.gradient(psi, axis=0) / dtheta if theta.ndim > 1 else np.gradient(psi, theta)
    dpsi_dphi = np.gradient(psi, axis=1) / dphi if phi.ndim > 1 else np.gradient(psi, phi)

    sin_theta = np.sin(theta) + 1e-14
    V_theta = dpsi_dphi / sin_theta
    V_phi = -dpsi_dtheta
    return V_theta, V_phi


def surface_grad(phi_pot, theta, phi):
    """Compute surface gradient of scalar potential phi on sphere."""
    dphi_val = phi[0, 1] - phi[0, 0] if phi.ndim > 1 else phi[1] - phi[0]
    dtheta_val = theta[0, 1] - theta[0, 0] if theta.ndim > 1 else theta[1] - theta[0]

    dpot_dtheta = np.gradient(phi_pot, axis=0) / dtheta_val
    dpot_dphi = np.gradient(phi_pot, axis=1) / dphi_val

    sin_theta = np.sin(theta) + 1e-14
    G_theta = dpot_dtheta
    G_phi = dpot_dphi / sin_theta
    return G_theta, G_phi


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/sphere')
    os.makedirs(outdir, exist_ok=True)

    # Grid
    n_theta, n_phi = 60, 120
    theta_1d = np.linspace(0.05, np.pi - 0.05, n_theta)
    phi_1d = np.linspace(0, 2*np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta_1d, phi_1d, indexing='ij')

    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    # Stream function psi: divergence-free component
    psi = np.sin(THETA)**2 * np.cos(2*PHI)
    # Potential phi: curl-free component
    phi_pot = np.cos(THETA) * np.sin(PHI)

    # Compute components
    Vt_div, Vp_div = surface_curl(psi, THETA, PHI)
    Vt_curl, Vp_curl = surface_grad(phi_pot, THETA, PHI)

    # Total field
    Vt = Vt_div + Vt_curl
    Vp = Vp_div + Vp_curl

    fig = plt.figure(figsize=(15, 5))

    # --- Panel 1: Divergence-free component (quiver on sphere cross-section) ---
    ax1 = fig.add_subplot(131)
    # Show theta-component at equatorial cross-section
    eq_idx = n_theta // 2
    ax1.contourf(np.degrees(phi_1d), np.degrees(theta_1d),
                  Vt_div, levels=20, cmap='RdBu_r')
    ax1.set_title('Divergence-free part\ncurl(ψ) theta-component', fontsize=10)
    ax1.set_xlabel('φ (°)'); ax1.set_ylabel('θ (°)')
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Curl-free component ---
    ax2 = fig.add_subplot(132)
    ax2.contourf(np.degrees(phi_1d), np.degrees(theta_1d),
                  Vp_curl, levels=20, cmap='PiYG')
    ax2.set_title('Curl-free part\ngrad(φ) phi-component', fontsize=10)
    ax2.set_xlabel('φ (°)'); ax2.set_ylabel('θ (°)')
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Total field on sphere ---
    ax3 = fig.add_subplot(133, projection='3d')
    field_magnitude = np.sqrt(Vt**2 + Vp**2)
    colors3 = plt.cm.viridis(field_magnitude / (field_magnitude.max() + 1e-14))
    ax3.plot_surface(X, Y, Z, facecolors=colors3, alpha=0.85, linewidth=0)
    ax3.set_axis_off()
    ax3.set_title('|V| on sphere\n(combined field)', fontsize=10)

    print("Helmholtz-Hodge decomposition on sphere:")
    print(f"  Divergence-free (stream function): psi = sin²θ·cos(2φ)")
    print(f"  Curl-free (potential): φ = cosθ·sinφ")
    print(f"  Max |V_div|: {np.max(np.abs(Vt_div)):.4f}")
    print(f"  Max |V_curl|: {np.max(np.abs(Vp_curl)):.4f}")

    fig.suptitle('Helmholtz-Hodge Decomposition on the Sphere', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'helmholtz_decomposition.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("helmholtz_decomposition: done")
    return True


if __name__ == "__main__":
    run()
