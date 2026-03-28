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
from chebfunjax.plotting import chebfun_style, PARULA, _setup_3d_axes
chebfun_style()

def surface_curl(psi, theta, phi, theta_1d, phi_1d):
    """Compute surface curl of scalar potential psi on sphere.
    curl_S(psi) = (1/sin(theta)) * dpsi/dphi * e_theta - dpsi/dtheta * e_phi
    Grid uses indexing='ij': axis 0 = theta, axis 1 = phi.
    """
    dtheta = theta_1d[1] - theta_1d[0]
    dphi = phi_1d[1] - phi_1d[0]

    dpsi_dtheta = np.gradient(psi, dtheta, axis=0)
    dpsi_dphi = np.gradient(psi, dphi, axis=1)

    sin_theta = np.sin(theta) + 1e-14
    V_theta = dpsi_dphi / sin_theta
    V_phi = -dpsi_dtheta
    return V_theta, V_phi

def surface_grad(phi_pot, theta, phi, theta_1d, phi_1d):
    """Compute surface gradient of scalar potential phi on sphere.
    Grid uses indexing='ij': axis 0 = theta, axis 1 = phi.
    """
    dtheta = theta_1d[1] - theta_1d[0]
    dphi = phi_1d[1] - phi_1d[0]

    dpot_dtheta = np.gradient(phi_pot, dtheta, axis=0)
    dpot_dphi = np.gradient(phi_pot, dphi, axis=1)

    sin_theta = np.sin(theta) + 1e-14
    G_theta = dpot_dtheta
    G_phi = dpot_dphi / sin_theta
    return G_theta, G_phi

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

    # Fine grid for smooth 3D rendering
    n_theta, n_phi = 100, 200
    theta_1d = np.linspace(0.02, np.pi - 0.02, n_theta)
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
    Vt_div, Vp_div = surface_curl(psi, THETA, PHI, theta_1d, phi_1d)
    Vt_curl, Vp_curl = surface_grad(phi_pot, THETA, PHI, theta_1d, phi_1d)

    # Total field
    Vt = Vt_div + Vt_curl
    Vp = Vp_div + Vp_curl

    # Compute magnitudes for sphere colouring
    mag_div = np.sqrt(Vt_div**2 + Vp_div**2)
    mag_curl = np.sqrt(Vt_curl**2 + Vp_curl**2)
    mag_total = np.sqrt(Vt**2 + Vp**2)

    fig = plt.figure(figsize=(14, 4.5), facecolor='white')

    # --- Panel 1: Divergence-free part (stream function) on sphere ---
    ax1 = fig.add_subplot(131, projection='3d')
    _sphere_panel(ax1, fig, X, Y, Z, mag_div,
                  'Divergence-free $|\\mathrm{curl}(\\psi)|$', cmap=PARULA)

    # --- Panel 2: Curl-free part (gradient) on sphere ---
    ax2 = fig.add_subplot(132, projection='3d')
    _sphere_panel(ax2, fig, X, Y, Z, mag_curl,
                  'Curl-free $|\\nabla\\phi|$', cmap=PARULA)

    # --- Panel 3: Total field on sphere ---
    ax3 = fig.add_subplot(133, projection='3d')
    _sphere_panel(ax3, fig, X, Y, Z, mag_total,
                  'Total $|\\mathbf{V}|$ on sphere', cmap=PARULA)

    print("Helmholtz-Hodge decomposition on sphere:")
    print(f"  Divergence-free (stream function): psi = sin^2(th)*cos(2*phi)")
    print(f"  Curl-free (potential): phi = cos(th)*sin(phi)")
    print(f"  Max |V_div|: {np.max(mag_div):.4f}")
    print(f"  Max |V_curl|: {np.max(mag_curl):.4f}")

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, 'helmholtz_decomposition.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print("helmholtz_decomposition: done")
    return True

if __name__ == "__main__":
    run()
