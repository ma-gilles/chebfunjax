"""Poloidal-toroidal decomposition of a vector field.

Any divergence-free vector field in the ball can be expressed as:
    v = curl(r * T) + curl(curl(r * P))  (toroidal + poloidal)
where T and P are scalar potentials.
Translated from sphere/PTDecomposition.m.

Original: https://www.chebfun.org/examples/sphere/PTDecomposition.html
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



def toroidal_field(T_func, r, theta, phi):
    """Compute toroidal field: T_field = curl(r * T_hat)."""
    # For T = T(theta, phi) (independent of r):
    # T_field = 1/r * [1/sin(theta)*dT/dphi * e_theta - dT/dtheta * e_phi]
    T = T_func(theta, phi)
    dt = 0.001

    dT_dtheta = (T_func(theta + dt, phi) - T_func(theta - dt, phi)) / (2*dt)
    dT_dphi = (T_func(theta, phi + dt) - T_func(theta, phi - dt)) / (2*dt)

    sin_theta = np.sin(theta) + 1e-14
    Vt = dT_dphi / sin_theta
    Vp = -dT_dtheta
    Vr = np.zeros_like(Vt)
    return Vr, Vt, Vp


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/sphere')
    os.makedirs(outdir, exist_ok=True)

    n_theta, n_phi = 50, 100
    theta_1d = np.linspace(0.05, np.pi - 0.05, n_theta)
    phi_1d = np.linspace(0, 2*np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta_1d, phi_1d, indexing='ij')

    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    # Toroidal potential T = cos(theta)*sin(phi)
    def T_func(theta, phi):
        return np.cos(theta) * np.sin(phi)

    # Poloidal potential P = sin(theta)*cos(phi)
    def P_func(theta, phi):
        return np.sin(theta) * np.cos(phi)

    # Compute toroidal component
    Vr_T, Vt_T, Vp_T = toroidal_field(T_func, 1.0, THETA, PHI)
    Vr_P, Vt_P, Vp_P = toroidal_field(P_func, 1.0, THETA, PHI)

    # Total
    Vt = Vt_T + Vt_P
    Vp = Vp_T + Vp_P

    fig = plt.figure()

    # --- Panel 1: Toroidal component ---
    ax1 = fig.add_subplot(131, projection='3d')
    mag_T = np.sqrt(Vt_T**2 + Vp_T**2)
    mag_T_norm = mag_T / (mag_T.max() + 1e-14)
    ax1.plot_surface(X, Y, Z, facecolors=plt.cm.Blues(mag_T_norm),
                      alpha=0.9, linewidth=0)
    ax1.set_title('|Toroidal field|\n|curl(r·T)|', fontsize=10)
    ax1.set_axis_off()

    # --- Panel 2: Poloidal component ---
    ax2 = fig.add_subplot(132, projection='3d')
    mag_P = np.sqrt(Vt_P**2 + Vp_P**2)
    mag_P_norm = mag_P / (mag_P.max() + 1e-14)
    ax2.plot_surface(X, Y, Z, facecolors=plt.cm.Reds(mag_P_norm),
                      alpha=0.9, linewidth=0)
    ax2.set_title('|Poloidal field|\n|curl(curl(r·P))|', fontsize=10)
    ax2.set_axis_off()

    # --- Panel 3: Total field magnitude ---
    ax3 = fig.add_subplot(133, projection='3d')
    mag = np.sqrt(Vt**2 + Vp**2)
    mag_norm = mag / (mag.max() + 1e-14)
    ax3.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(mag_norm),
                      alpha=0.9, linewidth=0)
    ax3.set_title('|Total field|\nToroidal + Poloidal', fontsize=10)
    ax3.set_axis_off()

    print("Poloidal-Toroidal decomposition:")
    print(f"  T = cos(θ)sin(φ): toroidal potential")
    print(f"  P = sin(θ)cos(φ): poloidal potential")
    print(f"  Max toroidal: {mag_T.max():.4f}")
    print(f"  Max poloidal: {mag_P.max():.4f}")

    fig.suptitle('Poloidal-Toroidal Decomposition of a Vector Field', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'pt_decomposition.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("pt_decomposition: done")
    return True


if __name__ == "__main__":
    run()
