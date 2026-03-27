"""The Laplace equation on the unit ball.

Solves Delta(u) = 0 in the unit ball with given Dirichlet boundary data
on the unit sphere, using the Poisson integral formula.
Translated from sphere/LaplaceBall.m.

Original: https://www.chebfun.org/examples/sphere/LaplaceBall.html
Author: Nick Trefethen, June 2019
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
from chebfunjax.plotting import chebfun_style
chebfun_style()



def spherical_harmonic_real(l, m, theta, phi):
    Ylm = sph_harm_y(l, abs(m), theta, phi)
    if m > 0:
        return np.sqrt(2) * (-1)**m * np.real(Ylm)
    elif m < 0:
        return np.sqrt(2) * (-1)**m * np.imag(Ylm)
    else:
        return np.real(Ylm)


def sh_coeff(f, theta, phi):
    """Compute SH coefficients up to l_max=6."""
    l_max = 6
    coeffs = {}
    dtheta = theta[1, 0] - theta[0, 0]
    dphi = phi[0, 1] - phi[0, 0]
    for l in range(l_max + 1):
        for m in range(-l, l+1):
            Y = spherical_harmonic_real(l, m, theta, phi)
            coeffs[(l, m)] = np.sum(f * Y * np.sin(theta)) * dtheta * dphi
    return coeffs


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/sphere')
    os.makedirs(outdir, exist_ok=True)

    # Boundary data on unit sphere
    n_theta, n_phi = 60, 120
    theta_1d = np.linspace(0.01, np.pi - 0.01, n_theta)
    phi_1d = np.linspace(0, 2*np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta_1d, phi_1d, indexing='ij')

    # Smooth boundary data
    np.random.seed(1)
    lam = 0.2  # characteristic wavelength
    h = np.cos(np.pi * THETA / lam) * np.cos(np.pi * PHI / lam)
    h = h * np.exp(-((THETA - np.pi/2)**2 + (PHI - np.pi)**2) / 0.5)

    # Compute SH expansion of boundary data
    coeffs = sh_coeff(h, THETA, PHI)

    # Laplace solution in ball: u(r,theta,phi) = sum_lm c_lm * r^l * Y_lm
    def laplace_solution(r, theta, phi):
        u = np.zeros_like(theta)
        for (l, m), c in coeffs.items():
            Y = spherical_harmonic_real(l, m, theta, phi)
            u += c * r**l * Y
        return u

    # Grid points in the ball
    n_r = 20
    r_1d = np.linspace(0, 1, n_r)

    X_sph = np.sin(THETA) * np.cos(PHI)
    Y_sph = np.sin(THETA) * np.sin(PHI)
    Z_sph = np.cos(THETA)

    fig = plt.figure(figsize=(15, 5))

    # --- Panel 1: Boundary data ---
    ax1 = fig.add_subplot(131, projection='3d')
    h_norm = (h - h.min()) / (h.max() - h.min() + 1e-14)
    ax1.plot_surface(X_sph, Y_sph, Z_sph, facecolors=plt.cm.RdBu_r(h_norm),
                      alpha=0.9, linewidth=0)
    ax1.set_title('Boundary data h(θ,φ)\non unit sphere', fontsize=10)
    ax1.set_axis_off()

    # --- Panel 2: Interior solution at r=0.5 ---
    r_mid = 0.5
    u_mid = laplace_solution(r_mid, THETA, PHI)

    ax2 = fig.add_subplot(132, projection='3d')
    r_m = r_mid
    X_m = r_m * np.sin(THETA) * np.cos(PHI)
    Y_m = r_m * np.sin(THETA) * np.sin(PHI)
    Z_m = r_m * np.cos(THETA)
    u_m_norm = (u_mid - h.min()) / (h.max() - h.min() + 1e-14)
    ax2.plot_surface(X_m, Y_m, Z_m, facecolors=plt.cm.RdBu_r(u_m_norm),
                      alpha=0.9, linewidth=0)
    ax2.plot_surface(X_sph, Y_sph, Z_sph, facecolors=plt.cm.RdBu_r(h_norm),
                      alpha=0.3, linewidth=0)
    ax2.set_title('Solution at r=0.5\n(interior sphere)', fontsize=10)
    ax2.set_axis_off()

    # --- Panel 3: Radial profile (verify mean value property) ---
    ax3 = fig.add_subplot(133)
    theta_pt, phi_pt = np.pi/3, np.pi/4
    r_vals = np.linspace(0, 1, 100)
    u_radial = [laplace_solution(r, np.array([[theta_pt]]), np.array([[phi_pt]]))[0,0]
                for r in r_vals]

    ax3.plot(r_vals, u_radial, 'b-', linewidth=2.5, label='u(r, θ₀, φ₀)')
    ax3.axhline(u_radial[0], color='g', linestyle='--', linewidth=1.5,
                 label='u(0) = mean of boundary data')

    # Mean value: u(0) = (1/4pi) * integral of h
    dtheta = theta_1d[1] - theta_1d[0]
    dphi_v = phi_1d[1] - phi_1d[0]
    mean_h = np.sum(h * np.sin(THETA)) * dtheta * dphi_v / (4 * np.pi)
    ax3.axhline(mean_h, color='r', linestyle=':', linewidth=2, label=f'Mean(h)={mean_h:.4f}')

    ax3.set_title('Radial profile u(r, θ₀, φ₀)\nmean value at r=0', fontsize=10)
    ax3.set_xlabel('r'); ax3.set_ylabel('u')
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    u0 = laplace_solution(0, np.array([[theta_pt]]), np.array([[phi_pt]]))[0,0]
    print(f"Laplace equation in ball:")
    print(f"  u(0) = {u0:.6f} (should equal mean of boundary data)")
    print(f"  Mean(h) = {mean_h:.6f}")
    print(f"  Difference: {abs(u0-mean_h):.4e}")

    fig.suptitle('Laplace Equation on the Unit Ball', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'laplace_ball.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("laplace_ball: done")
    return True


if __name__ == "__main__":
    run()
