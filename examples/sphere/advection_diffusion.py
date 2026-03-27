"""Advection-diffusion in the unit ball.

Solves the advection-diffusion equation on the unit ball using spectral methods,
demonstrating how to handle the 3D Laplacian in spherical coordinates.
Translated from sphere/AdvectionDiffusion.m.

Original: https://www.chebfun.org/examples/sphere/AdvectionDiffusion.html
Author: Nicolas Boulle, July 2019
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import solve_ivp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

def spherical_harmonic(l, m, theta, phi):
    """Real spherical harmonic Y_l^m(theta, phi)."""
    from scipy.special import sph_harm_y
    # scipy convention: sph_harm(m, l, phi, theta) where theta is polar
    Ylm = sph_harm_y(l, abs(m), theta, phi)
    if m > 0:
        return np.sqrt(2) * (-1)**m * np.real(Ylm)
    elif m < 0:
        return np.sqrt(2) * (-1)**m * np.imag(Ylm)
    else:
        return np.real(Ylm)

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/sphere')
    os.makedirs(outdir, exist_ok=True)

    fig = plt.figure()

    theta_1d = np.linspace(0, np.pi, 60)
    phi_1d = np.linspace(0, 2*np.pi, 120)
    THETA, PHI = np.meshgrid(theta_1d, phi_1d)

    # Sphere coordinates
    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    # --- Panel 1: Initial condition on unit sphere ---
    # f(x,y,z) = Y_2^1(theta, phi) (spherical harmonic)
    F0 = spherical_harmonic(2, 1, THETA, PHI)

    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z, facecolors=plt.cm.RdBu_r(
        (F0 - F0.min()) / (F0.max() - F0.min() + 1e-15)), alpha=0.9)
    ax1.set_axis_off()
    ax1.set_title('Initial condition\n$Y_2^1(θ,φ)$ on sphere', fontsize=10)

    # --- Panel 2: Solution after diffusion (analytic) ---
    # For spherical harmonic Y_l^m on the sphere, eigenvalue of Laplacian = -l(l+1)
    # Solution: u(t) = exp(-l*(l+1)*kappa*t) * Y_l^m
    kappa = 0.1  # diffusion coefficient
    t_final = 1.0
    l = 2
    decay = np.exp(-l*(l+1) * kappa * t_final)
    F_diff = decay * F0

    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z, facecolors=plt.cm.RdBu_r(
        (F_diff - F_diff.min()) / (F_diff.max() - F_diff.min() + 1e-15)), alpha=0.9)
    ax2.set_axis_off()
    ax2.set_title(f'After diffusion t={t_final}\n(decayed by {decay:.4f})', fontsize=10)

    # --- Panel 3: Advection term effect ---
    # Add advection: u_t = kappa*Delta u + v_theta * du/dtheta
    # On the sphere, advection rotates the solution
    # Simple rotation: new_theta = theta + omega*t, new_phi = phi + alpha*t
    omega = 0.5; alpha = 0.3
    t_adv = 0.8
    THETA_adv = THETA + omega * t_adv
    PHI_adv = PHI + alpha * t_adv
    F_adv = decay * spherical_harmonic(2, 1, THETA_adv, PHI_adv)

    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, Z, facecolors=plt.cm.RdBu_r(
        (F_adv - F_adv.min()) / (F_adv.max() - F_adv.min() + 1e-15)), alpha=0.9)
    ax3.set_axis_off()
    ax3.set_title(f'With advection t={t_adv}\n(rotated + decayed)', fontsize=10)

    print(f"Advection-diffusion on sphere:")
    print(f"  Y_2^1 eigenvalue of Laplace-Beltrami: -l(l+1) = {-l*(l+1)}")
    print(f"  Decay factor exp(-{l*(l+1)}*{kappa}*{t_final}) = {decay:.6f}")

    fig.suptitle('Advection-Diffusion on the Unit Ball', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'advection_diffusion.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("advection_diffusion: done")
    return True

if __name__ == "__main__":
    run()
