"""Rayleigh quotient and eigenvalues on the sphere.

The Rayleigh quotient r(u) = <u, L u> / <u, u> for the Laplace-Beltrami
operator L on the sphere. The minimum over unit-norm functions is the
smallest eigenvalue -l*(l+1) = -2 (for l=1), achieved by spherical harmonics.
Translated from sphere/RayleighQuotientExample.m.

Original: https://www.chebfun.org/examples/sphere/RayleighQuotientExample.html
Author: Grady Wright, February 2017
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


def laplace_beltrami_sh_coefficient(f_coeffs, l, m):
    """Laplace-Beltrami acting on SH: Delta_S Y_l^m = -l(l+1) Y_l^m."""
    return -l * (l + 1) * f_coeffs.get((l, m), 0.0)


def rayleigh_quotient(f_coeffs):
    """Compute Rayleigh quotient r(f) = <f, L f> / <f, f> from SH coefficients."""
    norm2 = 0.0
    Lf_dot_f = 0.0
    for (l, m), c in f_coeffs.items():
        norm2 += c**2
        Lf_dot_f += c * (-l*(l+1)) * c  # = -l(l+1)|c|^2
    if norm2 < 1e-14:
        return 0.0
    return Lf_dot_f / norm2


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/sphere')
    os.makedirs(outdir, exist_ok=True)

    # Grid
    n_theta, n_phi = 60, 120
    theta_1d = np.linspace(0.01, np.pi - 0.01, n_theta)
    phi_1d = np.linspace(0, 2*np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta_1d, phi_1d, indexing='ij')

    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    fig, axes = plt.subplots(1, 3)

    # --- Panel 1: Eigenvalues vs l ---
    l_vals = np.arange(0, 15)
    eigenvals = [-l*(l+1) for l in l_vals]
    multiplicities = [2*l+1 for l in l_vals]

    ax_twin = axes[0].twinx()
    axes[0].plot(l_vals, eigenvals, 'b.-', markersize=10, linewidth=2,
                 label='λ_l = -l(l+1)')
    ax_twin.bar(l_vals, multiplicities, alpha=0.3, color='orange',
                label='Multiplicity 2l+1')
    axes[0].set_title('Eigenvalues of Laplace-Beltrami\non unit sphere', fontsize=10)
    axes[0].set_xlabel('Degree l'); axes[0].set_ylabel('Eigenvalue λ_l = -l(l+1)', color='b')
    ax_twin.set_ylabel('Multiplicity', color='orange')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    print(f"Eigenvalues: {eigenvals[:6]}")

    # --- Panel 2: Rayleigh quotient for various functions ---
    np.random.seed(42)
    # Test functions as combinations of SH
    test_cases = []

    # Pure Y_1^0 (should give R = -2)
    c_Y10 = {(1, 0): 1.0}
    rq_Y10 = rayleigh_quotient(c_Y10)

    # Pure Y_2^1 (should give R = -6)
    c_Y21 = {(2, 1): 1.0}
    rq_Y21 = rayleigh_quotient(c_Y21)

    # Mix of modes
    test_functions = [
        (r'$Y_1^0$', c_Y10, rq_Y10, -1*2),
        (r'$Y_2^1$', c_Y21, rq_Y21, -2*3),
        (r'$Y_3^0$', {(3, 0): 1.0}, rayleigh_quotient({(3,0): 1.0}), -3*4),
        (r'$Y_1^0 + Y_2^0$', {(1,0): 1/np.sqrt(2), (2,0): 1/np.sqrt(2)},
         rayleigh_quotient({(1,0): 1/np.sqrt(2), (2,0): 1/np.sqrt(2)}), None),
        (r'$Y_1^0 + Y_3^0$', {(1,0): 1/np.sqrt(2), (3,0): 1/np.sqrt(2)},
         rayleigh_quotient({(1,0): 1/np.sqrt(2), (3,0): 1/np.sqrt(2)}), None),
    ]

    names = [t[0] for t in test_functions]
    rqs = [t[2] for t in test_functions]
    exact = [t[3] for t in test_functions]

    axes[1].bar(range(len(rqs)), rqs, color=['b','g','r','m','c'], alpha=0.7)
    for i, (ex, rq) in enumerate(zip(exact, rqs)):
        if ex is not None:
            axes[1].axhline(ex, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, fontsize=9)
    axes[1].set_title('Rayleigh quotients\nfor various test functions', fontsize=10)
    axes[1].set_ylabel('R(f) = <f, Lf>/<f,f>'); axes[1].grid(True, alpha=0.3, axis='y')
    for name, rq in zip(names, rqs):
        print(f"  {name}: R = {rq:.4f}")

    # --- Panel 3: Rayleigh quotient minimization ---
    # Show that minimum over random functions with l_max=1 is -2
    np.random.seed(123)
    n_random = 200
    rq_random = []
    for _ in range(n_random):
        # Random function with l=0 and l=1 components
        c = {}
        for l in range(3):
            for m in range(-l, l+1):
                c[(l, m)] = np.random.randn()
        # Normalize
        norm2 = sum(v**2 for v in c.values())
        c = {k: v/np.sqrt(norm2) for k, v in c.items()}
        rq_random.append(rayleigh_quotient(c))

    axes[2].hist(rq_random, bins=20, color='steelblue', alpha=0.7,
                 edgecolor='black', linewidth=0.5)
    axes[2].axvline(-2, color='r', linestyle='--', linewidth=2,
                     label='λ_1 = -2')
    axes[2].axvline(0, color='g', linestyle='--', linewidth=1.5,
                     label='λ_0 = 0')
    axes[2].set_title('Rayleigh quotient for\nrandom l≤2 functions', fontsize=10)
    axes[2].set_xlabel('R(f)'); axes[2].set_ylabel('Count')
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3, axis='y')

    fig.suptitle('Rayleigh Quotient and Eigenvalues on the Sphere', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'rayleigh_quotient.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Assertions
    assert abs(rq_Y10 - (-2)) < 0.01, f"Y_1^0 Rayleigh = {rq_Y10}, expected -2"
    assert abs(rq_Y21 - (-6)) < 0.01, f"Y_2^1 Rayleigh = {rq_Y21}, expected -6"
    print(f"Assertions passed: R(Y_1^0)={rq_Y10:.4f}≈-2, R(Y_2^1)={rq_Y21:.4f}≈-6")

    print("rayleigh_quotient: done")
    return True


if __name__ == "__main__":
    run()
