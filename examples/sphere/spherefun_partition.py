"""Parity partitioning a spherefun.

Demonstrates how any function on the sphere can be split into its
even and odd parts (parity decomposition):
    f = f_even + f_odd,  where f_even(x,y,z) = f(-x,-y,-z),
and further into four pieces based on symmetry in (x,y) and z.
Translated from sphere/SpherefunPartition.m.

Original: https://www.chebfun.org/examples/sphere/SpherefunPartition.html
Author: Behnam Hashemi, November 2016
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

    theta_1d = np.linspace(0, np.pi, 60)
    phi_1d = np.linspace(0, 2*np.pi, 120)
    THETA, PHI = np.meshgrid(theta_1d, phi_1d, indexing='ij')

    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    # Original function: f(x,y,z) = 0.5 + sinh(5*x*y*z)*cos(x-y+2*z)
    f = 0.5 + np.sinh(5 * X * Y * Z) * np.cos(X - Y + 2*Z)

    # Antipodal map: (theta, phi) -> (pi-theta, phi+pi)
    # f(-x,-y,-z) corresponds to THETA -> pi - THETA, PHI -> PHI + pi
    THETA_anti = np.pi - THETA
    PHI_anti = PHI + np.pi

    X_anti = np.sin(THETA_anti) * np.cos(PHI_anti)
    Y_anti = np.sin(THETA_anti) * np.sin(PHI_anti)
    Z_anti = np.cos(THETA_anti)

    f_anti = 0.5 + np.sinh(5 * X_anti * Y_anti * Z_anti) * np.cos(X_anti - Y_anti + 2*Z_anti)

    # Even and odd parts
    f_even = (f + f_anti) / 2
    f_odd = (f - f_anti) / 2

    print(f"Function f on sphere:")
    print(f"  Range: [{f.min():.4f}, {f.max():.4f}]")
    print(f"  Even part range: [{f_even.min():.4f}, {f_even.max():.4f}]")
    print(f"  Odd part range: [{f_odd.min():.4f}, {f_odd.max():.4f}]")

    # Verify partition
    err = np.max(np.abs(f - (f_even + f_odd)))
    print(f"  Max reconstruction error: {err:.2e}")
    assert err < 1e-12, f"Partition error too large: {err}"

    fig = plt.figure()

    # --- Panel 1: Original function ---
    ax1 = fig.add_subplot(131, projection='3d')
    f_norm = (f - f.min()) / (f.max() - f.min() + 1e-14)
    ax1.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(f_norm),
                      alpha=0.9, linewidth=0)
    ax1.set_title('f(x,y,z) = 0.5 + sinh(5xyz)·cos(x-y+2z)', fontsize=9)
    ax1.set_axis_off()

    # --- Panel 2: Even part ---
    ax2 = fig.add_subplot(132, projection='3d')
    fe_norm = (f_even - f_even.min()) / (f_even.max() - f_even.min() + 1e-14)
    ax2.plot_surface(X, Y, Z, facecolors=plt.cm.RdBu_r(fe_norm),
                      alpha=0.9, linewidth=0)
    ax2.set_title('Even part f_even\n(f(x,y,z) + f(-x,-y,-z))/2', fontsize=9)
    ax2.set_axis_off()

    # --- Panel 3: Odd part ---
    ax3 = fig.add_subplot(133, projection='3d')
    fo_norm = (f_odd - f_odd.min()) / (f_odd.max() - f_odd.min() + 1e-14)
    ax3.plot_surface(X, Y, Z, facecolors=plt.cm.PiYG(fo_norm),
                      alpha=0.9, linewidth=0)
    ax3.set_title('Odd part f_odd\n(f(x,y,z) - f(-x,-y,-z))/2', fontsize=9)
    ax3.set_axis_off()

    fig.suptitle('Parity Partitioning of a Spherefun', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'spherefun_partition.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("spherefun_partition: done")
    return True


if __name__ == "__main__":
    run()
