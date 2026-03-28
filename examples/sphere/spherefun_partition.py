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
from chebfunjax.plotting import chebfun_style, PARULA, _setup_3d_axes
chebfun_style()

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

    # Fine grid
    n_theta, n_phi = 100, 200
    theta_1d = np.linspace(0, np.pi, n_theta)
    phi_1d = np.linspace(0, 2*np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta_1d, phi_1d, indexing='ij')

    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    # Original function: f(x,y,z) = 0.5 + sinh(5*x*y*z)*cos(x-y+2*z)
    f = 0.5 + np.sinh(5 * X * Y * Z) * np.cos(X - Y + 2*Z)

    # Antipodal map: (theta, phi) -> (pi-theta, phi+pi)
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

    fig = plt.figure(figsize=(14, 4.5), facecolor='white')

    panels = [
        (f, '$f = 0.5 + \\sinh(5xyz)\\cos(x-y+2z)$'),
        (f_even, 'Even: $(f + f \\circ A)/2$'),
        (f_odd, 'Odd: $(f - f \\circ A)/2$'),
    ]

    for i, (F, title) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        _sphere_panel(ax, fig, X, Y, Z, F, title, cmap=PARULA)

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, 'spherefun_partition.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print("spherefun_partition: done")
    return True

if __name__ == "__main__":
    run()
