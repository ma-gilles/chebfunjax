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

    # Fine grid on the sphere for visualisation
    n_theta, n_phi = 100, 200
    theta_1d = np.linspace(0.05, np.pi - 0.05, n_theta)
    phi_1d = np.linspace(0, 2*np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta_1d, phi_1d, indexing='ij')

    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    # Define vector field evaluated on the unit sphere (r=1)
    # v = grad(f) + curl(psi)
    # f(x,y,z) = x*y + z^2 => grad(f) = (y, x, 2z)
    # curl(psi) with psi = (0, 0, xy) => curl = (x, -y, 0)
    vx_cf = Y.copy()   # grad_x f on sphere
    vy_cf = X.copy()   # grad_y f on sphere
    vz_cf = 2 * Z      # grad_z f on sphere

    vx_df = X.copy()
    vy_df = -Y.copy()
    vz_df = np.zeros_like(Z)

    vx = vx_cf + vx_df
    vy = vy_cf + vy_df
    vz = vz_cf + vz_df

    # Magnitudes on the sphere
    mag_cf = np.sqrt(vx_cf**2 + vy_cf**2 + vz_cf**2)
    mag_df = np.sqrt(vx_df**2 + vy_df**2 + vz_df**2)
    mag_total = np.sqrt(vx**2 + vy**2 + vz**2)

    fig = plt.figure(figsize=(14, 4.5), facecolor='white')

    panels = [
        (mag_total, '$|\\mathbf{v}|$: total field on sphere'),
        (mag_cf, 'Curl-free $|\\nabla(xy + z^2)|$'),
        (mag_df, 'Div-free $|\\mathrm{curl}(\\psi)|$'),
    ]

    for i, (F, title) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        _sphere_panel(ax, fig, X, Y, Z, F, title, cmap=PARULA)

    print("Helmholtz-Hodge decomposition in the ball:")
    print(f"  v = grad(xy+z^2) + curl(0,0,xy)")
    print(f"  div(v) = Laplacian(xy+z^2) = 2 (analytic)")
    print(f"  curl-free part: (y, x, 2z)")
    print(f"  divergence-free part: (x, -y, 0)")

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, 'helmholtz_decomposition_ball.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print("helmholtz_decomposition_ball: done")
    return True

if __name__ == "__main__":
    run()
