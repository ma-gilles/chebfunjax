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
from chebfunjax.plotting import chebfun_style, PARULA, _setup_3d_axes
chebfun_style()

def toroidal_field(T_func, r, theta, phi):
    """Compute toroidal field: T_field = curl(r * T_hat)."""
    T = T_func(theta, phi)
    dt = 0.001

    dT_dtheta = (T_func(theta + dt, phi) - T_func(theta - dt, phi)) / (2*dt)
    dT_dphi = (T_func(theta, phi + dt) - T_func(theta, phi - dt)) / (2*dt)

    sin_theta = np.sin(theta) + 1e-14
    Vt = dT_dphi / sin_theta
    Vp = -dT_dtheta
    Vr = np.zeros_like(Vt)
    return Vr, Vt, Vp

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

    mag_T = np.sqrt(Vt_T**2 + Vp_T**2)
    mag_P = np.sqrt(Vt_P**2 + Vp_P**2)
    mag = np.sqrt(Vt**2 + Vp**2)

    fig = plt.figure(figsize=(14, 4.5), facecolor='white')

    panels = [
        (mag_T, '$|$Toroidal$|$: $|\\mathrm{curl}(\\hat{r} T)|$'),
        (mag_P, '$|$Poloidal$|$: $|\\mathrm{curl}(\\hat{r} P)|$'),
        (mag, '$|$Total field$|$: Toroidal + Poloidal'),
    ]

    for i, (F, title) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        _sphere_panel(ax, fig, X, Y, Z, F, title, cmap=PARULA)

    print("Poloidal-Toroidal decomposition:")
    print(f"  T = cos(th)sin(phi): toroidal potential")
    print(f"  P = sin(th)cos(phi): poloidal potential")
    print(f"  Max toroidal: {mag_T.max():.4f}")
    print(f"  Max poloidal: {mag_P.max():.4f}")

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, 'pt_decomposition.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print("pt_decomposition: done")
    return True

if __name__ == "__main__":
    run()
