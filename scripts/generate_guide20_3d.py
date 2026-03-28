"""Generate MATLAB-style 3D cross-section plots for Guide 20 (Ballfun).

Produces the characteristic 3-orthogonal-slices visualization matching
MATLAB's Ballfun plot: three colored disks (x=0, y=0, z=0 planes)
displayed in 3D space inside the unit ball wireframe.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from chebfunjax.plotting import chebfun_style, PARULA, _setup_3d_axes
chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)


def plot_ball_3slices(func_xyz, title="", fname="guide20_01.png", n=60):
    """Plot 3 orthogonal cross-sections through the ball in 3D.

    func_xyz: callable(X, Y, Z) -> values (works with numpy arrays)
    """
    fig = plt.figure(figsize=(6.1, 5.0))
    ax = fig.add_subplot(111, projection='3d')

    # Create polar grid for disk cross-sections
    r_pts = np.linspace(0, 1, n)
    theta_pts = np.linspace(0, 2*np.pi, 2*n)
    R, TH = np.meshgrid(r_pts, theta_pts)

    # Compute all values to get global color normalization
    all_vals = []

    # z=0 plane (xy-disk)
    X1 = R * np.cos(TH)
    Y1 = R * np.sin(TH)
    Z1 = np.zeros_like(X1)
    V1 = func_xyz(X1, Y1, Z1)
    all_vals.append(V1)

    # y=0 plane (xz-disk)
    X2 = R * np.cos(TH)
    Z2 = R * np.sin(TH)
    Y2 = np.zeros_like(X2)
    V2 = func_xyz(X2, Y2, Z2)
    all_vals.append(V2)

    # x=0 plane (yz-disk)
    Y3 = R * np.cos(TH)
    Z3 = R * np.sin(TH)
    X3 = np.zeros_like(Y3)
    V3 = func_xyz(X3, Y3, Z3)
    all_vals.append(V3)

    # Global normalization
    vmin = min(v.min() for v in all_vals)
    vmax = max(v.max() for v in all_vals)
    if vmax <= vmin:
        vmax = vmin + 1

    def to_colors(V):
        norm = (V - vmin) / (vmax - vmin)
        return PARULA(norm)

    # Plot z=0 slice
    ax.plot_surface(X1, Y1, Z1, facecolors=to_colors(V1),
                    rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)

    # Plot y=0 slice
    ax.plot_surface(X2, Y2, Z2, facecolors=to_colors(V2),
                    rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)

    # Plot x=0 slice
    ax.plot_surface(X3, Y3, Z3, facecolors=to_colors(V3),
                    rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)

    # Wireframe sphere outline
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.08, linewidth=0.3)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.view_init(elev=25, azim=-55)

    # MATLAB-style 3D box
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
    ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
    ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
    ax.grid(True, alpha=0.15)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])

    if title:
        ax.set_title(title, fontsize=10, pad=5)

    fig.set_facecolor('white')
    fig.tight_layout()
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# Plot 1: f = x^2 + y^2 + z^2 (radial function)
plot_ball_3slices(
    lambda X, Y, Z: X**2 + Y**2 + Z**2,
    title=r"$f(x,y,z) = x^2+y^2+z^2$",
    fname="guide20_01.png"
)

# Plot 2: f = cos(pi*x)*sin(pi*y)*exp(z)
plot_ball_3slices(
    lambda X, Y, Z: np.cos(np.pi*X) * np.sin(np.pi*Y) * np.exp(Z),
    title=r"$\cos(\pi x)\sin(\pi y)e^z$",
    fname="guide20_02.png"
)

# Plot 3: Solid harmonic Y_2^0 = (3z^2 - r^2)
plot_ball_3slices(
    lambda X, Y, Z: 3*Z**2 - (X**2 + Y**2 + Z**2),
    title=r"$Y_2^0: 3z^2 - r^2$",
    fname="guide20_03.png"
)

# Plot 4: Gaussian exp(-5r^2)
plot_ball_3slices(
    lambda X, Y, Z: np.exp(-5*(X**2 + Y**2 + Z**2)),
    title=r"$e^{-5r^2}$",
    fname="guide20_04.png"
)

# Plot 5: x*y*z (mixed)
plot_ball_3slices(
    lambda X, Y, Z: X * Y * Z,
    title=r"$f = xyz$",
    fname="guide20_05.png"
)

# Plot 6: sin(pi*r)/r-like
plot_ball_3slices(
    lambda X, Y, Z: np.sinc(3*np.sqrt(X**2 + Y**2 + Z**2)),
    title=r"$\mathrm{sinc}(3r)$",
    fname="guide20_06.png"
)

print(f"\nGuide 20: Generated 6 plots.")
