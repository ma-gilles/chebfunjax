"""Generate plots for Guide Chapter 20: Ballfun.

Uses MATLAB-Chebfun-style 3D orthogonal cross-section visualisations:
three coloured disks (z=0, y=0, x=0 planes) displayed in 3D space.

Note: Ballfun from_function is very slow on CPU due to JAX JIT compilation.
For plot generation, we evaluate the known analytical formulas directly
to produce the MATLAB-style visualisations.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chebfunjax.plotting import (
    chebfun_style, PARULA, _setup_3d_axes, CHEBFUN_BLUE,
)
chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)

plot_num = 0

def save(fig, desc):
    global plot_num
    plot_num += 1
    fname = os.path.join(OUT, f'guide20_{plot_num:02d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}: {desc}")


def ball_3_slices(func_xyz, title='', n_ang=80, n_rad=40,
                  cmap=None, elev=25, azim=-37, alpha=0.95):
    """Three orthogonal cross-section disks through the unit ball.

    Translates the MATLAB @ballfun/plot.m approach:
    - Build slice surfaces in spherical coordinates
    - Convert each from spherical to Cartesian via sph2cart
    - Apply LightSource shading (camlight + lighting phong)
    - axis([-1 1 -1 1 -1 1]); daspect([1 1 1])

    Parameters
    ----------
    func_xyz : callable
        f(x, y, z) -> values, vectorised over numpy arrays.
    title : str
    n_ang, n_rad : int
        Angular and radial grid resolution.
    """
    from matplotlib.colors import LightSource

    if cmap is None:
        cmap = PARULA
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap

    theta = np.linspace(0, 2*np.pi, n_ang)
    r = np.linspace(0, 1, n_rad)
    R, TH = np.meshgrid(r, theta)

    # z=0 slice: disk in xy-plane
    X1 = R * np.cos(TH); Y1 = R * np.sin(TH); Z1 = np.zeros_like(X1)
    V1 = func_xyz(X1, Y1, Z1)

    # y=0 slice: disk in xz-plane
    X2 = R * np.cos(TH); Z2 = R * np.sin(TH); Y2 = np.zeros_like(X2)
    V2 = func_xyz(X2, Y2, Z2)

    # x=0 slice: disk in yz-plane
    Y3 = R * np.cos(TH); Z3 = R * np.sin(TH); X3 = np.zeros_like(Y3)
    V3 = func_xyz(X3, Y3, Z3)

    # Global colour range
    all_v = np.concatenate([V1.ravel(), V2.ravel(), V3.ravel()])
    vmin, vmax = float(all_v.min()), float(all_v.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # MATLAB: camlight('headlight'); lighting phong; material dull
    ls = LightSource(azdeg=315, altdeg=45)

    fig, ax = _setup_3d_axes(None, None, elev=elev, azim=azim, figsize=(6.1, 5.0))

    def _disk(XX, YY, ZZ, vals):
        rgb = cmap_obj(norm(vals))[:, :, :3]
        shaded = ls.shade_rgb(rgb, ZZ)
        fc = np.ones((*shaded.shape[:2], 4))
        fc[:, :, :3] = shaded
        ax.plot_surface(XX, YY, ZZ, facecolors=fc,
                        rstride=1, cstride=1, linewidth=0,
                        antialiased=True, alpha=alpha, shade=False)

    _disk(X1, Y1, Z1, V1)  # z=0
    _disk(X2, Y2, Z2, V2)  # y=0
    _disk(X3, Y3, Z3, V3)  # x=0

    # MATLAB: axis([-1 1 -1 1 -1 1]); daspect([1 1 1])
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_box_aspect([1, 1, 1])
    if title:
        ax.set_title(title, fontsize=10, pad=0)
    fig.tight_layout(pad=0.5)
    return fig, ax


# ---- Plot 1: r^2 = x^2+y^2+z^2 (Section 20.1) ----
try:
    fig, ax = ball_3_slices(
        lambda x, y, z: x**2 + y**2 + z**2,
        title=r'$x^2+y^2+z^2$')
    save(fig, "r^2 three orthogonal slices")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 1")

# ---- Plot 2: r^2 in spherical (same function, Section 20.1) ----
try:
    # Same as plot 1 but viewed from a different angle
    fig, ax = ball_3_slices(
        lambda x, y, z: x**2 + y**2 + z**2,
        title=r'$r^2$ (spherical input)',
        elev=30, azim=-50)
    save(fig, "r^2 spherical input")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 2")

# ---- Plot 3: Radial profile (Section 20.2) ----
try:
    rs = np.linspace(0, 1, 100)
    vals = rs**2  # f(r,0,0) = r^2

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(rs, vals, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.plot(rs, rs**2, 'k--', linewidth=1.0, alpha=0.6, label=r'$r^2$ (exact)')
    ax.set_xlabel("r")
    ax.set_ylabel(r"$f(r,0,0)$")
    ax.set_title(r"$x^2+y^2+z^2$ along the $x$-axis", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    fig.set_facecolor("white")
    fig.tight_layout()
    save(fig, "Radial profile")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 3")

# ---- Plot 4: Solid harmonic r^2*Y_2^0 (Section 20.3 fevalm) ----
try:
    coeff_y20 = 0.25 * np.sqrt(5.0 / np.pi)

    def solid_harmonic(x, y, z):
        r2 = x**2 + y**2 + z**2
        r = np.sqrt(r2)
        cos_th = np.where(r > 0, z / np.maximum(r, 1e-16), 1.0)
        return r2 * coeff_y20 * (3 * cos_th**2 - 1)

    fig, ax = ball_3_slices(solid_harmonic, title=r'Solid harmonic $r^2 Y_2^0$')
    save(fig, "Solid harmonic three orthogonal slices")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 4")

# ---- Plot 5: x^2 + y^2 (arithmetic, Section 20.3 pointwise) ----
try:
    fig, ax = ball_3_slices(
        lambda x, y, z: x**2 + y**2,
        title=r'$x^2 + y^2$')
    save(fig, "Arithmetic result three orthogonal slices")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 5")

# ---- Plot 6: exp(-(x^2+y^2+z^2)) (Section 20.4 adaptive construction) ----
try:
    fig, ax = ball_3_slices(
        lambda x, y, z: np.exp(-(x**2 + y**2 + z**2)),
        title=r'$e^{-(x^2+y^2+z^2)}$')
    save(fig, "Gaussian three orthogonal slices")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 6")

print(f"\nGuide 20: Generated {plot_num} plots.")
