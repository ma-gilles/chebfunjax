"""Generate plots for Guide Chapter 20: Ballfun.

Faithfully translates MATLAB @ballfun/plot.m: 5 surfaces (1 sphere at r=0.5,
2 cones at constant theta, 2 half-planes at constant lambda), converted from
spherical to Cartesian via sph2cart, with camlight/phong shading.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys, traceback
from matplotlib.colors import LightSource

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from chebfunjax.plotting import PARULA, chebfun_style
chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)
plot_num = 0


def sph2cart(az, el, r):
    """MATLAB sph2cart: azimuth, elevation, radius -> x, y, z."""
    return r*np.cos(el)*np.cos(az), r*np.cos(el)*np.sin(az), r*np.sin(el)


def plot_ballfun(func_xyz, title='', n_r=30, n_ang=40):
    """Plot like MATLAB @ballfun/plot.m plotBall() using analytical function.

    Generates 5 surfaces:
      - 1 sphere at r ≈ 0.5
      - 2 cones at constant theta (theta=0 and theta≈pi/4)
      - 2 half-planes at constant lambda (lambda≈-pi and lambda≈-pi/2)
    Each converted from spherical to Cartesian via sph2cart.
    """
    r = np.linspace(0, 1, n_r)
    lam = np.linspace(-np.pi, np.pi, n_ang)
    th = np.linspace(0, np.pi, n_ang)

    # Slice positions matching MATLAB (lines 96-100 of plot.m)
    rslice = [r[np.argmin(np.abs(r - 0.5))]]
    tslice = [th[0], th[n_ang // 4]]
    lslice = [lam[0], lam[n_ang // 4]]

    slices = []

    # 2 constant-theta cones
    for th_val in tslice:
        R2, LAM2 = np.meshgrid(r, lam, indexing='ij')
        el = np.pi / 2 - th_val
        xs, ys, zs = sph2cart(LAM2, np.full_like(R2, el), R2)
        slices.append((xs, ys, zs, func_xyz(xs, ys, zs)))

    # 1 constant-r sphere
    LAM2, TH2 = np.meshgrid(lam, th, indexing='ij')
    xs, ys, zs = sph2cart(LAM2, np.pi / 2 - TH2, np.full_like(LAM2, rslice[0]))
    slices.append((xs, ys, zs, func_xyz(xs, ys, zs)))

    # 2 constant-lambda half-planes
    for lam_val in lslice:
        R2, TH2 = np.meshgrid(r, th, indexing='ij')
        xs, ys, zs = sph2cart(np.full_like(R2, lam_val), np.pi / 2 - TH2, R2)
        slices.append((xs, ys, zs, func_xyz(xs, ys, zs)))

    # Global color normalization
    all_v = np.concatenate([s[3].ravel() for s in slices])
    vmin, vmax = float(all_v.min()), float(all_v.max())
    if vmax <= vmin:
        vmax = vmin + 1
    norm_fn = lambda v: (v - vmin) / (vmax - vmin)

    ls = LightSource(azdeg=315, altdeg=45)

    fig = plt.figure(figsize=(6.0, 2.7))
    ax = fig.add_subplot(111, projection='3d')

    # MATLAB: surf(xs,ys,zs,CData,'facecolor','interp','edgecolor','none')
    for xs, ys, zs, vals in slices:
        rgb = PARULA(norm_fn(vals))[:, :, :3]
        shaded = ls.shade_rgb(rgb, zs)
        fc = np.ones((*shaded.shape[:2], 4))
        fc[:, :, :3] = shaded
        ax.plot_surface(xs, ys, zs, facecolors=fc,
                        rstride=1, cstride=1, linewidth=0,
                        antialiased=True, shade=False)

    # MATLAB: axis([-1 1 -1 1 -1 1]); daspect([1 1 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.tick_params(labelsize=7, pad=-3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.set_edgecolor((0.85, 0.85, 0.85, 0.3))
    ax.grid(True, alpha=0.1, linewidth=0.3)
    ax.view_init(elev=25, azim=-55)

    if title:
        ax.set_title(title, fontsize=9, pad=0)
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
    return fig, ax


def save(fig, desc):
    global plot_num
    plot_num += 1
    fname = os.path.join(OUT, f'guide20_{plot_num:02d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved {fname}: {desc}")


# Plot 1: f = x^2 + y^2 + z^2
try:
    fig, ax = plot_ballfun(lambda x, y, z: x**2 + y**2 + z**2,
                           title=r"$x^2+y^2+z^2$")
    save(fig, "r^2 three orthogonal slices")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 1")

# Plot 2: f = cos(pi*x)*sin(pi*y)*exp(z)
try:
    fig, ax = plot_ballfun(
        lambda x, y, z: np.cos(np.pi*x) * np.sin(np.pi*y) * np.exp(z),
        title=r"$\cos(\pi x)\sin(\pi y)e^z$")
    save(fig, "cos*sin*exp")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 2")

# Plot 3: Solid harmonic Y_2^0
try:
    fig, ax = plot_ballfun(lambda x, y, z: 3*z**2 - (x**2 + y**2 + z**2),
                           title=r"$Y_2^0: 3z^2 - r^2$")
    save(fig, "solid harmonic")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 3")

# Plot 4: Gaussian
try:
    fig, ax = plot_ballfun(lambda x, y, z: np.exp(-5*(x**2 + y**2 + z**2)),
                           title=r"$e^{-5r^2}$")
    save(fig, "Gaussian")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 4")

# Plot 5: xyz
try:
    fig, ax = plot_ballfun(lambda x, y, z: x * y * z,
                           title=r"$xyz$")
    save(fig, "xyz")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 5")

# Plot 6: sinc(3r)
try:
    fig, ax = plot_ballfun(lambda x, y, z: np.sinc(3*np.sqrt(x**2+y**2+z**2)/np.pi),
                           title=r"$\mathrm{sinc}(3r)$")
    save(fig, "sinc")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 6")

print(f"\nGuide 20: Generated {plot_num} plots.")
