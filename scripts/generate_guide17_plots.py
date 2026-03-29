"""Generate all plots for Guide Chapter 17 (Spherefun).

Uses PARULA colormap on coloured spheres matching MATLAB Chebfun style.
"""
import matplotlib
matplotlib.use('Agg')
import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from chebfunjax.plotting import (
    chebfun_style, plot_sphere, contour_sphere, PARULA, _setup_3d_axes,
)
from chebfunjax.spherefun import Spherefun

chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)
plot_num = 0

def save(fig, desc=""):
    global plot_num
    plot_num += 1
    fname = os.path.join(OUT, f'guide17_{plot_num:02d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  guide17_{plot_num:02d}.png: {desc}")

def eval_on_sphere(f, n_pts=200):
    """Evaluate Spherefun on a lat-lon grid (matching MATLAB's 200x200)."""
    l = np.linspace(-np.pi, np.pi, n_pts)
    t = np.linspace(0.0, np.pi, n_pts)
    ll, tt = np.meshgrid(l, t)  # MATLAB ordering: (n_pts, n_pts)
    ZZ = np.array(f(jnp.array(ll.ravel()), jnp.array(tt.ravel()))).reshape(ll.shape)
    # Return sph2cart-style coordinates
    elev = np.pi / 2 - tt
    XX = np.cos(elev) * np.cos(ll)
    YY = np.cos(elev) * np.sin(ll)
    ZZ_c = np.sin(elev)
    return ll, tt, XX, YY, ZZ_c, ZZ

def sphere_3d(f, title='', cmap=None):
    """3D coloured sphere surface plot using plot_sphere (MATLAB-faithful)."""
    return plot_sphere(f, title=title, cmap=cmap)

def sphere_3d_from_values(XX, YY, ZZ_c, vals, title='', cmap=None):
    """3D coloured sphere from pre-computed Cartesian coordinates and values.

    Uses LightSource shading to match MATLAB camlight + lighting phong.
    """
    from matplotlib.colors import LightSource
    if cmap is None:
        cmap = PARULA
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap
    fmin, fmax = vals.min(), vals.max()
    norm_v = (vals - fmin) / (fmax - fmin) if fmax > fmin else np.zeros_like(vals)

    ls = LightSource(azdeg=315, altdeg=45)
    rgb = cmap_obj(norm_v)[:, :, :3]
    shaded = ls.shade_rgb(rgb, ZZ_c)
    fcolors = np.ones((*shaded.shape[:2], 4))
    fcolors[:, :, :3] = shaded

    fig, ax = _setup_3d_axes(None, None, elev=8, azim=-36, figsize=(6.1, 5.0))
    ax.plot_surface(XX, YY, ZZ_c, facecolors=fcolors, linewidth=0,
                    antialiased=True, shade=False,
                    rstride=1, cstride=1)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_box_aspect([1, 1, 1])
    if title:
        ax.set_title(title, fontsize=10, pad=0)
    return fig, ax

def sphere_flat(f, title='', cmap=None):
    """Flat (lambda, theta) projection of a Spherefun."""
    if cmap is None:
        cmap = PARULA
    ll, tt, XX, YY, ZZ_c, ZZ = eval_on_sphere(f)
    fig, ax = plt.subplots(figsize=(8, 4))
    pcm = ax.pcolormesh(ll, tt, ZZ, cmap=cmap, shading='auto')
    fig.colorbar(pcm, ax=ax, fraction=0.02, pad=0.04)
    ax.set_xlabel('lambda', fontsize=9)
    ax.set_ylabel('theta', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.invert_yaxis()
    fig.set_facecolor('white'); fig.tight_layout()
    return fig, ax

# Plot 01: cos(x+y+z) on sphere (Section 17.1)
try:
    f = Spherefun.from_function(
        lambda lam, th: jnp.cos(jnp.cos(lam)*jnp.sin(th)
            + jnp.sin(lam)*jnp.sin(th) + jnp.cos(th)))
    fig, ax = sphere_3d(f, title='cos(x+y+z)')
    save(fig, "cos(x+y+z)")
except Exception as e:
    plot_num += 1; print(f"  guide17_{plot_num:02d}.png FAILED: {e}")

# Plot 02: f in (lambda, theta) coords (Section 17.1)
try:
    fig, ax = sphere_flat(f, title='f in (lambda, theta) coords')
    save(fig, "flat view")
except Exception as e:
    plot_num += 1; print(f"  guide17_{plot_num:02d}.png FAILED: {e}")

# Plot 03: Gaussian bump (Section 17.2)
try:
    g = Spherefun.from_function(
        lambda lam, th: jnp.exp(-10*(jnp.cos(lam)*jnp.sin(th)-0.5)**2
            - 10*(jnp.sin(lam)*jnp.sin(th))**2 - 10*(jnp.cos(th)-0.5)**2))
    fig, ax = sphere_3d(g, title='Gaussian bump')
    save(fig, "Gaussian bump")
except Exception as e:
    plot_num += 1; print(f"  guide17_{plot_num:02d}.png FAILED: {e}")

# Plot 04: f + g (Section 17.2)
try:
    f_plus_g = Spherefun.from_function(lambda lam, th: f(lam, th) + g(lam, th))
    fig, ax = sphere_3d(f_plus_g, title='f + g')
    save(fig, "f + g")
except Exception as e:
    plot_num += 1; print(f"  guide17_{plot_num:02d}.png FAILED: {e}")

# Plot 05: f .* g (Section 17.2)
try:
    f_times_g = Spherefun.from_function(lambda lam, th: f(lam, th) * g(lam, th))
    fig, ax = sphere_3d(f_times_g, title='f .* g')
    save(fig, "f .* g")
except Exception as e:
    plot_num += 1; print(f"  guide17_{plot_num:02d}.png FAILED: {e}")

# Plot 06: Contour plot on sphere (Section 17.3)
try:
    fig, ax = contour_sphere(f, levels=20, title='Contour plot of f')
    save(fig, "contour")
except Exception as e:
    plot_num += 1; print(f"  guide17_{plot_num:02d}.png FAILED: {e}")

# Plot 07: Spherical harmonic Y_6^0 (Section 17.4)
try:
    def P6(x): return (231*x**6 - 315*x**4 + 105*x**2 - 5)/16
    Y60 = Spherefun.from_function(lambda lam, th: P6(jnp.cos(th)))
    fig, ax = sphere_3d(Y60, title=r'$Y_6^0$')
    save(fig, "Y_6^0")
except Exception as e:
    plot_num += 1; print(f"  guide17_{plot_num:02d}.png FAILED: {e}")

# Plot 08: Earth-like map (Section 17.5)
try:
    earth = Spherefun.from_function(
        lambda lam, th: jnp.cos(3*lam)*jnp.sin(th)**3 + jnp.cos(5*th))
    fig, ax = sphere_3d(earth, title='Spherefun example')
    save(fig, "earth-like")
except Exception as e:
    plot_num += 1; print(f"  guide17_{plot_num:02d}.png FAILED: {e}")

# Plot 09: Column slices (Section 17.6)
try:
    fig, ax = plt.subplots(figsize=(6, 4))
    th_eval = jnp.linspace(-1, 1, 200)
    nc = min(5, len(f.cols))
    for j in range(nc):
        ax.plot(np.array(th_eval), np.array(f.cols[j](th_eval)), linewidth=1.2)
    ax.set_title(f'{nc} column slices of f', fontsize=10)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig, "column slices")
except Exception as e:
    plot_num += 1; print(f"  guide17_{plot_num:02d}.png FAILED: {e}")

# Plot 10: Row slices (Section 17.6)
try:
    fig, ax = plt.subplots(figsize=(6, 4))
    lam_eval = jnp.linspace(-1, 1, 200)
    nr = min(5, len(f.rows))
    for j in range(nr):
        ax.plot(np.array(lam_eval), np.array(f.rows[j](lam_eval)), linewidth=1.2)
    ax.set_title(f'{nr} row slices of f', fontsize=10)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig, "row slices")
except Exception as e:
    plot_num += 1; print(f"  guide17_{plot_num:02d}.png FAILED: {e}")

# Plot 11: Plotcoeffs (Section 17.6)
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for c in f.cols:
        cf = np.array(jnp.abs(c.coeffs))
        ax1.semilogy(range(len(cf)), cf+1e-17, 'o-', ms=3, alpha=0.5)
    ax1.set_title('Fourier coefficients (columns)', fontsize=10)
    ax1.set_xlabel('Index', fontsize=9)
    for r in f.rows:
        cf = np.array(jnp.abs(r.coeffs))
        ax2.semilogy(range(len(cf)), cf+1e-17, 'o-', ms=3, alpha=0.5)
    ax2.set_title('Fourier coefficients (rows)', fontsize=10)
    ax2.set_xlabel('Index', fontsize=9)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig, "plotcoeffs")
except Exception as e:
    plot_num += 1; print(f"  guide17_{plot_num:02d}.png FAILED: {e}")

# Plot 12: Poisson on sphere (Section 17.5)
try:
    poisson_rhs = Spherefun.from_function(
        lambda lam, th: jnp.sin(5*lam)*jnp.sin(th)**4*jnp.cos(th))
    fig, ax = sphere_3d(poisson_rhs, title='Poisson RHS on sphere')
    save(fig, "Poisson RHS")
except Exception as e:
    plot_num += 1; print(f"  guide17_{plot_num:02d}.png FAILED: {e}")

print(f"\nGuide 17: {plot_num} plots generated.")
