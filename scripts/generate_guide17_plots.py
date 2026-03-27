"""Generate all plots for Guide Chapter 17 (Spherefun).

Matches figures from the original Chebfun guide chapter 17.
"""
import matplotlib
matplotlib.use('Agg')
import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from chebfunjax.plotting import chebfun_style, plot_sphere
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

def eval_on_sphere(f, n_lam=200, n_theta=100):
    """Evaluate Spherefun on a lat-lon grid."""
    lam = np.linspace(-np.pi, np.pi, n_lam, endpoint=False)
    theta = np.linspace(0.0, np.pi, n_theta)
    LAM, THETA = np.meshgrid(lam, theta, indexing='ij')
    ZZ = np.array(f(jnp.array(LAM.ravel()), jnp.array(THETA.ravel()))).reshape(LAM.shape)
    return LAM, THETA, ZZ

def sphere_3d(f, title='', cmap='RdBu_r'):
    """3D coloured sphere surface plot."""
    LAM, THETA, ZZ = eval_on_sphere(f)
    XX = np.sin(THETA) * np.cos(LAM)
    YY = np.sin(THETA) * np.sin(LAM)
    ZZ_c = np.cos(THETA)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    fmin, fmax = ZZ.min(), ZZ.max()
    if fmax > fmin:
        norm_vals = (ZZ - fmin) / (fmax - fmin)
    else:
        norm_vals = np.zeros_like(ZZ)
    cmap_obj = plt.get_cmap(cmap)
    fcolors = cmap_obj(norm_vals)
    ax.plot_surface(XX, YY, ZZ_c, facecolors=fcolors, linewidth=0, antialiased=True, alpha=0.95)
    if title: ax.set_title(title, fontsize=11)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    fig.set_facecolor('white'); fig.tight_layout()
    return fig, ax

def sphere_flat(f, title='', cmap='RdBu_r'):
    """Flat (lambda, theta) projection of a Spherefun."""
    LAM, THETA, ZZ = eval_on_sphere(f)
    fig, ax = plt.subplots(figsize=(8, 4))
    pcm = ax.pcolormesh(LAM, THETA, ZZ, cmap=cmap, shading='auto')
    fig.colorbar(pcm, ax=ax, fraction=0.02, pad=0.04)
    ax.set_xlabel('lambda'); ax.set_ylabel('theta')
    ax.set_title(title); ax.invert_yaxis()
    fig.set_facecolor('white'); fig.tight_layout()
    return fig, ax

# Plot 01: cos(x+y+z) on sphere (Section 17.1)
try:
    f = Spherefun.from_function(
        lambda lam, th: jnp.cos(jnp.cos(lam)*jnp.sin(th)
            + jnp.sin(lam)*jnp.sin(th) + jnp.cos(th)))
    fig, ax = sphere_3d(f); save(fig, "cos(x+y+z)")
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
    # Evaluate both on the same grid
    LAM, THETA, Zf = eval_on_sphere(f)
    _, _, Zg = eval_on_sphere(g)
    XX = np.sin(THETA)*np.cos(LAM); YY = np.sin(THETA)*np.sin(LAM); ZC = np.cos(THETA)
    Zsum = Zf + Zg
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    fmin, fmax = Zsum.min(), Zsum.max()
    norm_v = (Zsum-fmin)/(fmax-fmin) if fmax>fmin else np.zeros_like(Zsum)
    ax.plot_surface(XX, YY, ZC, facecolors=plt.get_cmap('RdBu_r')(norm_v),
                    linewidth=0, antialiased=True, alpha=0.95)
    ax.set_title('f + g'); fig.set_facecolor('white'); fig.tight_layout()
    save(fig, "f + g")
except Exception as e:
    plot_num += 1; print(f"  guide17_{plot_num:02d}.png FAILED: {e}")

# Plot 05: f .* g (Section 17.2)
try:
    Zprod = Zf * Zg
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    fmin, fmax = Zprod.min(), Zprod.max()
    norm_v = (Zprod-fmin)/(fmax-fmin) if fmax>fmin else np.zeros_like(Zprod)
    ax.plot_surface(XX, YY, ZC, facecolors=plt.get_cmap('RdBu_r')(norm_v),
                    linewidth=0, antialiased=True, alpha=0.95)
    ax.set_title('f .* g'); fig.set_facecolor('white'); fig.tight_layout()
    save(fig, "f .* g")
except Exception as e:
    plot_num += 1; print(f"  guide17_{plot_num:02d}.png FAILED: {e}")

# Plot 06: Contour plot on sphere (Section 17.3)
try:
    LAM, THETA, ZZf = eval_on_sphere(f)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.contour(LAM, THETA, ZZf, levels=20, linewidths=0.8)
    ax.set_xlabel('lambda'); ax.set_ylabel('theta')
    ax.set_title('Contour plot of f'); ax.invert_yaxis()
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig, "contour")
except Exception as e:
    plot_num += 1; print(f"  guide17_{plot_num:02d}.png FAILED: {e}")

# Plot 07: Spherical harmonic Y_6^0 (Section 17.4)
try:
    from scipy.special import lpmv
    # Y_6^0 = normalization * P_6(cos(theta))
    def P6(x): return (231*x**6 - 315*x**4 + 105*x**2 - 5)/16
    Y60 = Spherefun.from_function(lambda lam, th: P6(jnp.cos(th)))
    fig, ax = sphere_3d(Y60, title='Y_6^0')
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
        ax.plot(np.array(th_eval), np.array(f.cols[j](th_eval)))
    ax.set_title(f'{nc} column slices of f')
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
        ax.plot(np.array(lam_eval), np.array(f.rows[j](lam_eval)))
    ax.set_title(f'{nr} row slices of f')
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
    ax1.set_title('Fourier coefficients (columns)'); ax1.set_xlabel('Index')
    for r in f.rows:
        cf = np.array(jnp.abs(r.coeffs))
        ax2.semilogy(range(len(cf)), cf+1e-17, 'o-', ms=3, alpha=0.5)
    ax2.set_title('Fourier coefficients (rows)'); ax2.set_xlabel('Index')
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
