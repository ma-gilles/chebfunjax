"""Generate plots for Guide Chapter 12: Chebfun2: Getting Started.

Faithful translation of all figures from the original MATLAB Chebfun Guide
Chapter 12 (https://www.chebfun.org/docs/guide/guide12.html).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax.numpy as jnp
from chebfunjax.plotting import chebfun_style, surf, contour, phaseplot
from chebfunjax.chebfun2d import chebfun2, Chebfun2

chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)

plot_num = 0

def save(fig, desc):
    global plot_num
    plot_num += 1
    fname = os.path.join(OUT, f'guide12_{plot_num:02d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}: {desc}")


# ---- Plot 1: 12.1 - Peaks function surface plot ----
try:
    # MATLAB peaks function
    def peaks(x, y):
        return (3*(1-x)**2 * jnp.exp(-x**2 - (y+1)**2)
                - 10*(x/5 - x**3 - y**5) * jnp.exp(-x**2 - y**2)
                - 1/3 * jnp.exp(-(x+1)**2 - y**2))

    f = chebfun2(peaks, domain=(-3.0, 3.0, -3.0, 3.0))
    fig, ax = surf(f, title='Chebfun2 Peaks')
    ax.set_zlim(-7, 9)
    save(fig, "Peaks surface")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 1")

# ---- Plot 2: 12.3 - Surface plot of cos(2*pi*x*y) ----
try:
    f = chebfun2(lambda x, y: jnp.cos(2*jnp.pi*x*y))
    fig, ax = surf(f, title=r'$\cos(2\pi x y)$')
    ax.set_zlim(-2, 2)
    save(fig, "cos(2*pi*x*y) surface")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 2")

# ---- Plot 3: 12.3 - Contour plot of cos(2*pi*x*y) ----
try:
    fig, ax = contour(f, title=r'$\cos(2\pi x y)$')
    ax.set_aspect('equal')
    save(fig, "cos(2*pi*x*y) contour")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 3")

# ---- Plot 4: 12.4 - Zero contours of f - 0.95 ----
try:
    f = chebfun2(lambda x, y: jnp.cos(2*jnp.pi*x*y))
    g = chebfun2(lambda x, y: jnp.cos(2*jnp.pi*x*y) - 0.95)
    curves = g.roots()
    fig, ax = plt.subplots(figsize=(5, 5))
    for c in curves:
        ax.plot(c[:, 0], c[:, 1], 'b-', linewidth=1.0)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title('Zero contours of f - 0.95')
    fig.tight_layout()
    save(fig, "Zero contours of f-0.95")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 4")

# ---- Plot 5: 12.4 - Partial derivative df/dy ----
try:
    f = chebfun2(lambda x, y: jnp.cos(2*jnp.pi*x*y))
    fy = f.diff(dim=1)
    fig, ax = surf(fy, title=r'$\partial f/\partial y$')
    save(fig, "df/dy surface")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 5")

# ---- Plot 6: 12.6 - Composition: 1/(2+cos(.25+x^2*y+y^2)) contour ----
try:
    f = chebfun2(lambda x, y: 1.0 / (2 + jnp.cos(0.25 + x**2 * y + y**2)),
                 domain=(-4.0, 4.0, -2.0, 2.0))
    fig, ax = contour(f, title=r'$1/(2+\cos(0.25+x^2 y+y^2))$')
    ax.set_aspect('equal')
    save(fig, "Composition contour")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 6")

# ---- Plot 7: 12.7 - Phase portrait of sin(z)-sinh(z) ----
try:
    def f_complex(z):
        return np.sin(z) - np.sinh(z)
    region = [-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi]
    fig, ax = phaseplot(f_complex, region=region,
                        title=r'Phase portrait of $\sin(z) - \sinh(z)$')
    save(fig, "Phase portrait sin(z)-sinh(z)")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 7")

# ---- Plot 8: 12.8 - Smoke ring contour with rank ----
try:
    ff = lambda x, y: jnp.exp(-40*(x**2 - x*y + 2*y**2 - 0.5)**2)
    f = chebfun2(ff)
    fig, ax = contour(f, title=f'rank {f.rank}', filled=False,
                      levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    save(fig, "Smoke ring contour")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 8")

# ---- Plot 9: 12.8 - Low-rank approximations grid (ranks 1-9) ----
try:
    ff_np = lambda x, y: np.exp(-40*(x**2 - x*y + 2*y**2 - 0.5)**2)
    levels = [0.2, 0.4, 0.6, 0.8]
    fig = plt.figure(figsize=(9, 9))
    xs = np.linspace(-1, 1, 200)
    ys = np.linspace(-1, 1, 200)
    XX, YY = np.meshgrid(xs, ys)
    ZZ_exact = ff_np(XX, YY)
    for k in range(1, 10):
        ax = fig.add_axes([0.03 + 0.33*((k-1) % 3),
                           0.67 - 0.30*((k-1) // 3),
                           0.28, 0.28])
        # Build rank-k approximation using SVD of sampled values
        n_sample = 100
        xs_s = np.linspace(-1, 1, n_sample)
        ys_s = np.linspace(-1, 1, n_sample)
        XXs, YYs = np.meshgrid(xs_s, ys_s)
        ZZs = ff_np(XXs, YYs)
        U, S, Vt = np.linalg.svd(ZZs, full_matrices=False)
        # Rank-k approximation
        ZZ_k_sample = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        # Interpolate to plotting grid
        from scipy.interpolate import RectBivariateSpline
        interp = RectBivariateSpline(ys_s, xs_s, ZZ_k_sample)
        ZZ_k = interp(ys, xs)
        ax.contour(XX, YY, ZZ_k, levels=levels, colors='k', linewidths=0.8)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'rank {k}', fontsize=9)
    save(fig, "Low-rank approximation grid")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 9")

print(f"\nGuide 12: Generated {plot_num} plots.")
