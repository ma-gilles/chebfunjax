"""Generate all plots for Guide Chapter 18 (Chebfun3).

Uses PARULA colormap and three-orthogonal-slice 3D plots matching MATLAB.
"""
import matplotlib
matplotlib.use('Agg')
import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from chebfunjax.plotting import (
    chebfun_style, plot_slices, plot_chebfun3, PARULA, _setup_3d_axes,
)
from chebfunjax.chebfun3d import chebfun3, Chebfun3

chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)
plot_num = 0

def save(fig, desc=""):
    global plot_num
    plot_num += 1
    fname = os.path.join(OUT, f'guide18_{plot_num:02d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  guide18_{plot_num:02d}.png: {desc}")

def slices_plot(f, title='', n=80):
    """Three orthogonal slices of a Chebfun3 using plot_chebfun3."""
    fig, ax = plot_chebfun3(f, title=title, n_pts=n)
    return fig, ax

# Plot 01: cos(xyz) slice plot (Section 18.1)
try:
    f = chebfun3(lambda x, y, z: jnp.cos(x * y * z))
    fig, ax = slices_plot(f, 'cos(xyz)')
    save(fig, "cos(xyz)")
except Exception as e:
    plot_num += 1; print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Plot 02: slice of 1/(1+x^2+y^2+z^2) (Section 18.2)
try:
    f2 = chebfun3(lambda x, y, z: 1.0 / (1.0 + x**2 + y**2 + z**2))
    fig, ax = slices_plot(f2, '1/(1+x^2+y^2+z^2)')
    save(fig, "Runge 3D")
except Exception as e:
    plot_num += 1; print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Plot 03: exp(x+y+z) (Section 18.3)
try:
    f3 = chebfun3(lambda x, y, z: jnp.exp(x + y + z))
    fig, ax = slices_plot(f3, 'exp(x+y+z)')
    save(fig, "exp(x+y+z)")
except Exception as e:
    plot_num += 1; print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Plot 04: z-slices gallery (Section 18.4)
try:
    n = 80
    xs = np.linspace(-1, 1, n); ys = np.linspace(-1, 1, n)
    XX, YY = np.meshgrid(xs, ys)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, zv in enumerate([-0.5, 0.0, 0.5]):
        ZM = np.full_like(XX, zv)
        FF = np.array(f2(jnp.array(XX.ravel()), jnp.array(YY.ravel()),
                         jnp.array(ZM.ravel()))).reshape(n, n)
        cs = axes[i].contourf(XX, YY, FF, levels=15, cmap=PARULA)
        fig.colorbar(cs, ax=axes[i], fraction=0.046, pad=0.04)
        axes[i].set_title(f'z = {zv}', fontsize=10)
        axes[i].set_xlabel('x', fontsize=9)
        axes[i].set_ylabel('y', fontsize=9)
        axes[i].set_aspect('equal')
    fig.suptitle('1/(1+x^2+y^2+z^2) slices', fontsize=11)
    fig.set_facecolor('white')
    fig.tight_layout()
    save(fig, "z-slices")
except Exception as e:
    plot_num += 1; print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Plot 05: sin(x+y+z) on [0,2]^3 (Section 18.5)
try:
    f5 = chebfun3(lambda x, y, z: jnp.sin(x + y + z),
                  domain=(0., 2., 0., 2., 0., 2.))
    fig, ax = slices_plot(f5, 'sin(x+y+z) on [0,2]^3')
    save(fig, "sin on [0,2]^3")
except Exception as e:
    plot_num += 1; print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Plot 06: scan plot showing column/row/tube functions (Section 18.6)
try:
    f6 = chebfun3(lambda x, y, z: jnp.cos(x * y * z))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    t = jnp.linspace(-1, 1, 200)
    # Columns (x)
    for j in range(min(3, len(f6.cols))):
        axes[0].plot(np.array(t), np.array(f6.cols[j](t)), linewidth=1.2)
    axes[0].set_title(f'Column functions (x), rank={len(f6.cols)}', fontsize=10)
    # Rows (y)
    for j in range(min(3, len(f6.rows))):
        axes[1].plot(np.array(t), np.array(f6.rows[j](t)), linewidth=1.2)
    axes[1].set_title(f'Row functions (y), rank={len(f6.rows)}', fontsize=10)
    # Tubes (z)
    for j in range(min(3, len(f6.tubes))):
        axes[2].plot(np.array(t), np.array(f6.tubes[j](t)), linewidth=1.2)
    axes[2].set_title(f'Tube functions (z), rank={len(f6.tubes)}', fontsize=10)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig, "Tucker factors")
except Exception as e:
    plot_num += 1; print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Plot 07: isosurface-like (Section 18.7)
try:
    n = 40
    xs = np.linspace(-1, 1, n); ys = np.linspace(-1, 1, n); zs = np.linspace(-1, 1, n)
    XX3, YY3, ZZ3 = np.meshgrid(xs, ys, zs, indexing='ij')
    VV = np.array(f2(jnp.array(XX3.ravel()), jnp.array(YY3.ravel()),
                      jnp.array(ZZ3.ravel()))).reshape(XX3.shape)
    fig, ax = _setup_3d_axes(None, None, elev=25, azim=-37, figsize=(6.1, 5.0))

    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from skimage.measure import marching_cubes
        verts, faces, _, _ = marching_cubes(VV, level=0.5)
        verts = verts / (n-1) * 2 - 1  # map to [-1,1]
        mesh = Poly3DCollection(verts[faces], alpha=0.3, edgecolor='k', linewidth=0.1)
        mesh.set_facecolor(PARULA(0.5))
        ax.add_collection3d(mesh)
    except ImportError:
        # Fallback: show three orthogonal contour slices as coloured surfaces
        import matplotlib.colors as mcolors
        mid = n // 2
        norm = mcolors.Normalize(vmin=float(VV.min()), vmax=float(VV.max()))
        # z=0 slice
        Xp, Yp = np.meshgrid(xs, ys, indexing='ij')
        Zp = np.zeros_like(Xp)
        Fp = VV[:, :, mid]
        ax.plot_surface(Xp, Yp, Zp, facecolors=PARULA(norm(Fp)),
                        rstride=1, cstride=1, linewidth=0, alpha=0.7, shade=False)
        # y=0 slice
        Xp2, Zp2 = np.meshgrid(xs, zs, indexing='ij')
        Yp2 = np.zeros_like(Xp2)
        Fp2 = VV[:, mid, :]
        ax.plot_surface(Xp2, Yp2, Zp2, facecolors=PARULA(norm(Fp2)),
                        rstride=1, cstride=1, linewidth=0, alpha=0.7, shade=False)
        # x=0 slice
        Yp3, Zp3 = np.meshgrid(ys, zs, indexing='ij')
        Xp3 = np.zeros_like(Yp3)
        Fp3 = VV[mid, :, :]
        ax.plot_surface(Xp3, Yp3, Zp3, facecolors=PARULA(norm(Fp3)),
                        rstride=1, cstride=1, linewidth=0, alpha=0.7, shade=False)

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_title('Isosurface at 0.5', fontsize=10, pad=0)
    save(fig, "isosurface")
except Exception as e:
    plot_num += 1; print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

print(f"\nGuide 18: {plot_num} plots generated.")
