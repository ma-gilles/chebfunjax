"""Generate all 16 plots for Guide Chapter 18 (Chebfun3).

Figure order follows https://www.chebfun.org/docs/guide/guide18.html
(see /scratch/.../chebfunjax_audit_20260702/guide18_figure_map.md for
the block-by-block mapping); each file is saved at the reference size
(600x270).
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.chebfun3d import chebfun3
from chebfunjax.plotting import (
    CHEBFUN_BLUE,
    PARULA,
    _setup_3d_axes,
    chebfun_style,
    save_chebfun_figure,
)

chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images", "guide")
os.makedirs(OUT, exist_ok=True)
plot_num = 0


def save(fig, desc=""):
    global plot_num
    plot_num += 1
    fname = os.path.join(OUT, f"guide18_{plot_num:02d}.png")
    save_chebfun_figure(fig, fname, size=(600, 270))
    plt.close(fig)
    print(f"  guide18_{plot_num:02d}.png: {desc}")


def _grid_vals(f3, n=60, dom=(-1, 1, -1, 1, -1, 1)):
    xa, xb, ya, yb, za, zb = dom
    xs = np.linspace(xa, xb, n)
    ys = np.linspace(ya, yb, n)
    zs = np.linspace(za, zb, n)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    VV = np.asarray(
        f3(jnp.asarray(XX.ravel()), jnp.asarray(YY.ravel()),
           jnp.asarray(ZZ.ravel()))
    ).reshape(XX.shape)
    return xs, ys, zs, VV


def _isosurface(ax, VV, level, color, dom=(-1, 1, -1, 1, -1, 1),
                alpha=0.8, n=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    if n is None:
        n = VV.shape[0]
    xa, xb, ya, yb, za, zb = dom
    verts, faces, _, _ = marching_cubes(VV, level=level)
    verts[:, 0] = verts[:, 0] / (n - 1) * (xb - xa) + xa
    verts[:, 1] = verts[:, 1] / (n - 1) * (yb - ya) + ya
    verts[:, 2] = verts[:, 2] / (n - 1) * (zb - za) + za
    mesh = Poly3DCollection(verts[faces], alpha=alpha, linewidth=0)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)
    ax.set_xlim(xa, xb)
    ax.set_ylim(ya, yb)
    ax.set_zlim(za, zb)


def _slices_render(ax, f3, n=70):
    """Three orthogonal mid-slices, MATLAB slice(f) style."""
    xs = np.linspace(-1, 1, n)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    flat = jnp.asarray(XX.ravel()), jnp.asarray(YY.ravel())
    zeros = jnp.zeros(n * n)
    import matplotlib.colors as mcolors

    Fz = np.asarray(f3(flat[0], flat[1], zeros)).reshape(n, n)
    Fy = np.asarray(f3(flat[0], zeros, flat[1])).reshape(n, n)
    Fx = np.asarray(f3(zeros, flat[0], flat[1])).reshape(n, n)
    vmin = min(F.min() for F in (Fx, Fy, Fz))
    vmax = max(F.max() for F in (Fx, Fy, Fz))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    Z0 = np.zeros_like(XX)
    ax.plot_surface(XX, YY, Z0, facecolors=PARULA(norm(Fz)), rstride=1,
                    cstride=1, linewidth=0, shade=False)
    ax.plot_surface(XX, Z0, YY, facecolors=PARULA(norm(Fy)), rstride=1,
                    cstride=1, linewidth=0, shade=False)
    ax.plot_surface(Z0, XX, YY, facecolors=PARULA(norm(Fx)), rstride=1,
                    cstride=1, linewidth=0, shade=False)
    return norm


def _plotcoeffs_3panel(fig, f3, xlab="Degree of Chebyshev polynomial"):
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, (name, title) in enumerate(
            [("cols", "Cols"), ("rows", "Rows"), ("tubes", "Tubes")]):
        ax = fig.add_subplot(1, 3, k + 1)
        for i, t in enumerate(getattr(f3, name)):
            c = np.abs(np.asarray(t.coeffs))
            ax.semilogy(np.arange(len(c)), np.maximum(c, 1e-300), ".-",
                        color=cyc[i % len(cyc)], markersize=3,
                        linewidth=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(1e-20, 10)
        if k == 0:
            ax.set_ylabel("Magnitude of coefficient", fontsize=8)
        if k == 1:
            ax.set_xlabel(xlab, fontsize=8)
        ax.tick_params(labelsize=7)


# Fig 1: slice(f) of f = cos(xyz)-ish demo function [block 6]
try:
    f = chebfun3(lambda x, y, z: jnp.sin(8 * (x + y / 2 + z / 3)) / 2
                 + jnp.cos(5 * (x * y * z)))
    fig = plt.figure()
    ax = fig.add_axes([0.1, -0.05, 0.75, 1.05], projection="3d")
    ax.view_init(elev=30, azim=-127.5)
    norm = _slices_render(ax, f)
    m = plt.cm.ScalarMappable(norm=norm, cmap=PARULA)
    fig.colorbar(m, ax=ax, fraction=0.04, pad=0.06)
    save(fig, "slice(f)")
except Exception as e:  # noqa: BLE001
    plot_num += 1
    print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Fig 2: isosurface(f) [block 7]
try:
    _, _, _, VV = _grid_vals(f, n=50)
    fig, ax = _setup_3d_axes(None, None)
    _isosurface(ax, VV, 0.62, (0.55, 0.05, 0.05), alpha=1.0, n=50)
    save(fig, "isosurface(f)")
except Exception as e:  # noqa: BLE001
    plot_num += 1
    print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Fig 3: plotcoeffs(f, '.-') for f = sin(x+yz), g companion [blocks 15-16]
try:
    f3a = chebfun3(lambda x, y, z: jnp.sin(x + y * z))
    fig = plt.figure()
    _plotcoeffs_3panel(fig, f3a)
    fig.subplots_adjust(left=0.1, right=0.97, wspace=0.35, bottom=0.16)
    save(fig, "plotcoeffs(f)")
except Exception as e:  # noqa: BLE001
    plot_num += 1
    print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# g companion used in figs 4-5 [block 16]
g3 = None
try:
    g3 = chebfun3(lambda x, y, z: jnp.cos(15 * jnp.exp(z))
                  / (5 + x**3 + 2 * y**2 + z))
except Exception as e:  # noqa: BLE001
    print(f"  g3 construction FAILED: {e}")

# Fig 4: contourf(sum(f), 20) with colorbar [block 20]
try:
    s_yz = f3a.sum(dim=1)  # integrate over x -> chebfun2(y, z)
    ys = np.linspace(-1, 1, 300)
    YY, ZZ = np.meshgrid(ys, ys, indexing="ij")
    S = np.asarray(s_yz(jnp.asarray(YY.ravel()),
                        jnp.asarray(ZZ.ravel()))).reshape(YY.shape)
    fig, ax = plt.subplots()
    cs = ax.contourf(YY, ZZ, S, levels=20, cmap=PARULA)
    ax.contour(YY, ZZ, S, levels=cs.levels, colors="k", linewidths=0.5)
    fig.colorbar(cs, ax=ax, fraction=0.045)
    save(fig, "contourf(sum(f))")
except Exception as e:  # noqa: BLE001
    plot_num += 1
    print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Fig 5: plot(sum2(exp(g+2f))) [block 21]
try:
    h3 = chebfun3(lambda x, y, z: jnp.exp(
        jnp.cos(15 * jnp.exp(z)) / (5 + x**3 + 2 * y**2 + z)
        + 2 * jnp.sin(x + y * z)))
    s_z = h3.sum2(dims=(1, 2))  # 1D chebfun of z
    zs = jnp.linspace(-1.0, 1.0, 600)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(zs), np.asarray(s_z(zs)), color=CHEBFUN_BLUE,
            linewidth=1.4)
    save(fig, "plot(sum2(exp(g+2f)))")
except Exception as e:  # noqa: BLE001
    plot_num += 1
    print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Fig 6: helix [block 22]
try:
    t8pi = 8 * float(np.pi)
    ts = np.linspace(0.0, t8pi, 1500)
    fig, ax = _setup_3d_axes(None, None)
    ax.plot3D(np.cos(ts), np.sin(ts), ts / t8pi, color=CHEBFUN_BLUE,
              linewidth=1.2)
    ax.set_title("Helix")
    save(fig, "helix")
except Exception as e:  # noqa: BLE001
    plot_num += 1
    print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Figs 7-8: plot(g.cols), plot(g.tubes) [blocks 24-25]
try:
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    xs = np.linspace(-1, 1, 400)
    for name in ("cols", "tubes"):
        fig, ax = plt.subplots()
        for i, t in enumerate(getattr(g3, name)):
            ax.plot(xs, np.asarray(t(jnp.asarray(xs))),
                    color=cyc[i % len(cyc)], linewidth=1.0)
        save(fig, f"plot(g.{name})")
except Exception as e:  # noqa: BLE001
    plot_num += 1
    print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Fig 9: plotcoeffs(g) [block 26]
try:
    fig = plt.figure()
    _plotcoeffs_3panel(fig, g3)
    fig.subplots_adjust(left=0.1, right=0.97, wspace=0.35, bottom=0.16)
    save(fig, "plotcoeffs(g)")
except Exception as e:  # noqa: BLE001
    plot_num += 1
    print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Fig 10: plotcoeffs(g.cols) single panel [block 27]
try:
    fig, ax = plt.subplots()
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, t in enumerate(g3.cols):
        c = np.abs(np.asarray(t.coeffs))
        ax.semilogy(np.arange(len(c)), np.maximum(c, 1e-300), ".-",
                    color=cyc[i % len(cyc)], markersize=4, linewidth=0.8)
    ax.set_title("Chebyshev coefficients", fontsize=10)
    ax.set_xlabel("Degree of Chebyshev polynomial", fontsize=9)
    ax.set_ylabel("Magnitude of coefficient", fontsize=9)
    ax.set_ylim(1e-20, 10)
    save(fig, "plotcoeffs(g.cols)")
except Exception as e:  # noqa: BLE001
    plot_num += 1
    print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Fig 11: plotcoeffs of the trig chebfun3 [blocks 28-29].
# chebfun3 has no 'trig' mode yet; the honest equivalent renders the
# per-direction FOURIER coefficient profiles of the periodic function
# computed directly by FFT along each axis (what the trig factors'
# plotcoeffs displays, up to factor mixing).
try:
    L = float(np.pi)

    def ff18(x, y, z):
        return (np.tanh(3 * np.sin(x)) - np.sin(y + 0.5) ** 2
                + np.cos(6 * z))

    n = 128
    grid = np.linspace(-L, L, n, endpoint=False)
    mids = np.array([0.37, -0.83])  # generic fixed sections
    fig = plt.figure()
    for k, (axis, title) in enumerate(
            [(0, "Cols"), (1, "Rows"), (2, "Tubes")]):
        ax = fig.add_subplot(1, 3, k + 1)
        for a in mids:
            for b in mids:
                if axis == 0:
                    vals = ff18(grid, a, b)
                elif axis == 1:
                    vals = ff18(a, grid, b)
                else:
                    vals = ff18(a, b, grid)
                c = np.abs(np.fft.fftshift(np.fft.fft(vals))) / n
                ks = np.arange(-(n // 2), n // 2)
                ax.semilogy(ks, np.maximum(c, 1e-300), ".-",
                            markersize=3, linewidth=0.5)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(1e-20, 10)
        if k == 0:
            ax.set_ylabel("Magnitude of coefficient", fontsize=8)
        if k == 1:
            ax.set_xlabel("Wave number", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.1, right=0.97, wspace=0.35, bottom=0.16)
    save(fig, "trig coefficient profiles")
except Exception as e:  # noqa: BLE001
    plot_num += 1
    print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Fig 12: quiver of F = grad(f) on the big domain [block 37]
try:
    L5 = 5 * float(np.pi)
    fbig = chebfun3(
        lambda x, y, z: jnp.sin(x + 20 * y + z**2)
        * jnp.exp(-(3 + y**2)),
        domain=(-L5, L5, -L5, L5, -L5, L5))
    fx, fy, fz = fbig.grad()
    m = 8
    gs = np.linspace(-L5, L5, m)
    XX, YY, ZZ = np.meshgrid(gs, gs, gs, indexing="ij")
    flat = (jnp.asarray(XX.ravel()), jnp.asarray(YY.ravel()),
            jnp.asarray(ZZ.ravel()))
    U = np.asarray(fx(*flat)).reshape(XX.shape)
    V = np.asarray(fy(*flat)).reshape(XX.shape)
    W = np.asarray(fz(*flat)).reshape(XX.shape)
    fig, ax = _setup_3d_axes(None, None)
    ax.quiver(XX, YY, ZZ, U, V, W, length=3.0, normalize=True,
              color=CHEBFUN_BLUE, linewidth=0.6)
    save(fig, "quiver(grad(f))")
except Exception as e:  # noqa: BLE001
    plot_num += 1
    print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Figs 13-14: conical spiral, then with the radial red line [block 39]
try:
    t5pi = 5 * float(np.pi)
    ts = np.linspace(0.0, t5pi, 1200)
    cx, cy, cz = ts * np.cos(ts), ts * np.sin(ts), ts
    fig, ax = _setup_3d_axes(None, None)
    ax.plot3D(cx, cy, cz, "b", linewidth=1.2)
    ax.set_title("Conical spiral")
    save(fig, "conical spiral")

    fig, ax = _setup_3d_axes(None, None)
    ax.plot3D(cx, cy, cz, "b", linewidth=1.2)
    r2 = np.linspace(0.0, t5pi, 200)
    ax.plot3D(r2 * np.cos(t5pi), r2 * np.sin(t5pi), r2, "r",
              linewidth=1.4)
    ax.set_title("Conical spiral")
    save(fig, "conical spiral + line")
except Exception as e:  # noqa: BLE001
    plot_num += 1
    print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Fig 15: two zero isosurfaces [block 47]
f15 = g15 = None
try:
    f15 = chebfun3(lambda x, y, z: y - x**2)
    g15 = chebfun3(lambda x, y, z: z - x**3)
    _, _, _, VF = _grid_vals(f15, n=50)
    _, _, _, VG = _grid_vals(g15, n=50)
    fig, ax = _setup_3d_axes(None, None)
    ax.view_init(elev=43, azim=112)
    _isosurface(ax, VF, 0.0, "g", alpha=0.75, n=50)
    _isosurface(ax, VG, 0.0, "b", alpha=0.75, n=50)
    save(fig, "isosurfaces f=0 (g), g=0 (b)")
except Exception as e:  # noqa: BLE001
    plot_num += 1
    print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

# Fig 16: three isosurfaces + common root marker [block 50]
try:
    h16 = chebfun3(lambda x, y, z: jnp.cos(x * y * z) - x - y - z)
    _, _, _, VH = _grid_vals(h16, n=50)
    # Common root of {y=x^2, z=x^3, h=0}: solve h(x, x^2, x^3) = 0
    from scipy.optimize import brentq

    def h_line(x):
        return float(np.cos(x * x**2 * x**3) - x - x**2 - x**3)

    r = brentq(h_line, 0.0, 1.0)
    root = (r, r**2, r**3)
    fig, ax = _setup_3d_axes(None, None)
    ax.view_init(elev=30, azim=100)
    _isosurface(ax, VF, 0.0, "g", alpha=0.6, n=50)
    _isosurface(ax, VG, 0.0, "b", alpha=0.6, n=50)
    _isosurface(ax, VH, 0.0, "r", alpha=0.6, n=50)
    ax.plot([root[0]], [root[1]], [root[2]], marker="*", markersize=16,
            color="y", markeredgecolor="k")
    save(fig, f"three isosurfaces + root {root[0]:.4f}")
except Exception as e:  # noqa: BLE001
    plot_num += 1
    print(f"  guide18_{plot_num:02d}.png FAILED: {e}")

print(f"\nGuide 18: {plot_num} slots processed.")
