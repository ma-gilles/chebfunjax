"""Generate plots for Guide Chapter 14: Chebfun2: Rootfinding and Optimisation.

Faithful translation of all figures from the original MATLAB Chebfun Guide
Chapter 14 (https://www.chebfun.org/docs/guide/guide14.html).
"""
import matplotlib

matplotlib.use('Agg')
import os
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax.numpy as jnp

from chebfunjax.chebfun2d import chebfun2
from chebfunjax.plotting import PARULA, chebfun_style, surf

chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)

plot_num = 0

def save(fig, desc, size=(610, 258)):
    global plot_num
    plot_num += 1
    fname = os.path.join(OUT, f'guide14_{plot_num:02d}.png')
    from chebfunjax.plotting import save_chebfun_figure
    save_chebfun_figure(fig, fname, size=size)
    plt.close(fig)
    print(f"  Saved {fname}: {desc}")


def eval_on_grid(f, xa, xb, ya, yb, n=300):
    """Evaluate a chebfun2 on a grid, returning XX, YY, ZZ."""
    xs = np.linspace(xa, xb, n)
    ys = np.linspace(ya, yb, n)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.array(f(jnp.array(XX.ravel()), jnp.array(YY.ravel()))).reshape(n, n)
    return XX, YY, ZZ


# ---- Plot 1: 14.1 - Trott's curve zero contours ----
try:
    # Trott's curve: 144*(x^4+y^4) - 225*(x^2+y^2) + 350*x^2*y^2 + 81
    trott = chebfun2(lambda x, y:
        144*(x**4 + y**4) - 225*(x**2 + y**2) + 350*x**2*y**2 + 81)
    XX, YY, ZZ = eval_on_grid(trott, -1, 1, -1, 1, n=500)
    fig, ax = plt.subplots()
    ax.contour(XX, YY, ZZ, levels=[0.0], colors='b', linewidths=1.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title("Trott's curve", fontsize=11)
    fig.tight_layout()
    save(fig, "Trott's curve")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 1")

# ---- Plot 2: 14.2 - Trott + circle intersections ----
try:
    trott = chebfun2(lambda x, y:
        144*(x**4 + y**4) - 225*(x**2 + y**2) + 350*x**2*y**2 + 81)
    g = chebfun2(lambda x, y: x**2 + y**2 - 0.9**2)
    XX, YY, ZZ_t = eval_on_grid(trott, -1, 1, -1, 1, n=500)
    _, _, ZZ_g = eval_on_grid(g, -1, 1, -1, 1, n=500)

    # Known intersection points (from MATLAB output)
    r = np.array([
        [-0.799441089368585, -0.413393208252350],
        [-0.799441089368583,  0.413393208252352],
        [-0.413393208252347, -0.799441089368586],
        [-0.413393208252346,  0.799441089368587],
        [ 0.413393208252345, -0.799441089368587],
        [ 0.413393208252344,  0.799441089368588],
        [ 0.799441089368588, -0.413393208252343],
        [ 0.799441089368587,  0.413393208252346],
    ])

    fig, ax = plt.subplots()
    ax.contour(XX, YY, ZZ_t, levels=[0.0], colors='b', linewidths=1.5)
    ax.contour(XX, YY, ZZ_g, levels=[0.0], colors='r', linewidths=1.5)
    ax.plot(r[:, 0], r[:, 1], 'k.', markersize=15)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title("Trott's curve (blue) and circle $r=0.9$ (red)", fontsize=10)
    fig.tight_layout()
    save(fig, "Trott + circle intersections")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 3")

# ---- Plot 3: 14.3 - Splat and figure-of-eight curve intersections ----
try:
    t = np.linspace(0, 2*np.pi, 1000)
    sp_x = np.cos(t) + (1+0)*np.sin(6*t)**2
    sp_y = np.sin(t) + (0+1)*np.sin(6*t)**2
    # splat: exp(it) + (1+i)*sin(6t)^2
    sp = np.exp(1j*t) + (1+1j)*np.sin(6*t)**2
    sp_x = np.real(sp)
    sp_y = np.imag(sp)
    # figure of eight: cos(t) + i*sin(2t)
    f8_x = np.cos(t)
    f8_y = np.sin(2*t)

    fig, ax = plt.subplots()
    ax.plot(sp_x, sp_y, 'b-', linewidth=1.5, label='Splat')
    ax.plot(f8_x, f8_y, 'r-', linewidth=1.5, label='Figure-of-eight')

    # Find approximate intersections by sampling
    # This is a simplified version; exact intersections need bivariate rootfinding
    n_fine = 5000
    t_fine = np.linspace(0, 2*np.pi, n_fine)
    sp_fine = np.exp(1j*t_fine) + (1+1j)*np.sin(6*t_fine)**2
    f8_fine = np.cos(t_fine) + 1j*np.sin(2*t_fine)
    # Brute force: find closest pairs
    from scipy.spatial import cKDTree
    sp_pts = np.column_stack([np.real(sp_fine), np.imag(sp_fine)])
    f8_pts = np.column_stack([np.real(f8_fine), np.imag(f8_fine)])
    tree = cKDTree(f8_pts)
    dists, idxs = tree.query(sp_pts, k=1)
    # Find local minima in distance
    crossings = []
    for i in range(1, len(dists)-1):
        if dists[i] < 0.02 and dists[i] < dists[i-1] and dists[i] < dists[i+1]:
            crossings.append(sp_pts[i])
    if crossings:
        crossings = np.array(crossings)
        ax.plot(crossings[:, 0], crossings[:, 1], 'k.', markersize=15)

    ax.set_aspect('equal')
    ax.set_ylim(-1.1, 2.1)
    ax.legend(fontsize=9)
    ax.set_title('Splat and figure-of-eight intersections', fontsize=10)
    fig.tight_layout()
    save(fig, "Splat and figure-of-eight")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 4")

# ---- Plot 4: 14.4 - Global optimisation with min/max dots ----
try:
    f = chebfun2(lambda x, y:
        jnp.sin(30*x*y) + jnp.sin(10*y*x**2) + jnp.exp(-x**2-(y-0.8)**2))
    fig, ax = surf(f, cmap='bone')
    # Find min/max on grid
    n = 400
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.array(f(jnp.array(XX.ravel()), jnp.array(YY.ravel()))).reshape(n, n)
    idx_min = np.unravel_index(np.argmin(ZZ), ZZ.shape)
    idx_max = np.unravel_index(np.argmax(ZZ), ZZ.shape)
    mn = ZZ[idx_min]
    mx = ZZ[idx_max]
    mnloc = (xs[idx_min[1]], ys[idx_min[0]])
    mxloc = (xs[idx_max[1]], ys[idx_max[0]])
    ax.scatter([mnloc[0]], [mnloc[1]], [mn], c='r', s=100, zorder=5)
    ax.scatter([mxloc[0]], [mxloc[1]], [mx], c='r', s=80, zorder=5)
    ax.set_zlim(-6, 6)
    ax.set_title('Global min/max', fontsize=11)
    save(fig, "Global optimisation surface")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 5")

# ---- Plot 5: 14.4 - Challenge function contour with minimum ----
try:
    # SIAM 100-digit challenge function
    def challenge_fn(x, y):
        return (jnp.exp(jnp.sin(50*x)) + jnp.sin(60*jnp.exp(y))
                + jnp.sin(70*jnp.sin(x)) + jnp.sin(jnp.sin(80*y))
                - jnp.sin(10*(x+y)) + (x**2 + y**2)/4)
    f_ch = chebfun2(challenge_fn)
    # MATLAB: colormap('default'), contour(f), full-width axes, no title,
    # minimum marked with an UNFILLED circle 'ok'.
    XX, YY, ZZ = eval_on_grid(f_ch, -1, 1, -1, 1, n=700)
    fig, ax = plt.subplots()
    ax.contour(XX, YY, ZZ, levels=10, cmap=PARULA,
               linewidths=0.9)
    idx_min = np.unravel_index(np.argmin(ZZ), ZZ.shape)
    minpos = (XX[0, idx_min[1]], YY[idx_min[0], 0])
    ax.plot(minpos[0], minpos[1], 'o', markerfacecolor='none',
            markeredgecolor='k', markersize=11, markeredgewidth=1.2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    save(fig, "Challenge function contour")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 6")

# ---- Plot 6: 14.5 - Critical points ----
def _critical_points_figure():
    """Zero contours of f_x (blue) and f_y (red) with critical points."""
    f = chebfun2(lambda x, y: (x**2 - y**3 + 1.0/8)*jnp.sin(10*x*y))
    fx = f.diff(dim=2)
    fy = f.diff(dim=1)
    XX, YY, ZZ_fx = eval_on_grid(fx, -1, 1, -1, 1, n=400)
    _, _, ZZ_fy = eval_on_grid(fy, -1, 1, -1, 1, n=400)

    fig, ax = plt.subplots()
    ax.contour(XX, YY, ZZ_fx, levels=[0.0], colors='b', linewidths=0.8)
    ax.contour(XX, YY, ZZ_fy, levels=[0.0], colors='r', linewidths=0.8)

    # Critical points: cells where both partials are near zero, clustered
    mask = (np.abs(ZZ_fx) < 0.1 * np.max(np.abs(ZZ_fx))) & \
           (np.abs(ZZ_fy) < 0.1 * np.max(np.abs(ZZ_fy)))
    crit_x = XX[mask]
    crit_y = YY[mask]
    if len(crit_x) > 1:
        from scipy.cluster.hierarchy import fclusterdata
        pts = np.column_stack([crit_x, crit_y])
        clusters = fclusterdata(pts, t=0.1, criterion='distance')
        centers = np.array([
            [np.mean(pts[clusters == c, 0]), np.mean(pts[clusters == c, 1])]
            for c in np.unique(clusters)
        ])
        ax.plot(centers[:, 0], centers[:, 1], 'k.', markersize=15)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title('Critical points: $f_x=0$ (blue), $f_y=0$ (red)',
                 fontsize=10)
    return fig, ax


try:
    fig, ax = _critical_points_figure()
    save(fig, "Critical points")
except Exception:
    plot_num += 1
    traceback.print_exc()

# ---- Plot 7: 14.5 - Critical points (large render, as published) ----
try:
    fig, ax = _critical_points_figure()
    save(fig, "Critical points (large)", size=(814, 785))
except Exception:
    traceback.print_exc()
    print("  SKIP plot 7")

print(f"\nGuide 14: Generated {plot_num} plots.")
