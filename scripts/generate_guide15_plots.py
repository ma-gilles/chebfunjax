"""Generate plots for Guide Chapter 15: Chebfun2: Vector Calculus and 2D Surfaces.

Faithful translation of all figures from the original MATLAB Chebfun Guide
Chapter 15 (https://www.chebfun.org/docs/guide/guide15.html).
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
from chebfunjax.plotting import (
    chebfun_style, surf, contour, CHEBFUN_BLUE, PARULA, _setup_3d_axes,
)
from chebfunjax.chebfun2d import chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)

plot_num = 0

def save(fig, desc):
    global plot_num
    plot_num += 1
    fname = os.path.join(OUT, f'guide15_{plot_num:02d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}: {desc}")


# ---- Plot 1: 15.2 - Orthogonality curve: dot(F,G) = 0 ----
try:
    d = (0.0, 1.0, 0.0, 2.0)
    F = Chebfun2v.from_functions(
        lambda x, y: jnp.sin(x*y),
        lambda x, y: jnp.cos(y),
        domain=d)
    G = Chebfun2v.from_functions(
        lambda x, y: jnp.cos(4*x*y),
        lambda x, y: x + x*y**2,
        domain=d)
    dot_fg = F.dot(G)
    # Plot roots of dot(F,G)
    n = 400
    xs = np.linspace(0, 1, n)
    ys = np.linspace(0, 2, n)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.array(dot_fg(jnp.array(XX.ravel()), jnp.array(YY.ravel()))).reshape(n, n)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contour(XX, YY, ZZ, levels=[0.0], colors=CHEBFUN_BLUE, linewidths=1.5)
    ax.set_xlim(d[0], d[1])
    ax.set_ylim(d[2], d[3])
    ax.set_aspect('equal')
    ax.set_title(r'$\mathbf{F}\cdot\mathbf{G} = 0$: orthogonality curve', fontsize=10)
    fig.set_facecolor('white')
    fig.tight_layout()
    save(fig, "dot(F,G)=0 orthogonality curve")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 1")

# ---- Plot 2: 15.3 - Gradient critical points (sum of Gaussian bumps) ----
try:
    rng = np.random.RandomState(0)
    bump_centers = []
    for k in range(10):
        x0 = 2*rng.rand() - 1
        y0 = 2*rng.rand() - 1
        bump_centers.append((x0, y0))

    def bump_sum(x, y):
        result = jnp.zeros_like(x)
        for x0, y0 in bump_centers:
            result = result + jnp.exp(-10*((x - x0)**2 + (y - y0)**2))
        return result

    f_bump = chebfun2(bump_sum)
    fx = f_bump.diff(dim=2)
    fy = f_bump.diff(dim=1)

    n = 200
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.array(f_bump(jnp.array(XX.ravel()), jnp.array(YY.ravel()))).reshape(n, n)
    ZZ_fx = np.array(fx(jnp.array(XX.ravel()), jnp.array(YY.ravel()))).reshape(n, n)
    ZZ_fy = np.array(fy(jnp.array(XX.ravel()), jnp.array(YY.ravel()))).reshape(n, n)

    # Approximate critical points
    mask = (np.abs(ZZ_fx) < 0.02 * np.max(np.abs(ZZ_fx))) & \
           (np.abs(ZZ_fy) < 0.02 * np.max(np.abs(ZZ_fy)))
    crit_x = XX[mask]
    crit_y = YY[mask]

    fig, ax = _setup_3d_axes(None, None, elev=30, azim=-50, figsize=(6.1, 5.0))
    ax.plot_surface(XX, YY, ZZ, cmap=PARULA, linewidth=0,
                    antialiased=True, alpha=0.85, shade=True)

    if len(crit_x) > 0:
        from scipy.cluster.hierarchy import fclusterdata
        pts = np.column_stack([crit_x, crit_y])
        if len(pts) > 1:
            clusters = fclusterdata(pts, t=0.08, criterion='distance')
            for c in np.unique(clusters):
                mask_c = clusters == c
                cx = np.mean(pts[mask_c, 0])
                cy = np.mean(pts[mask_c, 1])
                cz = float(f_bump(jnp.float64(cx), jnp.float64(cy)))
                ax.scatter([cx], [cy], [cz], c='k', s=60, zorder=5)

    ax.set_zlim(0, 4)
    ax.set_title('Sum of Gaussian bumps with critical points', fontsize=10, pad=0)
    save(fig, "Gradient critical points")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 2")

# ---- Plot 3: 15.4 - Line integral (gradient theorem) ----
try:
    f = chebfun2(lambda x, y: jnp.cos(10*x*y**2) + jnp.exp(-x**2))
    # Curve C = t*exp(10it), t in [0,1]
    t = np.linspace(0, 1, 500)
    cx = t * np.cos(10*t)
    cy = t * np.sin(10*t)
    cz = np.array([float(f(jnp.float64(xi), jnp.float64(yi))) for xi, yi in zip(cx, cy)])

    fig, ax = _setup_3d_axes(None, None, elev=25, azim=-50, figsize=(6.1, 5.0))
    n = 80
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.array(f(jnp.array(XX.ravel()), jnp.array(YY.ravel()))).reshape(n, n)
    ax.plot_surface(XX, YY, ZZ, cmap=PARULA, linewidth=0,
                    antialiased=True, alpha=0.6, shade=True)
    ax.plot3D(cx, cy, cz, 'k-', linewidth=2)
    ax.set_title(r'$f$ and $F=\nabla f$ along $C$', fontsize=10, pad=0)
    save(fig, "Line integral / gradient theorem")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 3")

# ---- Plot 4: 15.5 - Duffing oscillator phase diagram ----
try:
    delta = 0.04
    a = 1
    b = -0.75
    n_quiv = 20
    xs = np.linspace(-2, 2, n_quiv)
    ys = np.linspace(-2, 2, n_quiv)
    XX, YY = np.meshgrid(xs, ys)
    UU = YY
    VV = -delta*YY - b*XX - a*XX**3

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.quiver(XX, YY, UU, VV, color=CHEBFUN_BLUE, alpha=0.6, scale=40)

    # Solve ODE trajectory with simple Euler method
    dt = 0.01
    n_steps = 4000
    trajectory_x = [0.0]
    trajectory_y = [0.5]
    for _ in range(n_steps):
        x_cur = trajectory_x[-1]
        y_cur = trajectory_y[-1]
        trajectory_x.append(x_cur + dt * y_cur)
        trajectory_y.append(y_cur + dt * (-delta*y_cur - b*x_cur - a*x_cur**3))
    ax.plot(trajectory_x, trajectory_y, 'r-', linewidth=1.0)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('The Duffing oscillator', fontsize=11)
    fig.set_facecolor('white')
    fig.tight_layout()
    save(fig, "Duffing oscillator")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 4")

# ---- Plot 5: 15.6 - Unit sphere via spherical coordinates ----
try:
    n = 60
    th = np.linspace(0, np.pi, n)
    phi = np.linspace(0, 2*np.pi, n)
    TH, PHI = np.meshgrid(th, phi)
    XX = np.sin(TH) * np.cos(PHI)
    YY = np.sin(TH) * np.sin(PHI)
    ZZ = np.cos(TH)
    # Colour by z-coordinate (like MATLAB default)
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=-1, vmax=1)
    fcolors = PARULA(norm(ZZ))

    fig, ax = _setup_3d_axes(None, None, elev=20, azim=-30, figsize=(6.1, 5.0))
    ax.plot_surface(XX, YY, ZZ, facecolors=fcolors, linewidth=0,
                    antialiased=True, alpha=0.9, shade=False)
    ax.set_title('Unit sphere', fontsize=11, pad=0)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    save(fig, "Unit sphere")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 5")

# ---- Plot 6: 15.6 - Cylinder ----
try:
    h = 5
    n = 60
    th = np.linspace(0, 2*np.pi, n)
    z = np.linspace(0, h, n)
    TH, ZZ = np.meshgrid(th, z)
    XX = np.cos(TH)
    YY = np.sin(TH)
    # Colour by height
    norm = Normalize(vmin=0, vmax=h)
    fcolors = PARULA(norm(ZZ))

    fig, ax = _setup_3d_axes(None, None, elev=20, azim=-30, figsize=(5.5, 5.5))
    ax.plot_surface(XX, YY, ZZ, facecolors=fcolors, linewidth=0,
                    antialiased=True, alpha=0.9, shade=False)
    ax.set_title('Cylinder', fontsize=11, pad=0)
    save(fig, "Cylinder")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 6")

# ---- Plot 7: 15.6 - Surface of revolution ----
try:
    n = 80
    t_vals = np.linspace(0, 5, n)
    r_vals = (np.sin(np.pi*t_vals) + 1.1) * t_vals * (t_vals - 10)
    th = np.linspace(0, 2*np.pi, n)
    TH, T = np.meshgrid(th, t_vals)
    _, R = np.meshgrid(th, r_vals)
    XX = R * np.cos(TH)
    YY = R * np.sin(TH)
    ZZ = T
    # Colour by height
    norm = Normalize(vmin=float(ZZ.min()), vmax=float(ZZ.max()))
    fcolors = PARULA(norm(ZZ))

    fig, ax = _setup_3d_axes(None, None, elev=20, azim=-30, figsize=(6.1, 5.0))
    ax.plot_surface(XX, YY, ZZ, facecolors=fcolors, linewidth=0,
                    antialiased=True, alpha=0.9, shade=False)
    ax.set_xlim(-70, 70)
    ax.set_ylim(-70, 70)
    ax.set_zlim(-2, 6)
    ax.set_title('Surface of revolution', fontsize=11, pad=0)
    save(fig, "Surface of revolution")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 7")

# ---- Plot 8: 15.6 - Torus with gap ----
try:
    n = 80
    x_ref = np.linspace(-1, 1, n)
    y_ref = np.linspace(-1, 1, n)
    X_ref, Y_ref = np.meshgrid(x_ref, y_ref)
    theta = 0.9 * np.pi * X_ref
    phi = np.pi * Y_ref
    XX = -(1 + 0.3*np.cos(phi)) * np.sin(theta)
    YY = (1 + 0.3*np.cos(phi)) * np.cos(theta)
    ZZ = 0.3 * np.sin(phi)
    # Colour by phi (tube angle)
    norm = Normalize(vmin=float(phi.min()), vmax=float(phi.max()))
    fcolors = PARULA(norm(phi))

    fig, ax = _setup_3d_axes(None, None, elev=20, azim=-40, figsize=(6.1, 5.0))
    ax.plot_surface(XX, YY, ZZ, facecolors=fcolors, linewidth=0,
                    antialiased=True, alpha=0.9, shade=False)
    ax.set_title('Torus with gap', fontsize=11, pad=0)
    save(fig, "Torus with gap")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 8")

# ---- Plot 9: 15.7 - Torus with normal vectors ----
try:
    r1 = 1
    r2 = 1.0/3.0
    n = 40
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, 2*np.pi, n)
    UU, VV = np.meshgrid(u, v)
    XX = -(r1 + r2*np.cos(VV)) * np.sin(UU)
    YY = (r1 + r2*np.cos(VV)) * np.cos(UU)
    ZZ = r2 * np.sin(VV)

    # Compute normal vectors
    dXdu = -(r1 + r2*np.cos(VV)) * np.cos(UU)
    dYdu = -(r1 + r2*np.cos(VV)) * np.sin(UU)
    dZdu = np.zeros_like(UU)
    dXdv = r2*np.sin(VV) * np.sin(UU)
    dYdv = -r2*np.sin(VV) * np.cos(UU)
    dZdv = r2*np.cos(VV)
    Nx = dYdu*dZdv - dZdu*dYdv
    Ny = dZdu*dXdv - dXdu*dZdv
    Nz = dXdu*dYdv - dYdu*dXdv
    Nmag = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    Nmag[Nmag == 0] = 1
    Nx /= Nmag; Ny /= Nmag; Nz /= Nmag

    # Colour by tube angle
    norm = Normalize(vmin=0, vmax=2*np.pi)
    fcolors = PARULA(norm(VV))

    fig, ax = _setup_3d_axes(None, None, elev=20, azim=-40, figsize=(7, 5.5))
    ax.plot_surface(XX, YY, ZZ, facecolors=fcolors, linewidth=0,
                    antialiased=True, alpha=0.7, shade=False)
    step = 4
    ax.quiver(XX[::step, ::step], YY[::step, ::step], ZZ[::step, ::step],
              Nx[::step, ::step], Ny[::step, ::step], Nz[::step, ::step],
              length=0.15, color='k', linewidth=0.8)
    ax.set_title('Torus with surface normals', fontsize=11, pad=0)
    save(fig, "Torus with normals")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 9")

# ---- Plot 10: 15.7 - Klein Bagel ----
try:
    n = 80
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, 2*np.pi, n)
    UU, VV = np.meshgrid(u, v)
    XX = (3 + np.cos(UU/2)*np.sin(VV) - np.sin(UU/2)*np.sin(2*VV)) * np.cos(UU)
    YY = (3 + np.cos(UU/2)*np.sin(VV) - np.sin(UU/2)*np.sin(2*VV)) * np.sin(UU)
    ZZ = np.sin(UU/2)*np.sin(VV) + np.cos(UU/2)*np.sin(2*VV)
    # Colour by u-parameter
    norm = Normalize(vmin=0, vmax=2*np.pi)
    fcolors = PARULA(norm(UU))

    fig, ax = _setup_3d_axes(None, None, elev=20, azim=-40, figsize=(7, 5.5))
    ax.plot_surface(XX, YY, ZZ, facecolors=fcolors, linewidth=0,
                    antialiased=True, alpha=0.8, shade=False)
    ax.axis('off')
    ax.set_title('Klein Bagel', fontsize=11, pad=0)
    save(fig, "Klein Bagel")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 10")

print(f"\nGuide 15: Generated {plot_num} plots.")
