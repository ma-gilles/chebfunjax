"""Generate plots for Guide Chapter 20: Ballfun."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chebfunjax.plotting import chebfun_style
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

import jax.numpy as jnp
from chebfunjax.ballfun.ballfun import Ballfun

# ---- Plot 1: r^2 = x^2+y^2+z^2 on the ball (equatorial slice) ----
try:
    f = Ballfun.from_function(lambda x, y, z: x**2 + y**2 + z**2)

    # Equatorial (z=0) slice: show function in x-y plane
    n = 100
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    XX, YY = np.meshgrid(xs, ys)
    R2 = XX**2 + YY**2
    mask = R2 <= 1.0

    ZZ = np.zeros_like(XX)
    # Evaluate at points inside the disk
    inside_x = XX[mask]
    inside_y = YY[mask]
    inside_z = np.zeros_like(inside_x)
    vals = np.array(f(jnp.array(inside_x), jnp.array(inside_y), jnp.array(inside_z)))
    ZZ[mask] = vals
    ZZ[~mask] = np.nan

    fig, ax = plt.subplots(figsize=(5, 5))
    cs = ax.pcolormesh(XX, YY, ZZ, cmap='RdBu_r', shading='auto')
    plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.8)
    ax.set_aspect('equal')
    ax.set_title(r"$x^2+y^2+z^2$ equatorial slice ($z=0$)", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.set_facecolor("white")
    fig.tight_layout()
    save(fig, "r^2 equatorial slice")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 1")

# ---- Plot 2: exp(-(x^2+y^2+z^2)) on the ball, three slices ----
try:
    f2 = Ballfun.from_function(lambda x, y, z: jnp.exp(-(x**2 + y**2 + z**2)))

    n = 80
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    XX, YY = np.meshgrid(xs, ys)
    R2 = XX**2 + YY**2
    mask = R2 <= 1.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    slice_labels = [("z=0", 0), ("y=0", 1), ("x=0", 2)]

    for i, (label, dim) in enumerate(slice_labels):
        ZZ = np.full_like(XX, np.nan)
        inside_x = XX[mask]
        inside_y = YY[mask]
        inside_z = np.zeros_like(inside_x)

        if dim == 0:  # z=0 slice: x-y plane
            vals = np.array(f2(jnp.array(inside_x), jnp.array(inside_y), jnp.array(inside_z)))
            xlabel, ylabel = "x", "y"
        elif dim == 1:  # y=0 slice: x-z plane
            vals = np.array(f2(jnp.array(inside_x), jnp.array(inside_z), jnp.array(inside_y)))
            xlabel, ylabel = "x", "z"
        else:  # x=0 slice: y-z plane
            vals = np.array(f2(jnp.array(inside_z), jnp.array(inside_x), jnp.array(inside_y)))
            xlabel, ylabel = "y", "z"

        ZZ[mask] = vals
        cs = axes[i].pcolormesh(XX, YY, ZZ, cmap='RdBu_r', shading='auto')
        plt.colorbar(cs, ax=axes[i], fraction=0.046, pad=0.04)
        theta = np.linspace(0, 2*np.pi, 300)
        axes[i].plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.8)
        axes[i].set_aspect('equal')
        axes[i].set_title(f"{label} slice", fontsize=10)
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(ylabel)

    fig.suptitle(r"$e^{-(x^2+y^2+z^2)}$ on the unit ball", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    save(fig, "Gaussian three slices")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 2")

# ---- Plot 3: Radial profile plot ----
try:
    f3 = Ballfun.from_function(lambda x, y, z: x**2 + y**2 + z**2)
    # Evaluate along a radial line (x-axis)
    rs = np.linspace(0, 1, 100)
    vals = np.array(f3(jnp.array(rs), jnp.zeros_like(jnp.array(rs)), jnp.zeros_like(jnp.array(rs))))

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(rs, vals, color='#4169E1', linewidth=1.8)
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

# ---- Plot 4: Solid harmonic (if scipy available) ----
try:
    # Y_2^0(theta) = (1/4)*sqrt(5/pi) * (3*cos^2(theta) - 1)
    coeff_y20 = 0.25 * np.sqrt(5.0 / np.pi)
    f4 = Ballfun.from_function(
        lambda r, lam, th: r**2 * coeff_y20 * (3 * jnp.cos(th)**2 - 1),
        spherical=True,
    )

    # Equatorial slice
    n = 80
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    XX, YY = np.meshgrid(xs, ys)
    R2 = XX**2 + YY**2
    mask = R2 <= 1.0

    ZZ = np.full_like(XX, np.nan)
    inside_x = XX[mask]
    inside_y = YY[mask]
    inside_z = np.zeros_like(inside_x)
    vals = np.array(f4(jnp.array(inside_x), jnp.array(inside_y), jnp.array(inside_z)))
    ZZ[mask] = vals

    fig, ax = plt.subplots(figsize=(5, 5))
    cs = ax.pcolormesh(XX, YY, ZZ, cmap='RdBu_r', shading='auto')
    plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    theta_bdy = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta_bdy), np.sin(theta_bdy), 'k-', linewidth=0.8)
    ax.set_aspect('equal')
    ax.set_title(r"Solid harmonic $r^2 Y_2^0$ (equatorial)", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.set_facecolor("white")
    fig.tight_layout()
    save(fig, "Solid harmonic equatorial")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 4")

# ---- Plot 5: x^2 and y^2 sum (arithmetic) equatorial ----
try:
    fx2 = Ballfun.from_function(lambda x, y, z: x**2)
    fy2 = Ballfun.from_function(lambda x, y, z: y**2)
    h = fx2 + fy2  # x^2 + y^2

    n = 80
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    XX, YY = np.meshgrid(xs, ys)
    R2 = XX**2 + YY**2
    mask = R2 <= 1.0

    ZZ = np.full_like(XX, np.nan)
    inside_x = XX[mask]
    inside_y = YY[mask]
    inside_z = np.zeros_like(inside_x)
    vals = np.array(h(jnp.array(inside_x), jnp.array(inside_y), jnp.array(inside_z)))
    ZZ[mask] = vals

    fig, ax = plt.subplots(figsize=(5, 5))
    cs = ax.pcolormesh(XX, YY, ZZ, cmap='viridis', shading='auto')
    plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    theta_bdy = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta_bdy), np.sin(theta_bdy), 'k-', linewidth=0.8)
    ax.set_aspect('equal')
    ax.set_title(r"$x^2 + y^2$ equatorial slice ($z=0$)", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.set_facecolor("white")
    fig.tight_layout()
    save(fig, "Arithmetic result equatorial")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 5")

print(f"\nGuide 20: Generated {plot_num} plots.")
