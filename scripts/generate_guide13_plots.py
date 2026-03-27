"""Generate plots for Guide Chapter 13: Chebfun2: Integration and Differentiation.

Faithful translation of all figures from the original MATLAB Chebfun Guide
Chapter 13 (https://www.chebfun.org/docs/guide/guide13.html).
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
from chebfunjax.plotting import chebfun_style, surf, contour
from chebfunjax.chebfun2d import chebfun2

chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)

plot_num = 0

def save(fig, desc):
    global plot_num
    plot_num += 1
    fname = os.path.join(OUT, f'guide13_{plot_num:02d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}: {desc}")


# ---- Plot 1: 13.1 - sum(f,2) = integral over x, a function of y ----
try:
    f = chebfun2(lambda x, y: jnp.sin(10*x*y), domain=(0.0, float(jnp.pi/4), 0.0, 3.0))
    # Integrate over x (dim=2), leaving a function of y
    h = f.sum(dim=2)
    # Plot h as a 1D function of y
    ys = np.linspace(0, 3, 200)
    vals = np.array([float(h(0.0, yi)) for yi in ys])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(ys, vals, 'b-', linewidth=1.8)
    ax.set_xlabel('y')
    ax.set_title(r'$\int_0^{\pi/4} \sin(10xy)\,dx$ as a function of $y$')
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    save(fig, "sum(f,2) plot")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 1")

# ---- Plot 2: 13.2 - 2D Runge function surface ----
try:
    runge = chebfun2(lambda x, y: 1.0 / (0.01 + x**2 + y**2))
    fig, ax = surf(runge, title=r'$1/(0.01 + x^2 + y^2)$')
    save(fig, "2D Runge function")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 2")

# ---- Plot 3: 13.2 - Mean of Runge wrt y ----
try:
    runge = chebfun2(lambda x, y: 1.0 / (0.01 + x**2 + y**2))
    # mean wrt y = (1/2) * integral over y from -1 to 1
    h = runge.sum(dim=1)  # integral over y, function of x
    xs = np.linspace(-1, 1, 200)
    # The sum gives integral; mean = integral / domain_length = integral / 2
    vals = np.array([float(h(xi, 0.0)) / 2.0 for xi in xs])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(xs, vals, 'b-', linewidth=1.8)
    ax.set_xlabel('x')
    ax.set_title('Mean value of 2D Runge function wrt y')
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    save(fig, "Mean of Runge wrt y")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 3")

# ---- Plot 4: 13.3 - Contours of cumsum2(f) ----
try:
    f = chebfun2(lambda x, y: jnp.sin(3*((x+1)**2 + (y+1)**2)))
    # cumsum2 not directly available in chebfunjax; approximate numerically
    n = 150
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    XX, YY = np.meshgrid(xs, ys)
    xf = jnp.array(XX.ravel())
    yf = jnp.array(YY.ravel())
    ZZ = np.array(f(xf, yf)).reshape(n, n)
    # Double cumulative sum (trapezoid rule)
    cumsum_y = np.cumsum(ZZ, axis=0) * dy  # cumsum along y (axis 0)
    cumsum_xy = np.cumsum(cumsum_y, axis=1) * dx  # then along x (axis 1)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    cs = ax.contourf(XX, YY, cumsum_xy, levels=15, cmap='RdBu_r')
    ax.contour(XX, YY, cumsum_xy, levels=15, colors='k', linewidths=0.5)
    plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title('Contours of cumsum2(f)')
    fig.tight_layout()
    save(fig, "cumsum2 contour")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 4")

# ---- Plot 5: 13.4 - Circle: two real vs one complex ----
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    t = np.linspace(0, 2*np.pi, 200)
    # Two real-valued functions
    ax1.plot(np.cos(t), np.sin(t), 'b-', linewidth=1.8)
    ax1.set_aspect('equal')
    ax1.set_title('Two real-valued functions')
    ax1.grid(True, alpha=0.3, linestyle='--')
    # One complex-valued function
    ax2.plot(np.cos(t), np.sin(t), 'b-', linewidth=1.8)
    ax2.set_aspect('equal')
    ax2.set_title('One complex-valued function')
    ax2.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    save(fig, "Circle: real vs complex encoding")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 5")

# ---- Plot 6: 13.5 - Spiral curve C ----
try:
    t = np.linspace(0, 1, 500)
    C_real = t * np.cos(10*t)
    C_imag = t * np.sin(10*t)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(C_real, C_imag, 'k-', linewidth=1.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title(r'Curve $C = t\, e^{10it}$, $t \in [0,1]$')
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    save(fig, "Spiral curve C")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 6")

# ---- Plot 7: 13.5 - f(x,y) = cos(10*x*y^2) + exp(-x^2) with curve ----
try:
    f = chebfun2(lambda x, y: jnp.cos(10*x*y**2) + jnp.exp(-x**2))
    fig, ax = surf(f, title=r'$\cos(10xy^2) + e^{-x^2}$')
    # Overlay the curve on the surface
    t = np.linspace(0, 1, 500)
    cx = t * np.cos(10*t)
    cy = t * np.sin(10*t)
    cz = np.array([float(f(jnp.float64(xi), jnp.float64(yi))) for xi, yi in zip(cx, cy)])
    ax.plot3D(cx, cy, cz, 'k-', linewidth=2)
    save(fig, "f(x,y) with curve on surface")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 7")

print(f"\nGuide 13: Generated {plot_num} plots.")
