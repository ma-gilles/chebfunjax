"""Generate plots for Guide Chapter 17: Spherefun."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chebfunjax.plotting import chebfun_style, plot_sphere
chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)

plot_num = 0

def save(fig, desc):
    global plot_num
    plot_num += 1
    fname = os.path.join(OUT, f'guide17_{plot_num:02d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}: {desc}")

import jax.numpy as jnp
from chebfunjax.spherefun import Spherefun

# ---- Plot 1: cos(theta) = z (spherical harmonic Y_1^0) ----
try:
    f = Spherefun.from_function(lambda lam, theta: jnp.cos(theta))
    fig, ax = plot_sphere(f, title=r"$\cos\theta$ (Y$_1^0$) on the sphere")
    save(fig, "cos(theta) on sphere")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 1")

# ---- Plot 2: x*y on the sphere ----
try:
    g = Spherefun.from_function(
        lambda lam, theta: jnp.cos(lam) * jnp.sin(theta) * jnp.sin(lam) * jnp.sin(theta))
    fig, ax = plot_sphere(g, title=r"$xy$ on the sphere")
    save(fig, "x*y on sphere")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 2")

# ---- Plot 3: Concentrated Gaussian on sphere ----
try:
    h = Spherefun.from_function(
        lambda lam, theta: jnp.exp(-10 * (jnp.cos(lam) * jnp.sin(theta))**2))
    fig, ax = plot_sphere(h, title=r"$e^{-10(\cos\lambda\sin\theta)^2}$")
    save(fig, "Gaussian on sphere")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 3")

# ---- Plot 4: Y_2^1 spherical harmonic ----
try:
    # sph_harm removed in scipy >= 1.14; use explicit formula for Y_2^1
    # Y_2^1(theta, phi) = -sqrt(15/(8*pi)) * sin(theta)*cos(theta) * exp(i*phi)
    # Real part: -sqrt(15/(8*pi)) * sin(theta)*cos(theta)*cos(phi)
    coeff = -np.sqrt(15.0 / (8.0 * np.pi))
    Y21 = Spherefun.from_function(
        lambda lam, theta: coeff * jnp.sin(theta) * jnp.cos(theta) * jnp.cos(lam))
    fig, ax = plot_sphere(Y21, title=r"Spherical harmonic $Y_2^1$")
    save(fig, "Y_2^1 spherical harmonic")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 4")

# ---- Plot 5: Flat projection (Mollweide) of cos(theta) ----
try:
    f_flat = Spherefun.from_function(lambda lam, theta: jnp.cos(theta))
    n_lam, n_theta = 200, 100
    lam = np.linspace(-np.pi, np.pi, n_lam)
    theta = np.linspace(0.0, np.pi, n_theta)
    LAM, THETA = np.meshgrid(lam, theta, indexing='ij')
    ZZ = np.array(f_flat(jnp.array(LAM.ravel()), jnp.array(THETA.ravel()))).reshape(n_lam, n_theta)

    fig, ax = plt.subplots(figsize=(8, 4))
    cs = ax.pcolormesh(LAM, THETA, ZZ, cmap='RdBu_r', shading='auto')
    plt.colorbar(cs, ax=ax, fraction=0.02, pad=0.04)
    ax.set_xlabel(r"$\lambda$ (longitude)")
    ax.set_ylabel(r"$\theta$ (colatitude)")
    ax.set_title(r"$\cos\theta$ in $(\lambda, \theta)$ coordinates")
    ax.invert_yaxis()
    fig.set_facecolor("white")
    fig.tight_layout()
    save(fig, "cos(theta) flat map")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 5")

print(f"\nGuide 17: Generated {plot_num} plots.")
