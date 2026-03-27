"""Generate plots for Guide Chapter 18: Chebfun3."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chebfunjax.plotting import chebfun_style, plot_slices
chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)

plot_num = 0

def save(fig, desc):
    global plot_num
    plot_num += 1
    fname = os.path.join(OUT, f'guide18_{plot_num:02d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}: {desc}")

import jax.numpy as jnp
from chebfunjax.chebfun3d import chebfun3

# ---- Plot 1: 3D Runge function slices ----
try:
    f = chebfun3(lambda x, y, z: 1.0 / (1.0 + x**2 + y**2 + z**2))
    fig, ax = plot_slices(f, title=r"$\frac{1}{1+x^2+y^2+z^2}$ (mid-plane slices)")
    save(fig, "3D Runge function slices")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 1")

# ---- Plot 2: cos(x+y+z) slices ----
try:
    f2 = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
    fig, ax = plot_slices(f2, title=r"$\cos(x+y+z)$ (mid-plane slices)")
    save(fig, "cos(x+y+z) slices")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 2")

# ---- Plot 3: exp(-(x^2+y^2+z^2)) on [0,2]^3 slices ----
try:
    g = chebfun3(lambda x, y, z: jnp.exp(-(x**2 + y**2 + z**2)),
                 domain=(0.0, 2.0, 0.0, 2.0, 0.0, 2.0))
    fig, ax = plot_slices(g, title=r"$e^{-(x^2+y^2+z^2)}$ on $[0,2]^3$")
    save(fig, "Gaussian in 3D")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 3")

# ---- Plot 4: exp(xyz) slices ----
try:
    f4 = chebfun3(lambda x, y, z: jnp.exp(x * y * z))
    fig, ax = plot_slices(f4, title=r"$e^{xyz}$ (mid-plane slices)")
    save(fig, "exp(xyz) slices")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 4")

# ---- Plot 5: Isosurface-like visualization of 1/(1+x^2+y^2+z^2) ----
try:
    # 2D slice gallery: z = -0.5, 0, 0.5
    f5 = chebfun3(lambda x, y, z: 1.0 / (1.0 + x**2 + y**2 + z**2))
    n = 80
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    XX, YY = np.meshgrid(xs, ys)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    z_vals = [-0.5, 0.0, 0.5]
    for i, zval in enumerate(z_vals):
        ZM = np.full_like(XX, zval)
        FF = np.array(f5(jnp.array(XX.ravel()), jnp.array(YY.ravel()),
                         jnp.array(ZM.ravel()))).reshape(n, n)
        cs = axes[i].contourf(XX, YY, FF, levels=15, cmap='RdBu_r')
        plt.colorbar(cs, ax=axes[i], fraction=0.046, pad=0.04)
        axes[i].set_title(f"z = {zval}", fontsize=10)
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        axes[i].set_aspect('equal')

    fig.suptitle(r"$\frac{1}{1+x^2+y^2+z^2}$ slices", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    save(fig, "3D Runge function z-slices")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 5")

print(f"\nGuide 18: Generated {plot_num} plots.")
