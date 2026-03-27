"""Generate plots for Guide Chapter 19: SPIN, SPIN2, SPIN3 and SPINSPHERE."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chebfunjax.plotting import chebfun_style, CHEBFUN_BLUE
chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)

plot_num = 0

def save(fig, desc):
    global plot_num
    plot_num += 1
    fname = os.path.join(OUT, f'guide19_{plot_num:02d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}: {desc}")

# ---- Plot 1: KdV equation solution ----
try:
    from chebfunjax.spin import spin
    # Use built-in defaults (N=512, dt=3e-6, tspan 0..0.03015)
    x, t, u = spin('KdV')
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, np.real(u), color=CHEBFUN_BLUE, linewidth=1.8)
    ax.set_title(f"KdV equation at $t = {t:.4g}$", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    fig.set_facecolor("white")
    fig.tight_layout()
    save(fig, "KdV solution")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 1")

# ---- Plot 2: Allen-Cahn equation solution ----
try:
    from chebfunjax.spin import spin
    x, t, u = spin('AC')
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, np.real(u), color=CHEBFUN_BLUE, linewidth=1.8)
    ax.set_title(f"Allen-Cahn at $t = {t:.4g}$", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    fig.set_facecolor("white")
    fig.tight_layout()
    save(fig, "Allen-Cahn solution")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 2")

# ---- Plot 3: Kuramoto-Sivashinsky equation solution ----
try:
    from chebfunjax.spin import spin
    x, t, u = spin('KS')
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, np.real(u), color=CHEBFUN_BLUE, linewidth=1.0)
    ax.set_title(f"Kuramoto-Sivashinsky at $t = {t:.4g}$", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    fig.set_facecolor("white")
    fig.tight_layout()
    save(fig, "KS solution")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 3")

# ---- Plot 4: NLS equation solution (|u|^2) ----
try:
    from chebfunjax.spin import spin
    x, t, u = spin('NLS')
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, np.abs(u)**2, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.set_title(f"NLS $|u|^2$ at $t = {t:.4g}$", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|u|^2$")
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    fig.set_facecolor("white")
    fig.tight_layout()
    save(fig, "NLS |u|^2")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 4")

# ---- Plot 5: 2D Ginzburg-Landau ----
try:
    from chebfunjax.spin import spin2
    xx, yy, t, u = spin2('GL')
    u_plot = np.abs(np.asarray(u)) if not isinstance(u, list) else np.abs(np.asarray(u[0]))

    fig, ax = plt.subplots(figsize=(5, 5))
    cs = ax.pcolormesh(xx, yy, u_plot, cmap='inferno', shading='auto')
    plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"2D Ginzburg-Landau $|u|$ at $t={t:.1f}$", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    fig.set_facecolor("white")
    fig.tight_layout()
    save(fig, "2D Ginzburg-Landau")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 5")

# ---- Plot 6: 2D Swift-Hohenberg ----
try:
    from chebfunjax.spin import spin2
    xx, yy, t, u = spin2('SH')
    u_plot = np.real(np.asarray(u)) if not isinstance(u, list) else np.real(np.asarray(u[0]))

    fig, ax = plt.subplots(figsize=(5, 5))
    cs = ax.pcolormesh(xx, yy, u_plot, cmap='RdBu_r', shading='auto')
    plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"2D Swift-Hohenberg at $t={t:.1f}$", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    fig.set_facecolor("white")
    fig.tight_layout()
    save(fig, "2D Swift-Hohenberg")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 6")

# ---- Plot 7: 2D Allen-Cahn ----
try:
    from chebfunjax.spin import spin2
    xx, yy, t, u = spin2('AC2')
    u_plot = np.real(np.asarray(u)) if not isinstance(u, list) else np.real(np.asarray(u[0]))

    fig, ax = plt.subplots(figsize=(5, 5))
    cs = ax.pcolormesh(xx, yy, u_plot, cmap='RdBu_r', shading='auto')
    plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"2D Allen-Cahn at $t={t:.1f}$", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    fig.set_facecolor("white")
    fig.tight_layout()
    save(fig, "2D Allen-Cahn")
except Exception:
    traceback.print_exc()
    print("  SKIP plot 7")

print(f"\nGuide 19: Generated {plot_num} plots.")
