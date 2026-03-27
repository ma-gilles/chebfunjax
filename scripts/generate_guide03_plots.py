"""Generate all plots for Guide Chapter 3: Rootfinding and Minima and Maxima.

Faithfully translates every plot from Chebfun Guide Chapter 3
(https://www.chebfun.org/docs/guide/guide03.html) to chebfunjax/Python.
"""

import matplotlib
matplotlib.use('Agg')

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import scipy.special as sp
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
import time

chebfun_style()

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT_DIR, exist_ok=True)

plot_idx = 0

# --------------------------------------------------------------------------
# Plot 1: p = x^3 + x^2 - x with roots -- Section 3.1
# MATLAB: plot(p), grid on; hold on, plot(r,p(r),'.r')
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    x = cj.chebfun(lambda x: x)
    p = x**3 + x**2 - x
    r = p.roots()
    fig, ax = cj.plot(p)
    ax.plot(np.asarray(r), [float(p(ri)) for ri in r], '.r', markersize=12)
    ax.grid(True)
    fig.savefig(os.path.join(OUT_DIR, f'guide03_{plot_idx:02d}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide03_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide03_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 2: Airy functions Ai and Bi with roots -- Section 3.1
# MATLAB: plot(Ai,'r'); plot(Bi,'b'); plot roots
# --------------------------------------------------------------------------
try:
    plot_idx += 1

    def airy_ai(x):
        x_np = np.asarray(x)
        return jnp.array(sp.airy(x_np)[0])

    def airy_bi(x):
        x_np = np.asarray(x)
        return jnp.array(sp.airy(x_np)[2])

    Ai = cj.chebfun(airy_ai, domain=[-10, 3])
    Bi = cj.chebfun(airy_bi, domain=[-10, 3])

    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Plot Ai in red
    xs = np.linspace(-10, 3, 600)
    ys_ai = np.array([float(Ai(xi)) for xi in xs])
    ys_bi = np.array([float(Bi(xi)) for xi in xs])
    ax.plot(xs, ys_ai, 'r', linewidth=1.8)
    ax.plot(xs, ys_bi, 'b', linewidth=1.8)

    rA = Ai.roots()
    rB = Bi.roots()
    ax.plot(np.asarray(rA), [float(Ai(ri)) for ri in rA], '.r', markersize=10)
    ax.plot(np.asarray(rB), [float(Bi(ri)) for ri in rB], '.b', markersize=10)
    ax.set_xlim(-10, 3)
    ax.set_ylim(-0.6, 1.5)
    ax.grid(True)
    fig.savefig(os.path.join(OUT_DIR, f'guide03_{plot_idx:02d}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide03_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide03_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 3: Oscillatory function with many roots (replaces 'fishfillet')
# We use an oscillatory function with ~130 roots
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    f = cj.chebfun(lambda x: jnp.sin(66 * jnp.pi * x) * jnp.exp(jnp.sin(x)),
                    domain=[-1, 1])
    r = f.roots()
    fig, ax = cj.plot(f)
    ax.plot(np.asarray(r), [float(f(ri)) for ri in r], '.r', markersize=4)
    ax.set_title(f'{len(r)} roots')
    fig.savefig(os.path.join(OUT_DIR, f'guide03_{plot_idx:02d}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide03_{plot_idx:02d}.png saved ({len(r)} roots)")
except Exception as e:
    print(f"guide03_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 4: x and cos(x) intersection -- Section 3.1
# MATLAB: plot(x); hold on, plot(f,'k'); r = roots(f-x); plot(r,f(r),'or')
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    x = cj.chebfun(lambda x: x, domain=[-2, 2])
    f = cj.cos(x)
    r = (f - x).roots()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    xs = np.linspace(-2, 2, 600)
    ax.plot(xs, np.array([float(x(xi)) for xi in xs]), linewidth=1.8)
    ax.plot(xs, np.array([float(f(xi)) for xi in xs]), 'k', linewidth=1.8)
    ax.plot(np.asarray(r), [float(f(ri)) for ri in r], 'or', markersize=8)
    ax.grid(True)
    fig.savefig(os.path.join(OUT_DIR, f'guide03_{plot_idx:02d}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide03_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide03_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 5: x and abs(x) side by side -- Section 3.2
# MATLAB: subplot(1,2,1), plot(x); subplot(1,2,2), plot(absx)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    x = cj.chebfun(lambda x: x)
    absx = cj.abs(x)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    xs = np.linspace(-1, 1, 300)
    ax1.plot(xs, np.array([float(x(xi)) for xi in xs]), linewidth=1.8)
    ax1.set_title('x')
    ax2.plot(xs, np.array([float(absx(xi)) for xi in xs]), linewidth=1.8)
    ax2.set_title('abs(x)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'guide03_{plot_idx:02d}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide03_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide03_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 6: exp(airy(x)) with extrema marked -- Section 3.3
# MATLAB: f = chebfun('exp(real(airy(x)))',[-15,0]); plot(f); roots(diff(f))
# --------------------------------------------------------------------------
try:
    plot_idx += 1

    def exp_airy(x):
        x_np = np.asarray(x)
        return jnp.array(np.exp(np.real(sp.airy(x_np)[0])))

    f = cj.chebfun(exp_airy, domain=[-15, 0])
    fp = f.diff()
    r = fp.roots()

    fig, ax = cj.plot(f)
    ax.plot(np.asarray(r), [float(f(ri)) for ri in r], '.r', markersize=8)
    ax.grid(True)
    fig.savefig(os.path.join(OUT_DIR, f'guide03_{plot_idx:02d}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide03_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide03_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 7: sin(x)+sin(x^2) on [0,15] -- Section 3.4
# MATLAB: f = chebfun('sin(x)+sin(x^2)',[0,15]); hold off, plot(f,'k')
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    f = cj.chebfun(lambda x: jnp.sin(x) + jnp.sin(x**2), domain=[0, 15])
    fig, ax = cj.plot(f, color='k')
    fig.savefig(os.path.join(OUT_DIR, f'guide03_{plot_idx:02d}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide03_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide03_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 8: sin(x)+sin(x^2) with global min/max marked -- Section 3.4
# MATLAB: plot(minpos,minval,'.b'); plot(maxpos,maxval,'.r')
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    f = cj.chebfun(lambda x: jnp.sin(x) + jnp.sin(x**2), domain=[0, 15])
    minpos, minval = f.min()
    maxpos, maxval = f.max()
    fig, ax = cj.plot(f, color='k')
    ax.plot(minpos, minval, '.b', markersize=15)
    ax.plot(maxpos, maxval, '.r', markersize=15)
    fig.savefig(os.path.join(OUT_DIR, f'guide03_{plot_idx:02d}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"guide03_{plot_idx:02d}.png saved")
except Exception as e:
    print(f"guide03_{plot_idx:02d}.png FAILED: {e}")

print(f"\nGuide 03 plot generation complete. {plot_idx} plots attempted.")
