"""Generate all plots for Guide Chapter 9: Infinite Intervals, Infinite Function Values, and Singularities."""

import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, CHEBFUN_BLUE, CHEBFUN_RED
from chebfunjax.fun.unbndfun import Unbndfun
from chebfunjax.fun.singfun import Singfun
from chebfunjax.domain import Domain
from scipy.special import gamma as scipy_gamma

chebfun_style()

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT_DIR, exist_ok=True)

plot_idx = 0


def save(fig, hint=""):
    global plot_idx
    plot_idx += 1
    path = os.path.join(OUT_DIR, f'guide09_{plot_idx:02d}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  guide09_{plot_idx:02d}.png saved  ({hint})")


# --------------------------------------------------------------------------
# Plot 1: f = 0.75 + sin(10x)/exp(x) on [0, inf)   (Section 9.1)
# --------------------------------------------------------------------------
try:
    f = Unbndfun.from_function(
        lambda x: 0.75 + jnp.sin(10 * x) * jnp.exp(-x),
        Domain((0.0, float('inf'))),
    )
    xs = np.linspace(0, 10, 600)
    ys = np.array([float(f(jnp.float64(xi))) for xi in xs])
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "0.75+sin(10x)/exp(x) on [0,inf)")
except Exception as e:
    plot_idx += 1
    print(f"  guide09_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 2: g = 1/gamma(x+1) on [0, inf)   (Section 9.1)
# --------------------------------------------------------------------------
try:
    g_fun = Unbndfun.from_function(
        lambda x: 1.0 / jnp.array(scipy_gamma(np.array(x + 1.0, dtype=np.float64))),
        Domain((0.0, float('inf'))),
    )
    xs = np.linspace(0, 10, 600)
    ys = np.array([float(g_fun(jnp.float64(xi))) for xi in xs])
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "1/gamma(x+1) on [0,inf)")
except Exception as e:
    plot_idx += 1
    print(f"  guide09_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 3: f and g together with intersection points   (Section 9.1)
# --------------------------------------------------------------------------
try:
    xs = np.linspace(0.01, 5.0, 800)
    ys_f = np.array([float(f(jnp.float64(xi))) for xi in xs])
    ys_g = np.array([float(g_fun(jnp.float64(xi))) for xi in xs])
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(xs, ys_f, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.plot(xs, ys_g, color=CHEBFUN_RED, linewidth=1.8)
    # Mark approximate intersection points
    diff_vals = ys_f - ys_g
    for i in range(len(diff_vals) - 1):
        if diff_vals[i] * diff_vals[i + 1] < 0:
            t = diff_vals[i] / (diff_vals[i] - diff_vals[i + 1])
            xr = xs[i] + t * (xs[i + 1] - xs[i])
            yr = ys_f[i] + t * (ys_f[i + 1] - ys_f[i])
            ax.plot(xr, yr, '.k', markersize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "f and g with intersections")
except Exception as e:
    plot_idx += 1
    print(f"  guide09_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 4: |tanh(x-1) - 1/3| on (-inf, inf)   (Section 9.1)
# --------------------------------------------------------------------------
try:
    g2 = Unbndfun.from_function(
        lambda x: jnp.abs(jnp.tanh(x - 1.0) - 1.0 / 3.0),
        Domain((float('-inf'), float('inf'))),
    )
    xs = np.linspace(-10, 10, 600)
    ys = np.array([float(g2(jnp.float64(xi))) for xi in xs])
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "|tanh(x-1)-1/3| on (-inf,inf)")
except Exception as e:
    plot_idx += 1
    print(f"  guide09_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 5: cos(x)/(1e5+(x-30)^6) on [0,100]   (Section 9.1)
# --------------------------------------------------------------------------
try:
    h_fun = Unbndfun.from_function(
        lambda x: jnp.cos(x) / (1e5 + (x - 30.0)**6),
        Domain((0.0, float('inf'))),
    )
    xs = np.linspace(0, 100, 600)
    ys = np.array([float(h_fun(jnp.float64(xi))) for xi in xs])
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "cos(x)/(1e5+(x-30)^6) on [0,100]")
except Exception as e:
    plot_idx += 1
    print(f"  guide09_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 6: sinc = sin(pi*x)/(pi*x) on [-10,10]   (Section 9.1)
# --------------------------------------------------------------------------
try:
    xs = np.linspace(-10, 10, 800)
    ys = np.sinc(xs)  # numpy sinc(x) = sin(pi*x)/(pi*x)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(xs, ys, color='m', linewidth=1.8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "sinc on [-10,10]")
except Exception as e:
    plot_idx += 1
    print(f"  guide09_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 7: sin(50x)+1/x on [0,4] with pole   (Section 9.2)
# --------------------------------------------------------------------------
try:
    xs = np.linspace(0.005, 4.0, 1000)
    ys = np.sin(50 * xs) + 1.0 / xs
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.set_ylim(-5, 30)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "sin(50x)+1/x on [0,4]")
except Exception as e:
    plot_idx += 1
    print(f"  guide09_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 8: sin(50x)+1/x on [-2,4] with pole at 0   (Section 9.2)
# --------------------------------------------------------------------------
try:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    xs_left = np.linspace(-2.0, -0.005, 400)
    xs_right = np.linspace(0.005, 4.0, 600)
    ax.plot(xs_left, np.sin(50 * xs_left) + 1.0 / xs_left, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.plot(xs_right, np.sin(50 * xs_right) + 1.0 / xs_right, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.set_ylim(-30, 30)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "sin(50x)+1/x on [-2,4]")
except Exception as e:
    plot_idx += 1
    print(f"  guide09_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 9: tan(x) with multiple poles   (Section 9.2)
# --------------------------------------------------------------------------
try:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    # Plot tan(x) between each pair of consecutive poles
    for k in range(-3, 3):
        lo = k * np.pi - np.pi / 2 + 0.02
        hi = k * np.pi + np.pi / 2 - 0.02
        xs = np.linspace(lo, hi, 200)
        ax.plot(xs, np.tan(xs), color=CHEBFUN_BLUE, linewidth=1.8)
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "tan(x) with poles")
except Exception as e:
    plot_idx += 1
    print(f"  guide09_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 10: tan(x) intersecting x/2, roots marked   (Section 9.2)
# --------------------------------------------------------------------------
try:
    from scipy.optimize import brentq
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for k in range(-3, 3):
        lo = k * np.pi - np.pi / 2 + 0.02
        hi = k * np.pi + np.pi / 2 - 0.02
        xs = np.linspace(lo, hi, 200)
        ax.plot(xs, np.tan(xs), color=CHEBFUN_BLUE, linewidth=1.8)
    # x/2 line
    xline = np.linspace(-5 * np.pi / 2, 5 * np.pi / 2, 200)
    ax.plot(xline, xline / 2.0, 'k', linewidth=1.2)
    # Roots of tan(x) - x/2
    for k in range(-3, 3):
        lo = k * np.pi - np.pi / 2 + 0.05
        hi = k * np.pi + np.pi / 2 - 0.05
        try:
            r = brentq(lambda x: np.tan(x) - x / 2.0, lo, hi)
            ax.plot(r, r / 2.0, 'or', markersize=8)
        except Exception:
            pass
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "tan(x) vs x/2")
except Exception as e:
    plot_idx += 1
    print(f"  guide09_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 11: g = sin(2*x2)+min(|f+2|,6)   (Section 9.2)
# --------------------------------------------------------------------------
try:
    # Compute g = sin(2*x/2) + min(|tan(x)+2|, 6) on several branches
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for k in range(-3, 3):
        lo = k * np.pi - np.pi / 2 + 0.02
        hi = k * np.pi + np.pi / 2 - 0.02
        xs = np.linspace(lo, hi, 200)
        gvals = np.sin(2 * xs / 2.0) + np.minimum(np.abs(np.tan(xs) + 2), 6.0)
        ax.plot(xs, gvals, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "sin(2*x2)+min(|f+2|,6)")
except Exception as e:
    plot_idx += 1
    print(f"  guide09_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 12: gamma function on [-4,4]   (Section 9.2)
# --------------------------------------------------------------------------
try:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    pole_locs = [-4, -3, -2, -1, 0, 4]
    for i in range(len(pole_locs) - 1):
        xs = np.linspace(pole_locs[i] + 0.01, pole_locs[i + 1] - 0.01, 400)
        ax.plot(xs, scipy_gamma(xs), color=CHEBFUN_BLUE, linewidth=1.8)
    ax.set_ylim(-10, 10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "gamma function on [-4,4]")
except Exception as e:
    plot_idx += 1
    print(f"  guide09_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 13: cos(100x)+sin(x)^(-2+sign(x)) asymmetric poles  (Section 9.2)
# --------------------------------------------------------------------------
try:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    xs_l = np.linspace(-1.0, -0.005, 500)
    xs_r = np.linspace(0.005, 1.0, 500)
    ys_l = np.cos(100 * xs_l) + np.abs(np.sin(xs_l))**(-2 + np.sign(xs_l))
    ys_r = np.cos(100 * xs_r) + np.abs(np.sin(xs_r))**(-2 + np.sign(xs_r))
    ax.plot(xs_l, ys_l, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.plot(xs_r, ys_r, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.set_ylim(-30, 30)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "asymmetric poles")
except Exception as e:
    plot_idx += 1
    print(f"  guide09_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 14: (2/pi)/sqrt(1-x^2) weight function   (Section 9.3)
# --------------------------------------------------------------------------
try:
    w = Singfun.from_function(
        lambda x: (2.0 / jnp.pi) / jnp.sqrt(1.0 - x**2),
        exponents=(-0.5, -0.5),
    )
    xs = np.linspace(-0.999, 0.999, 800)
    ys = np.array([float(w(jnp.float64(xi))) for xi in xs])
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(xs, ys, color='m', linewidth=1.8)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "(2/pi)/sqrt(1-x^2)")
except Exception as e:
    plot_idx += 1
    print(f"  guide09_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 15: sqrt(x*exp(x)) on [0,2]   (Section 9.3)
# --------------------------------------------------------------------------
try:
    xs = np.linspace(0, 2, 800)
    ys = np.sqrt(xs * np.exp(xs))
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "sqrt(x*exp(x))")
except Exception as e:
    plot_idx += 1
    print(f"  guide09_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 16: sqrt(|sin(theta)|) on [0, 4*pi]   (Section 9.3)
# --------------------------------------------------------------------------
try:
    xs = np.linspace(0, 4 * np.pi, 1000)
    ys = np.sqrt(np.abs(np.sin(xs)))
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "sqrt(|sin(theta)|)")
except Exception as e:
    plot_idx += 1
    print(f"  guide09_{plot_idx:02d}.png FAILED: {e}")

print(f"\nGuide 09: generated {plot_idx} plots.")
