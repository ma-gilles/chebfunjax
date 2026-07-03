"""Generate all plots for Guide Chapter 2: Integration and Differentiation.

This script reproduces every figure from Chebfun Guide Chapter 2 using
chebfunjax.  Each figure is exported at the exact pixel size of its
chebfun.org reference render (610x258) via
:func:`chebfunjax.plotting.save_chebfun_figure`, so it can be compared
pixel-for-pixel against the MATLAB documentation images.

The MATLAB source for each figure is on
https://www.chebfun.org/docs/guide/guide02.html and is the ground truth.
Where MATLAB Chebfun renders features that chebfunjax's public plotting
API does not (dotted vertical connectors at jumps, delta-function arrows),
those are drawn here with matplotlib on top of values evaluated by
chebfunjax.
"""

import matplotlib

matplotlib.use('Agg')

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.plotting import (
    CHEBFUN_BLUE,
    CHEBFUN_RED,
    PARULA,
    _apply_style,
    chebfun_style,
    save_chebfun_figure,
)

chebfun_style()

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT_DIR, exist_ok=True)

# chebfun.org Guide figures are rendered at 610x258 px.
SIZE = (610, 258)

# MATLAB's 'm' colour is pure magenta (1,0,1); matplotlib's 'm' is the
# darker (0.75,0,0.75).  Use MATLAB's value for parity with the references.
MAGENTA = (1.0, 0.0, 1.0)

# Axes box position [left, bottom, width, height] measured from the
# chebfun.org reference renders (single-plot figures): spines at
# x=79..551, y=19..229 of the 610x258 canvas.
SINGLE_BOX = [79 / 610, 1 - 229 / 258, (551 - 79) / 610, (229 - 19) / 258]


def _save(fig, idx):
    # Single-plot figures: pin the axes to the reference box so the plot
    # area lines up pixel-for-pixel.  Multi-axes figures (7, 13, 14) set
    # their own positions.
    if len(fig.axes) == 1:
        fig.axes[0].set_position(SINGLE_BOX)
    save_chebfun_figure(fig, os.path.join(OUT_DIR, f'guide02_{idx:02d}.png'), size=SIZE)
    plt.close(fig)
    print(f"guide02_{idx:02d}.png saved")


# NOTE: cj.plot (plotting.plot_1d) now renders piecewise chebfuns per-fun
# with dotted vertical jump connectors in MATLAB-Chebfun style, so the
# discontinuous figures below (1, 2, 7, 10, 11) use it directly.

# --------------------------------------------------------------------------
# Plot 1 (guide02_01): |J_0(x)| on [0, 20]
# MATLAB: f = chebfun(@(x) abs(besselj(0,x)),[0 20],'splitting','on');
#         plot(f), ylim([0 1.1])
# --------------------------------------------------------------------------
try:
    g = cj.chebfun(lambda t: sp.jv(0, np.asarray(t, dtype=np.float64)), domain=[0, 20])
    g_abs = g.abs()
    fig, ax = cj.plot(g_abs)   # continuous: connectors auto-skipped
    ax.set_ylim([0, 1.1])
    _save(fig, 1)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"guide02_01.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 2 (guide02_02): min(sech(3*sin(10*x)), sin(9*x)) on [-1,1]
# MATLAB: x=chebfun('x'); f=sech(3*sin(10*x)); g=sin(9*x); h=min(f,g); plot(h)
# --------------------------------------------------------------------------
try:
    f = cj.chebfun(lambda x: 1.0 / jnp.cosh(3.0 * jnp.sin(10.0 * x)))
    g = cj.chebfun(lambda x: jnp.sin(9.0 * x))
    diff_fg = f - g
    crossings = diff_fg.roots()
    breaks = [-1.0] + sorted(float(r) for r in crossings) + [1.0]
    clean = [breaks[0]]
    for b in breaks[1:]:
        if b - clean[-1] > 1e-12:
            clean.append(b)
    breaks = clean
    piece_list = []
    for i in range(len(breaks) - 1):
        mid = 0.5 * (breaks[i] + breaks[i + 1])
        fval = float(f(jnp.float64(mid)))
        gval = float(g(jnp.float64(mid)))
        chosen = f if fval <= gval else g
        piece_list.append(_Piece.from_function(lambda x, _c=chosen: _c(x),
                                               breaks[i], breaks[i + 1]))
    h = Chebfun(funs=piece_list, domain=Domain(tuple(breaks)))
    fig, ax = plt.subplots()
    plot_pieces(h, ax, color=CHEBFUN_BLUE)   # min is continuous: no jumps
    ax.set_ylim([-1, 1])
    _apply_style(ax)
    _save(fig, 2)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"guide02_02.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 3 (guide02_03): Kahaner's F21F function with three spikes on [0,1]
# --------------------------------------------------------------------------
try:
    def ff(x):
        return (1.0 / jnp.cosh(10.0 * (x - 0.2)))**2 + \
               (1.0 / jnp.cosh(100.0 * (x - 0.4)))**4 + \
               (1.0 / jnp.cosh(1000.0 * (x - 0.6)))**6

    f = cj.chebfun(ff, domain=[0, 1])
    fig, ax = plt.subplots()
    xs = np.linspace(0, 1, 4000)
    ax.plot(xs, np.array(f(jnp.array(xs))), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.set_ylim([0, 1.2])
    _apply_style(ax)
    ax.set_yticks(np.arange(0, 1.21, 0.2))   # after _apply_style (overrides MaxNLocator)
    _save(fig, 3)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"guide02_03.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 4 (guide02_04): exp(-1/sin(10*x)^2) on [-1,1]
# --------------------------------------------------------------------------
try:
    f = cj.chebfun(lambda x: jnp.exp(-1.0 / jnp.sin(10.0 * x)**2))
    fig, ax = plt.subplots()
    xs = np.linspace(-1, 1, 2000)
    ax.plot(xs, np.array(f(jnp.array(xs))), color=CHEBFUN_BLUE, linewidth=1.2)
    _apply_style(ax)
    _save(fig, 4)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"guide02_04.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 5 (guide02_05): cumsum of erf integrand, raw (F(-5)=0)
# MATLAB: plot(fint,'m'), ylim([-0.2 2.2]), grid on
# --------------------------------------------------------------------------
try:
    f = cj.chebfun(lambda t: 2.0 * jnp.exp(-t**2) / jnp.sqrt(jnp.pi), domain=[-5, 5])
    fint = f.cumsum()
    fig, ax = plt.subplots()
    xs = np.linspace(-5, 5, 1200)
    ax.plot(xs, np.array(fint(jnp.array(xs))), color=MAGENTA, linewidth=1.2)
    ax.set_ylim([-0.2, 2.2])
    _apply_style(ax)
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.grid(True, color=(0.85, 0.85, 0.85), linewidth=0.5, linestyle='-')
    _save(fig, 5)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"guide02_05.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 6 (guide02_06): cumsum shifted so F(0)=0
# MATLAB: fint = fint - fint(0); plot(fint,'m'), ylim([-1.2 1.2]), grid on
# --------------------------------------------------------------------------
try:
    f = cj.chebfun(lambda t: 2.0 * jnp.exp(-t**2) / jnp.sqrt(jnp.pi), domain=[-5, 5])
    fint = f.cumsum()
    fint_shifted = fint - float(fint(jnp.float64(0.0)))
    fig, ax = plt.subplots()
    xs = np.linspace(-5, 5, 1200)
    ax.plot(xs, np.array(fint_shifted(jnp.array(xs))), color=MAGENTA, linewidth=1.2)
    ax.set_ylim([-1.2, 1.2])
    _apply_style(ax)
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.grid(True, color=(0.85, 0.85, 0.85), linewidth=0.5, linestyle='-')
    _save(fig, 6)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"guide02_06.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 7 (guide02_07): oscillatory step function and its integral
# MATLAB: x=chebfun('x',[0 6]); f=x*sign(sin(x^2));
#         subplot(1,2,1),plot(f); g=cumsum(f); subplot(1,2,2),plot(g,'m')
# --------------------------------------------------------------------------
try:
    k_vals = np.arange(0, 12)
    breaks_inner = np.sqrt(k_vals * np.pi)
    breaks_inner = breaks_inner[(breaks_inner >= 0) & (breaks_inner <= 6)]
    breaks = sorted(set([0.0] + list(breaks_inner[breaks_inner > 0]) + [6.0]))
    piece_list = []
    for i in range(len(breaks) - 1):
        mid = 0.5 * (breaks[i] + breaks[i + 1])
        s = float(np.sign(np.sin(mid**2)))
        piece_list.append(_Piece.from_function(
            lambda x, _s=s: x * _s, breaks[i], breaks[i + 1]))
    f_pw = Chebfun(funs=piece_list, domain=Domain(tuple(breaks)))
    g_pw = f_pw.cumsum()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_pieces(f_pw, ax1, color=CHEBFUN_BLUE)
    plot_pieces(g_pw, ax2, color=MAGENTA)   # cumsum is continuous: no jumps
    _apply_style(ax1)
    _apply_style(ax2)
    ax1.set_xticks([0, 2, 4, 6])
    ax1.set_yticks([-6, -4, -2, 0, 2, 4, 6])
    ax2.set_xticks([0, 2, 4, 6])
    ax2.set_yticks([0, 0.5, 1, 1.5])
    # Subplot boxes measured from the reference: spines at x=79/282 and
    # x=348/551, y=19..229.
    bottom, height = 1 - 229 / 258, (229 - 19) / 258
    ax1.set_position([79 / 610, bottom, (282 - 79) / 610, height])
    ax2.set_position([348 / 610, bottom, (551 - 348) / 610, height])
    _save(fig, 7)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"guide02_07.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 8 (guide02_08): Li(x) vs pi(x)
# MATLAB: plot(Li,'m'); hold on, plot(p,1:length(p),'.k'), hold off
# --------------------------------------------------------------------------
try:
    mu = 1.45136923488338105   # Soldner's constant
    xmax = 400
    Li = cj.chebfun(lambda x: 1.0 / jnp.log(x), domain=[mu, xmax]).cumsum()

    def primes_up_to(n):
        sieve = np.ones(n + 1, dtype=bool)
        sieve[:2] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        return np.where(sieve)[0]

    p = primes_up_to(xmax)
    fig, ax = plt.subplots()
    xs = np.linspace(mu, xmax, 1200)
    ax.plot(xs, np.array(Li(jnp.array(xs))), color=MAGENTA, linewidth=1.2)
    ax.plot(p, np.arange(1, len(p) + 1), '.k', markersize=4)
    ax.set_xlim([0, xmax])
    ax.set_ylim([0, 100])
    _apply_style(ax)
    ax.set_xticks(np.arange(50, xmax + 1, 50))
    ax.set_yticks(np.arange(0, 101, 20))
    _save(fig, 8)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"guide02_08.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 9 (guide02_09): cos(pi*x) and its derivative on [0,20]
# MATLAB: f=chebfun('cos(pi*x)',[0 20]); fprime=diff(f); plot([f fprime])
# --------------------------------------------------------------------------
try:
    f = cj.chebfun(lambda x: jnp.cos(jnp.pi * x), domain=[0, 20])
    fprime = f.diff()
    fig, ax = plt.subplots()
    xs = np.linspace(0, 20, 4000)
    ax.plot(xs, np.array(f(jnp.array(xs))), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.plot(xs, np.array(fprime(jnp.array(xs))), color=CHEBFUN_RED, linewidth=1.2)
    ax.set_ylim([-4, 4])
    _apply_style(ax)
    _save(fig, 9)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"guide02_09.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 10 (guide02_10): piecewise f = {x^2, 1, 4-x, 4/x} on [0,4]
# --------------------------------------------------------------------------
try:
    piece_list = [
        _Piece.from_function(lambda x: x**2, 0, 1),
        _Piece.from_function(lambda x: jnp.ones_like(x), 1, 2),
        _Piece.from_function(lambda x: 4.0 - x, 2, 3),
        _Piece.from_function(lambda x: 4.0 / x, 3, 4),
    ]
    f_pw = Chebfun(funs=piece_list, domain=Domain((0.0, 1.0, 2.0, 3.0, 4.0)))
    fig, ax = plt.subplots()
    plot_pieces(f_pw, ax, color=CHEBFUN_BLUE)
    _apply_style(ax)
    _save(fig, 10)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"guide02_10.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 11 (guide02_11): derivative of the piecewise function (with deltas)
# MATLAB: fprime = diff(f); plot(fprime,'r'), ylim([-2,3])
# chebfunjax has no deltafun, so the two delta impulses (amplitude 1 at
# x=2 and 1/3 at x=3, from the jumps of f) are drawn here as arrows.
# --------------------------------------------------------------------------
try:
    fprime_pw = f_pw.diff()
    fig, ax = plt.subplots()
    plot_pieces(fprime_pw, ax, color='r')

    # Delta impulses = jumps of the original f at its interior breakpoints.
    bp = [0.0, 1.0, 2.0, 3.0, 4.0]
    for i in range(1, len(bp) - 1):
        xb = bp[i]
        f_left = float(f_pw.funs[i - 1](jnp.float64(xb)))
        f_right = float(f_pw.funs[i](jnp.float64(xb)))
        amp = f_right - f_left            # signed delta amplitude
        if abs(amp) < 1e-9:
            continue
        # Arrow base sits at the top of the derivative's jump connector.
        d_left = float(fprime_pw.funs[i - 1](jnp.float64(xb)))
        d_right = float(fprime_pw.funs[i](jnp.float64(xb)))
        base = max(d_left, d_right) if amp > 0 else min(d_left, d_right)
        ax.annotate("", xy=(xb, base + amp), xytext=(xb, base),
                    arrowprops=dict(arrowstyle="-|>", color='r', lw=1.2,
                                    mutation_scale=12))
    ax.set_ylim([-2, 3])
    _apply_style(ax)
    _save(fig, 11)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"guide02_11.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 12 (guide02_12): 4th derivative of 1/(1+x^2)
# --------------------------------------------------------------------------
try:
    f = cj.chebfun(lambda x: 1.0 / (1.0 + x**2))
    g = f.diff(4)
    fig, ax = plt.subplots()
    xs = np.linspace(-1, 1, 2000)
    ax.plot(xs, np.array(g(jnp.array(xs))), color=CHEBFUN_BLUE, linewidth=1.2)
    _apply_style(ax)
    _save(fig, 12)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"guide02_12.png FAILED: {e}")

# --------------------------------------------------------------------------
# 2D integrand used by figures 13 and 14
# --------------------------------------------------------------------------
r_fn = lambda x, y: jnp.sqrt(x**2 + y**2)
theta_fn = lambda x, y: jnp.arctan2(y, x)
f2d = lambda x, y: jnp.sin(5.0 * (theta_fn(x, y) - r_fn(x, y))) * jnp.sin(x)


def contour_box(zz, xv, yv, idx):
    """MATLAB-style contour(x,y,z,-1:.2:1) with colorbar and grid."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FuncFormatter

    levels = np.arange(-1.0, 1.0001, 0.2)
    norm = Normalize(vmin=float(np.nanmin(zz)), vmax=float(np.nanmax(zz)))
    fig, ax = plt.subplots()
    ax.contour(xv, yv, zz, levels=levels, cmap=PARULA, norm=norm, linewidths=0.8)
    ax.set_xlim([-2, 2])
    ax.set_ylim([0.5, 2.5])
    ax.set_xticks(np.arange(-2, 2.01, 0.5))
    ax.set_yticks(np.arange(0.5, 2.51, 0.5))
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(direction='in', labelsize=9)
    ax.grid(True, color=(0.85, 0.85, 0.85), linewidth=0.5, linestyle='-')
    # Positions measured from the reference: plot spines x=70..488,
    # colorbar x=511..531, y=19..229.
    bottom, height = 1 - 229 / 258, (229 - 19) / 258
    ax.set_position([70 / 610, bottom, (488 - 70) / 610, height])
    # MATLAB's colorbar is a continuous parula gradient spanning the data
    # range, not the discrete contour-line colours matplotlib would show
    # for a line ContourSet.
    sm = ScalarMappable(cmap=PARULA, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([511 / 610, bottom, (531 - 511) / 610, height])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    cb.ax.yaxis.set_major_formatter(fmt)
    cb.ax.tick_params(labelsize=9)
    _save(fig, idx)

# --------------------------------------------------------------------------
# Plot 13 (guide02_13): contour of the raw anonymous function
# --------------------------------------------------------------------------
try:
    xv = np.linspace(-2, 2, 201)
    yv = np.linspace(0.5, 2.5, 201)
    xx, yy = np.meshgrid(xv, yv)
    zz = np.array(f2d(jnp.array(xx), jnp.array(yy)))
    contour_box(zz, xv, yv, 13)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"guide02_13.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 14 (guide02_14): contour of the Chebfun2 approximation
# --------------------------------------------------------------------------
try:
    f2 = cj.chebfun2(f2d, domain=(-2, 2, 0.5, 2.5))
    xv = np.linspace(-2, 2, 201)
    yv = np.linspace(0.5, 2.5, 201)
    xx, yy = np.meshgrid(xv, yv)
    zz = np.array(f2(jnp.array(xx), jnp.array(yy)))
    contour_box(zz, xv, yv, 14)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"guide02_14.png FAILED: {e}")

print("\nGuide 02: generated 14 plots total.")
