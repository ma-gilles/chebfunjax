"""Generate all plots for Guide Chapter 3: Rootfinding and Minima and Maxima.

Faithfully translates every figure from Chebfun Guide Chapter 3
(https://www.chebfun.org/docs/guide/guide03.html) to chebfunjax/Python.

Each figure is exported at the exact pixel size of its chebfun.org reference
render (600x270) with the MATLAB-default axes box position, so it can be
compared pixel-for-pixel against the reference.  The MATLAB commands that
produce each figure are quoted above each block.

Axis limits and tick locations are set explicitly to the values MATLAB
produced in the reference renders (measured from the reference images), so
the figures do not depend on the library's automatic tick heuristic.

Several MATLAB Chebfun features used in this chapter are not yet available
in chebfunjax; where that is the case the figure is reproduced with the
public chebfunjax API plus matplotlib/NumPy, and the workaround is noted in
a comment.  See the accompanying report for the list of library gaps.
"""

import matplotlib

matplotlib.use('Agg')

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain
from chebfunjax.plotting import (
    CHEBFUN_BLUE,
    _apply_style,
    chebfun_style,
    save_chebfun_figure,
)

chebfun_style()

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT_DIR, exist_ok=True)

# Exact reference size (all Guide ch.3 refs are 600x270).
REF_SIZE = (600, 270)

# MATLAB-default axes-box position within the 600x270 canvas, measured from
# the reference renders (black box at L=78, R=542, T=20, B=239 px).
BOX = dict(left=0.130, right=0.903, bottom=0.1148, top=0.9259)

# Subplot (1x2) axes boxes, measured from guide03_07/08 (edges 78,278,342,542).
SUB_AX1 = [78 / 600, 1 - 239 / 270, (278 - 78) / 600, (239 - 20) / 270]
SUB_AX2 = [342 / 600, 1 - 239 / 270, (278 - 78) / 600, (239 - 20) / 270]

# Pure MATLAB single-letter colours (bright red / blue used with 'r','b').
RED = 'r'
BLUE = 'b'
BLACK = 'k'
MAGENTA = 'm'
MATLAB_ORANGE = '#D95319'  # MATLAB colour-order #2 (used by round staircase)

UNIT_TICKS = [-1, -0.5, 0, 0.5, 1]

plot_idx = 0


def _save(fig, idx):
    """Pin axes box + canvas to the MATLAB reference and write the PNG."""
    fig.subplots_adjust(**BOX)
    fig.set_facecolor("white")
    path = os.path.join(OUT_DIR, f'guide03_{idx:02d}.png')
    save_chebfun_figure(fig, path, size=REF_SIZE)
    plt.close(fig)
    print(f"  guide03_{idx:02d}.png saved")


def _save_sub(fig, ax1, ax2, idx):
    """Pin the two subplot boxes and write the PNG."""
    ax1.set_position(SUB_AX1)
    ax2.set_position(SUB_AX2)
    fig.set_facecolor("white")
    path = os.path.join(OUT_DIR, f'guide03_{idx:02d}.png')
    save_chebfun_figure(fig, path, size=REF_SIZE)
    plt.close(fig)
    print(f"  guide03_{idx:02d}.png saved")


def finish(fig, ax, idx, *, xlim=None, ylim=None, xticks=None, yticks=None,
           grid=False):
    """Apply the reference axis limits, ticks and grid, then save."""
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if grid:
        ax.grid(True, color='0.85', linewidth=0.6)
        ax.set_axisbelow(True)
    _save(fig, idx)


# ==========================================================================
# Helpers for features chebfunjax does not yet implement natively.
# ==========================================================================

def build_binary(fexpr, gexpr, op, lo, hi):
    """Piecewise ``op(f, g)`` (op = jnp.maximum / jnp.minimum) on [lo, hi].

    chebfunjax has no two-argument ``min(f,g)`` / ``max(f,g)``; we build the
    result by locating the crossover points (roots of f-g) and constructing a
    smooth piece on each sub-interval (on which one argument strictly
    dominates, so ``op`` is smooth).
    """
    cf = cj.chebfun(fexpr, domain=[lo, hi])
    cg = cj.chebfun(gexpr, domain=[lo, hi])
    r = np.asarray((cf - cg).roots())
    bps = sorted([lo, hi] + [float(v) for v in r if lo < v < hi])
    funs = []
    for a, b in zip(bps[:-1], bps[1:]):
        piece = cj.chebfun(lambda t: op(fexpr(t), gexpr(t)), domain=[a, b])
        funs += piece.funs
    return Chebfun(funs=funs, domain=Domain(tuple(bps)))


def cheb_complex_roots(coeffs, a=-1.0, b=1.0):
    """All complex roots of a single Chebyshev series via the colleague matrix.

    chebfunjax's ``roots`` returns only the real roots inside the domain; the
    MATLAB ``roots(f,'all')`` capability is reproduced here from the public
    Chebyshev coefficients (``f.funs[0].coeffs``) using the colleague-matrix
    eigenvalues [Good 1961], mapped from [-1,1] to [a,b].
    """
    c = np.asarray(coeffs, dtype=np.complex128).copy()
    mx = np.max(np.abs(c)) or 1.0
    n = len(c)
    while n > 1 and abs(c[n - 1]) < 1e-13 * mx:
        n -= 1
    c = c[:n]
    N = len(c) - 1
    if N < 1:
        return np.array([], dtype=np.complex128)
    A = np.zeros((N, N), dtype=np.complex128)
    if N >= 2:
        A[0, 1] = 1.0
    for i in range(1, N - 1):
        A[i, i - 1] = 0.5
        A[i, i + 1] = 0.5
    if N >= 2:
        A[N - 1, N - 2] = 0.5
    A[N - 1, :] -= c[:N] / (2 * c[N])
    ev = np.linalg.eigvals(A)
    return (a + b) / 2 + (b - a) / 2 * ev


def complex_roots_in_ellipse(f, a, b):
    """roots(f,'complex'): colleague roots filtered to the Bernstein ellipse."""
    coeffs = np.asarray(f.funs[0].coeffs)
    allr = cheb_complex_roots(coeffs, a, b)
    c = np.abs(coeffs) / (np.max(np.abs(coeffs)) or 1.0)
    ks = np.arange(len(c))
    m = c > 1e-14
    slope = np.polyfit(ks[m], np.log(c[m]), 1)[0]
    rho = np.exp(-slope)
    z = (allr - (a + b) / 2) / ((b - a) / 2)
    w = np.abs(z + np.sqrt(z ** 2 - 1))
    w = np.maximum(w, 1.0 / w)
    return allr[w < rho]


# ==========================================================================
# Section 3.1  roots
# ==========================================================================

# --------------------------------------------------------------------------
# Plot 1:  plot(p), grid on; hold on, plot(r,p(r),'.r')   with p = x^3+x^2-x
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: polynomial x^3+x^2-x")
    x = cj.chebfun(lambda x: x)
    p = x ** 3 + x ** 2 - x
    r = np.asarray(p.roots())
    fig, ax = cj.plot(p)
    ax.plot(r, [float(p(ri)) for ri in r], '.', color=RED, markersize=5)
    finish(fig, ax, plot_idx, xlim=(-1, 1), ylim=(-0.2, 1.0),
           xticks=UNIT_TICKS, yticks=[-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
           grid=True)
except Exception as e:
    print(f"  guide03_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 2:  plot(Ai,'r'); plot(Bi,'b'); roots; axis([-10 3 -.6 1.5]), grid on
#   Ai = airy(0,x), Bi = airy(2,x)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Airy Ai (red) and Bi (blue)")
    Ai = cj.chebfun(lambda x: jnp.array(sp.airy(np.asarray(x))[0]), domain=[-10, 3])
    Bi = cj.chebfun(lambda x: jnp.array(sp.airy(np.asarray(x))[2]), domain=[-10, 3])
    fig, ax = cj.plot(Ai, color=RED)
    cj.plot_1d(Bi, ax=ax, color=BLUE)
    rA = np.asarray(Ai.roots())
    rB = np.asarray(Bi.roots())
    ax.plot(rA, [float(Ai(ri)) for ri in rA], '.', color=RED, markersize=5)
    ax.plot(rB, [float(Bi(ri)) for ri in rB], '.', color=BLUE, markersize=5)
    finish(fig, ax, plot_idx, xlim=(-10, 3), ylim=(-0.6, 1.5),
           xticks=[-10, -8, -6, -4, -2, 0, 2], yticks=[-0.5, 0, 0.5, 1, 1.5],
           grid=True)
except Exception as e:
    print(f"  guide03_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 3:  f = cheb.gallery('fishfillet') = cos(x).*sin(exp(x)) on [0,6]
#   plot(f); hold on, plot(r,f(r),'.r')
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: fishfillet cos(x)*sin(exp(x))")
    f = cj.chebfun(lambda x: jnp.cos(x) * jnp.sin(jnp.exp(x)), domain=[0, 6])
    r = np.asarray(f.roots())
    fig, ax = cj.plot(f, n_pts=16000)
    ax.plot(r, np.zeros_like(r), '.', color=RED, markersize=5)
    finish(fig, ax, plot_idx, xlim=(0, 6), ylim=(-1, 1),
           xticks=[0, 1, 2, 3, 4, 5, 6], yticks=[-1, -0.5, 0, 0.5, 1])
except Exception as e:
    print(f"  guide03_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 4:  plot(x); plot(cos(x),'k'); r=roots(f-x); plot(r,f(r),'or')  on [-2,2]
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: x and cos(x) intersection")
    x = cj.chebfun(lambda x: x, domain=[-2, 2])
    f = cj.cos(x)
    r = np.asarray((f - x).roots())
    fig, ax = cj.plot(x, color=CHEBFUN_BLUE)
    cj.plot_1d(f, ax=ax, color=BLACK)
    ax.plot(r, [float(f(ri)) for ri in r], 'o', markerfacecolor='none',
            markeredgecolor=RED, markersize=8, markeredgewidth=1.3)
    finish(fig, ax, plot_idx, xlim=(-2, 2), ylim=(-2, 2),
           xticks=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
           yticks=[-2, -1, 0, 1, 2])
except Exception as e:
    print(f"  guide03_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plots 5 & 6:  f = x^3 - 3x - 2 + sign(sin(20x))  on [-2,2]
#   5: plot(f), grid on; r=roots(f); plot(r,0*r,'.r')          (roots at jumps)
#   6: plot(f), grid on; r=roots(f,'nojump'); plot(r,0*r,'.r') (interior only)
#   chebfunjax cannot add a smooth Chebfun to a piecewise sign() Chebfun
#   (binary ops require matching breakpoints), so f is assembled per-piece
#   with the constant sign value on each interval between zeros of sin(20x).
# --------------------------------------------------------------------------
try:
    kmax = int(np.floor(2 * 20 / np.pi))
    bps = sorted([-2.0] + [k * np.pi / 20 for k in range(-kmax, kmax + 1)] + [2.0])
    funs = []
    endpt_vals = []
    for a, b in zip(bps[:-1], bps[1:]):
        cval = float(np.sign(np.sin(20 * 0.5 * (a + b))))
        piece = cj.chebfun(lambda t, c=cval: t ** 3 - 3 * t - 2 + c, domain=[a, b])
        funs += piece.funs
        endpt_vals.append((float(piece(a)), float(piece(b))))
    fpw = Chebfun(funs=funs, domain=Domain(tuple(bps)))
    interior_roots = [float(v) for v in np.asarray(fpw.roots())]
    # roots at jumps: interior breakpoints where f changes sign across the gap
    jump_roots = []
    for j in range(1, len(bps) - 1):
        left = endpt_vals[j - 1][1]
        right = endpt_vals[j][0]
        if left * right < 0:
            jump_roots.append(bps[j])
    roots_all = sorted(interior_roots + jump_roots)

    for plot_idx_local, rset in ((5, roots_all), (6, interior_roots)):
        plot_idx += 1
        print(f"Plot {plot_idx}: x^3-3x-2+sign(sin(20x)) "
              f"({'all roots' if plot_idx_local == 5 else 'nojump'})")
        fig, ax = cj.plot(fpw, n_pts=3000)
        rr = np.asarray(rset)
        ax.plot(rr, np.zeros_like(rr), '.', color=RED, markersize=5)
        finish(fig, ax, plot_idx, xlim=(-2, 2), ylim=(-5, 1),
               xticks=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
               yticks=[-5, -4, -3, -2, -1, 0, 1], grid=True)
except Exception as e:
    plot_idx = 6
    print(f"  guide03_05/06.png FAILED: {e}")

# ==========================================================================
# Section 3.2  min, max, abs, sign, round, floor, ceil
# ==========================================================================

# --------------------------------------------------------------------------
# Plot 7:  subplot(1,2,1) plot(x);  subplot(1,2,2) plot(abs(x))
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: x and abs(x)")
    x = cj.chebfun(lambda x: x)
    absx = cj.abs(x)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    cj.plot_1d(x, ax=ax1, color=CHEBFUN_BLUE)
    ax1.set_xlim(-1, 1); ax1.set_ylim(-1, 1)
    ax1.set_xticks(UNIT_TICKS); ax1.set_yticks(UNIT_TICKS)
    cj.plot_1d(absx, ax=ax2, color=CHEBFUN_BLUE)
    ax2.set_xlim(-1, 1); ax2.set_ylim(0, 1)
    ax2.set_xticks(UNIT_TICKS); ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    _save_sub(fig, ax1, ax2, plot_idx)
except Exception as e:
    print(f"  guide03_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 8:  f = min(x,-x/2)  |  g = max(.6,1-x^2), ylim([.5,1])
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: min(x,-x/2) and max(.6,1-x^2)")
    fmin = build_binary(lambda t: t, lambda t: -t / 2, jnp.minimum, -1, 1)
    gmax = build_binary(lambda t: 1 - t ** 2, lambda t: 0.6 + 0 * t,
                        jnp.maximum, -1, 1)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    cj.plot_1d(fmin, ax=ax1, color=CHEBFUN_BLUE)
    ax1.set_xlim(-1, 1); ax1.set_ylim(-1, 0)
    ax1.set_xticks(UNIT_TICKS); ax1.set_yticks([-1, -0.8, -0.6, -0.4, -0.2, 0])
    cj.plot_1d(gmax, ax=ax2, color=CHEBFUN_BLUE)
    ax2.set_xlim(-1, 1); ax2.set_ylim(0.5, 1)
    ax2.set_xticks(UNIT_TICKS); ax2.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
    _save_sub(fig, ax1, ax2, plot_idx)
except Exception as e:
    print(f"  guide03_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 9:  g = exp(x); plot(g); gh = round(10*g)/10; plot(gh,'jumpline','-')
#   grid on.  round/floor/ceil are not in chebfunjax; the staircase is drawn
#   directly (round(10*exp(x))/10) with solid jumplines in MATLAB orange.
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: exp(x) and round(10*exp(x))/10 staircase")
    g = cj.chebfun(lambda x: jnp.exp(x))
    xs = np.linspace(-1, 1, 4000)
    gv = np.asarray(g(jnp.array(xs)))
    gh = np.round(10 * gv) / 10
    fig, ax = cj.plot(g, color=CHEBFUN_BLUE)
    # staircase with solid horizontal + vertical (jumpline '-') segments
    ax.plot(xs, gh, '-', color=MATLAB_ORANGE, linewidth=1.2)
    finish(fig, ax, plot_idx, xlim=(-1, 1), ylim=(0, 3),
           xticks=UNIT_TICKS, yticks=[0, 0.5, 1, 1.5, 2, 2.5, 3], grid=True)
except Exception as e:
    print(f"  guide03_{plot_idx:02d}.png FAILED: {e}")

# ==========================================================================
# Section 3.3  Local extrema
# ==========================================================================

# --------------------------------------------------------------------------
# Plot 10:  f = exp(real(airy(x))) on [-15,0]; plot(f); r=roots(diff(f));
#           plot(r,f(r),'.r'), grid on
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: exp(real(airy(x))) with extrema")
    f = cj.chebfun(lambda x: jnp.array(np.exp(np.real(sp.airy(np.asarray(x))[0]))),
                   domain=[-15, 0])
    r = np.asarray(f.diff().roots())
    fig, ax = cj.plot(f, n_pts=1200)
    ax.plot(r, [float(f(ri)) for ri in r], '.', color=RED, markersize=5)
    finish(fig, ax, plot_idx, xlim=(-15, 0), ylim=(0.6, 1.8),
           xticks=[-15, -10, -5, 0],
           yticks=[0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8], grid=True)
except Exception as e:
    print(f"  guide03_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plots 11-14:  h = max(exp(x)*sin(30x), 2-6x^2) on [-1,1] and its extrema.
# --------------------------------------------------------------------------
try:
    h = build_binary(lambda t: jnp.exp(t) * jnp.sin(30 * t),
                     lambda t: 2 - 6 * t ** 2, jnp.maximum, -1, 1)

    # local extrema: smooth critical points (roots of diff on each piece) plus
    # interior breakpoint corners plus the two domain endpoints.
    crit = [float(v) for v in np.asarray(h.diff().roots())]
    crit += [bp for bp in h.domain.breakpoints]
    crit = np.array(sorted(crit))
    # dedupe
    keep = np.concatenate([[True], np.diff(crit) > 1e-6])
    extrema = crit[keep]

    def classify_min(p):
        eps = 2e-4
        lo = max(-1.0, p - eps); hi = min(1.0, p + eps)
        vp = float(h(p)); vl = float(h(lo)); vr = float(h(hi))
        return (vp <= vl + 1e-9) and (vp <= vr + 1e-9)

    minima = np.array([p for p in extrema if classify_min(p)])

    def h_curve(idx, *, red=False, circ=False, kdot=False):
        fig, ax = cj.plot(h, n_pts=2500, color=CHEBFUN_BLUE)
        if red:
            ax.plot(extrema, [float(h(p)) for p in extrema], '.', color=RED,
                    markersize=5)
        if circ:
            ax.plot(minima, [float(h(p)) for p in minima], 'o',
                    markerfacecolor='none', markeredgecolor=BLACK,
                    markersize=8, markeredgewidth=1.1)
        if kdot:
            ax.plot(minima, [float(h(p)) for p in minima], '.', color=BLACK,
                    markersize=6)
        finish(fig, ax, idx, xlim=(-1, 1), ylim=(-3, 3), xticks=UNIT_TICKS,
               yticks=[-3, -2, -1, 0, 1, 2, 3])

    plot_idx += 1  # 11
    print(f"Plot {plot_idx}: h = max(exp(x)sin(30x), 2-6x^2)")
    h_curve(plot_idx)

    plot_idx += 1  # 12
    print(f"Plot {plot_idx}: h with all local extrema")
    h_curve(plot_idx, red=True)

    plot_idx += 1  # 13
    print(f"Plot {plot_idx}: h with local minima circled")
    h_curve(plot_idx, red=True, circ=True)

    plot_idx += 1  # 14
    print(f"Plot {plot_idx}: h with local minima (min 'local')")
    h_curve(plot_idx, red=True, circ=True, kdot=True)
except Exception as e:
    print(f"  guide03_11..14.png FAILED: {e}")
    plot_idx = 14

# ==========================================================================
# Section 3.4  Global extrema
# ==========================================================================

# --------------------------------------------------------------------------
# Plots 15 & 16:  f = sin(x)+sin(x^2) on [0,15]
#   15: plot(f,'k')
#   16: plot(f,'k'); plot(minpos,minval,'.b'); plot(maxpos,maxval,'.r')
# --------------------------------------------------------------------------
try:
    f = cj.chebfun(lambda x: jnp.sin(x) + jnp.sin(x ** 2), domain=[0, 15])

    plot_idx += 1  # 15
    print(f"Plot {plot_idx}: sin(x)+sin(x^2)")
    fig, ax = cj.plot(f, color=BLACK, n_pts=3000)
    finish(fig, ax, plot_idx, xlim=(0, 15), ylim=(-2, 2),
           xticks=[0, 5, 10, 15], yticks=[-2, -1, 0, 1, 2])

    plot_idx += 1  # 16
    print(f"Plot {plot_idx}: sin(x)+sin(x^2) with global min/max")
    minpos, minval = f.min()
    maxpos, maxval = f.max()
    fig, ax = cj.plot(f, color=BLACK, n_pts=3000)
    ax.plot(minpos, minval, '.', color=BLUE, markersize=8)
    ax.plot(maxpos, maxval, '.', color=RED, markersize=8)
    finish(fig, ax, plot_idx, xlim=(0, 15), ylim=(-2, 2),
           xticks=[0, 5, 10, 15], yticks=[-2, -1, 0, 1, 2])
except Exception as e:
    print(f"  guide03_15/16.png FAILED: {e}")
    plot_idx = 16

# ==========================================================================
# Section 3.6  Roots in the complex plane
# ==========================================================================

# --------------------------------------------------------------------------
# Plots 17 & 18:  F = 4+sin(x)+sin(sqrt2 x)+sin(pi x) on [-100,100]
#   17: r = roots(f,'complex'); plot(r,'.')
#   18: r2 = roots(f,'complex','norecursion'); plot(r,'om')
#   roots(...,'complex') is not in chebfunjax; reproduced from the Chebyshev
#   coefficients via the colleague matrix + Bernstein-ellipse filter.  The
#   'norecursion' variant is not distinguished (chebfunjax has no recursion),
#   so the same root set is drawn as magenta circles in plot 18.
# --------------------------------------------------------------------------
try:
    def F(x):
        return 4 + jnp.sin(x) + jnp.sin(jnp.sqrt(2) * x) + jnp.sin(jnp.pi * x)
    fF = cj.chebfun(F, domain=[-100, 100])
    r = complex_roots_in_ellipse(fF, -100, 100)

    plot_idx += 1  # 17
    print(f"Plot {plot_idx}: complex roots (plot(r,'.'))")
    fig, ax = plt.subplots()
    ax.plot(r.real, r.imag, '.', color=CHEBFUN_BLUE, markersize=5)
    _apply_style(ax)
    finish(fig, ax, plot_idx, xlim=(-100, 100), ylim=(-1, 1),
           xticks=[-100, -50, 0, 50, 100],
           yticks=[-1, -0.5, 0, 0.5, 1])

    plot_idx += 1  # 18
    print(f"Plot {plot_idx}: complex roots + norecursion (magenta circles)")
    fig, ax = plt.subplots()
    ax.plot(r.real, r.imag, '.', color=CHEBFUN_BLUE, markersize=5)
    ax.plot(r.real, r.imag, 'o', markerfacecolor='none', markeredgecolor=MAGENTA,
            markersize=6, markeredgewidth=1.0)
    _apply_style(ax)
    finish(fig, ax, plot_idx, xlim=(-100, 100), ylim=(-1, 1),
           xticks=[-100, -50, 0, 50, 100],
           yticks=[-1, -0.5, 0, 0.5, 1])
except Exception as e:
    print(f"  guide03_17/18.png FAILED: {e}")

print(f"\nGuide 03 plot generation complete. {plot_idx} plots attempted.")
