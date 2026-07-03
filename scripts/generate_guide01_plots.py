"""Generate all plots for Guide Chapter 1.

Produces one PNG for every code block in the chapter that creates a figure.
Files are saved as docs/images/guide/guide01_NN.png where NN is the sequential
plot number matching the order in the original Chebfun Guide Chapter 1.

Each figure is exported at the exact pixel size of its chebfun.org reference
render (610x258) with the MATLAB-default axes box position, so it can be
compared pixel-for-pixel against the reference.  The MATLAB commands that
produce each figure are quoted above each block.

Axis limits and tick locations are set explicitly to the values MATLAB
produced in the reference renders, so the figures do not depend on the
library's automatic tick heuristic.
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
from chebfunjax.plotting import (
    CHEBFUN_BLUE,
    _apply_style,
    chebfun_style,
    save_chebfun_figure,
)
from chebfunjax.utils.gallery import gallery

chebfun_style()

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT_DIR, exist_ok=True)

# Exact reference size (all Guide ch.1 refs are 610x258).
REF_SIZE = (610, 258)

# MATLAB-default axes-box position within the 610x258 canvas, measured from
# the reference renders (black box at L=79, R=551, T=19/23, B=229 px).
BOX = dict(left=0.130, right=0.903, bottom=0.112, top=0.926)
BOX_TITLED = dict(left=0.130, right=0.903, bottom=0.112, top=0.911)

plot_idx = 0


def _save(fig, idx, titled=False):
    """Pin axes box + canvas to the MATLAB reference and write the PNG."""
    fig.subplots_adjust(**(BOX_TITLED if titled else BOX))
    fig.set_facecolor("white")
    path = os.path.join(OUT_DIR, f'guide01_{idx:02d}.png')
    save_chebfun_figure(fig, path, size=REF_SIZE)
    plt.close(fig)
    print(f"  guide01_{idx:02d}.png saved")


def finish(fig, ax, idx, *, xlim=None, ylim=None, xticks=None, yticks=None,
           title=None):
    """Apply the reference axis limits, ticks and title, then save."""
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if title is not None:
        ax.set_title(title)
    _save(fig, idx, titled=title is not None)


UNIT_TICKS = [-1, -0.5, 0, 0.5, 1]


# --------------------------------------------------------------------------
# Plot 1  (Section 1.2):  plot(f), ylim([-1.2,1.2])   with f = cos(20x)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: cos(20x)")
    f = cj.chebfun(lambda x: jnp.cos(20 * x))
    fig, ax = f.plot()
    finish(fig, ax, plot_idx, ylim=(-1.2, 1.2),
           xticks=UNIT_TICKS, yticks=UNIT_TICKS)
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 2  (Section 1.2):  plot(f,'.-'), ylim([-1.2 1.2])
#   line through the Chebyshev points with dot markers at those points
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: cos(20x) with Chebyshev points")
    f = cj.chebfun(lambda x: jnp.cos(20 * x))
    fig, ax = f.plot()
    n = len(f)
    cheb_pts = -np.cos(np.pi * np.arange(n) / (n - 1))
    cheb_vals = np.asarray(f(jnp.array(cheb_pts)))
    ax.plot(cheb_pts, cheb_vals, '.', color=CHEBFUN_BLUE, markersize=4)
    finish(fig, ax, plot_idx, ylim=(-1.2, 1.2),
           xticks=UNIT_TICKS, yticks=UNIT_TICKS)
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 3  (Section 1.2):  plot(g), ylim([-.5 1])   with g = besselj(0,t) on [0,100]
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Bessel J0 on [0,100]")
    g = cj.chebfun(lambda t: jnp.array(sp.j0(np.asarray(t))), domain=[0, 100])
    fig, ax = g.plot(n_pts=1500)
    finish(fig, ax, plot_idx, ylim=(-0.5, 1.0),
           xticks=[0, 20, 40, 60, 80, 100], yticks=[-0.5, 0, 0.5, 1])
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 4  (Section 1.2):  clf, plot(f)   with f = 1/(1+25x^2) (Runge)
#   Built by operating on the identity chebfun x, as in MATLAB.
#   MATLAB autoscales to [0, 1.2].
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Runge function")
    x = cj.chebfun(lambda x: x)
    f = 1 / (1 + 25 * x**2)
    fig, ax = f.plot()
    finish(fig, ax, plot_idx, ylim=(0.0, 1.2), xticks=UNIT_TICKS,
           yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 5  (Section 1.4):  plot(f)   with f = {x^2, 1, 4-x} on [-1 1 2 4]
#   Chebfun.plot draws each piece separately and the x=2 jump as a dotted
#   connector.  MATLAB autoscales to [0, 2].
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Piecewise x^2, 1, 4-x")
    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.domain import Domain
    f1 = cj.chebfun(lambda x: x**2, domain=[-1, 1])
    f2 = cj.chebfun(lambda x: jnp.ones_like(x), domain=[1, 2])
    f3 = cj.chebfun(lambda x: 4.0 - x, domain=[2, 4])
    f = Chebfun(funs=f1.funs + f2.funs + f3.funs,
                domain=Domain((-1.0, 1.0, 2.0, 4.0)))
    fig, ax = f.plot()
    finish(fig, ax, plot_idx, xlim=(-1, 4), ylim=(0.0, 2.0),
           xticks=[-1, 0, 1, 2, 3, 4], yticks=[0, 0.5, 1, 1.5, 2])
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 6  (Section 1.4):  plot(1/(1+f),'r')
#   red curve, x=2 jump drawn as a dotted connector.  Autoscale [0.3, 1].
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: 1/(1+f) piecewise, red")
    g = 1 / (1 + f)
    fig, ax = g.plot(color='r')
    finish(fig, ax, plot_idx, xlim=(-1, 4), ylim=(0.3, 1.0),
           xticks=[-1, 0, 1, 2, 3, 4], yticks=[0.4, 0.6, 0.8, 1.0])
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 7  (Section 1.4):  f = abs(exp(x).*sin(8*x)); plot(f)
#   6 funs; abs introduces breakpoints at the zeros.  Autoscale [0, 3].
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: abs(exp(x)*sin(8x))")
    x = cj.chebfun(lambda x: x)
    f = (x.exp() * (8 * x).sin()).abs()
    fig, ax = f.plot(n_pts=1200)
    finish(fig, ax, plot_idx, ylim=(0.0, 3.0), xticks=UNIT_TICKS,
           yticks=[0, 0.5, 1, 1.5, 2, 2.5, 3])
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 8  (Section 1.4):  h = max(sin(20x), exp(x-1)); plot(h), ylim([0 1.2])
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: max(sin(20x), exp(x-1))")
    x = cj.chebfun(lambda x: x)
    f_sin = (20 * x).sin()
    g_exp = (x - 1).exp()
    diff_fg = f_sin - g_exp
    r = diff_fg.roots()
    bps = np.sort(np.concatenate([[-1.0], np.asarray(r), [1.0]]))
    h = cj.chebfun(
        lambda t: jnp.maximum(jnp.sin(20 * t), jnp.exp(t - 1)),
        domain=list(bps),
    )
    fig, ax = h.plot(n_pts=1500)
    finish(fig, ax, plot_idx, ylim=(0.0, 1.2), xticks=UNIT_TICKS,
           yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 9  (Section 1.5):  f = exp(-x^2/16).*(1+.2*cos(10x)) on [-inf,inf]; plot(f)
#   chebfunjax has no infinite intervals; the function is < 2e-3 beyond +-10,
#   so we build it on [-10,10], which reproduces the MATLAB display window.
#   Autoscale [0, 1.2].
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Gaussian-modulated cosine (finite approx)")
    f = cj.chebfun(
        lambda x: jnp.exp(-x**2 / 16) * (1 + 0.2 * jnp.cos(10 * x)),
        domain=[-10, 10],
    )
    fig, ax = f.plot(n_pts=2000)
    finish(fig, ax, plot_idx, ylim=(0.0, 1.2),
           xticks=[-10, -5, 0, 5, 10],
           yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 10 (Section 1.5):  h = (1/pi)/sqrt(1-x^2) with endpoint exps; plot(h)
#   chebfunjax has no endpoint singularities; we build the (smooth but steep)
#   function just inside [-1,1] and clip the y-axis the way MATLAB's autoscale
#   does, so the divergence at the endpoints leaves the top of the box.
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Arcsine distribution (smooth approx)")
    delta = 1e-3
    h = cj.chebfun(
        lambda x: (1 / jnp.pi) / jnp.sqrt(1 - x**2),
        domain=[-1 + delta, 1 - delta],
    )
    fig, ax = h.plot(n_pts=2000)
    finish(fig, ax, plot_idx, xlim=(-1, 1), ylim=(0.3, 1.15),
           xticks=UNIT_TICKS, yticks=[0.4, 0.6, 0.8, 1.0])
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 11 (Section 1.6):  f = ff on [-pi,pi]; plot(f)   (Chebyshev basis)
#   Autoscale [-3, 2].
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Periodic function (Chebyshev)")
    ff = lambda t: jnp.sin(t) + jnp.cos(2 * t) - jnp.cos(t) / 3 + jnp.cos(100 * t) / 6
    f = cj.chebfun(ff, domain=[-jnp.pi, jnp.pi])
    fig, ax = f.plot(n_pts=3000)
    finish(fig, ax, plot_idx, xlim=(-np.pi, np.pi), ylim=(-3.0, 2.0),
           xticks=[-3, -2, -1, 0, 1, 2, 3], yticks=[-3, -2, -1, 0, 1, 2])
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 12 (Section 1.6):  plot(f2,'m')   (no trig mode in chebfunjax; magenta)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Periodic function (magenta)")
    ff = lambda t: jnp.sin(t) + jnp.cos(2 * t) - jnp.cos(t) / 3 + jnp.cos(100 * t) / 6
    f2 = cj.chebfun(ff, domain=[-jnp.pi, jnp.pi])
    fig, ax = f2.plot(color='m', n_pts=3000)
    finish(fig, ax, plot_idx, xlim=(-np.pi, np.pi), ylim=(-3.0, 2.0),
           xticks=[-3, -2, -1, 0, 1, 2, 3], yticks=[-3, -2, -1, 0, 1, 2])
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 13 (Section 1.9):  plot(cheb.gallery('airy')), ylim([-.8 .8]); title(...)
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Airy function")
    f = cj.chebfun(lambda x: jnp.array(sp.airy(np.asarray(x))[0]), domain=[-40, 40])
    fig, ax = f.plot(n_pts=2000)
    finish(fig, ax, plot_idx, ylim=(-0.8, 0.8),
           xticks=list(np.arange(-40, 41, 10)), yticks=[-0.5, 0.0, 0.5],
           title='Airy function')
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 14 (Section 1.9):  plot(cheb.gallery('daubechies')), ylim([-.5 1.5]); title(...)
#   chebfunjax has no equispaced ('equi') constructor and no daubechies gallery,
#   and Chebyshev interpolation of this fractal produces endpoint Runge spikes,
#   so we evaluate the D4 (db2) cascade on a fine grid and plot the samples
#   directly, which is what MATLAB's 'equi' chebfun plot does visually.
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Daubechies (cascade approximation)")
    hfilt = np.array([
        (1 + np.sqrt(3)) / (4 * np.sqrt(2)),
        (3 + np.sqrt(3)) / (4 * np.sqrt(2)),
        (3 - np.sqrt(3)) / (4 * np.sqrt(2)),
        (1 - np.sqrt(3)) / (4 * np.sqrt(2)),
    ])
    N = 12
    x_pts = np.linspace(0, 3, 3 * 2**N + 1)
    phi = np.where((x_pts >= 0) & (x_pts <= 3), 1.0, 0.0)
    for _ in range(N):
        phi_new = np.zeros_like(phi)
        for k in range(len(hfilt)):
            shifted = np.interp(2 * x_pts - k, x_pts, phi, left=0, right=0)
            phi_new += np.sqrt(2) * hfilt[k] * shifted
        phi = phi_new
    # Normalise to unit integral (standard db2 normalisation); the cascade
    # preserves the box-init integral (=3), so amplitude is otherwise ~4x large.
    dx = x_pts[1] - x_pts[0]
    phi = phi / (np.sum((phi[:-1] + phi[1:]) * 0.5) * dx)
    fig, ax = plt.subplots()
    ax.plot(x_pts, phi, '-', color=CHEBFUN_BLUE, linewidth=1.2)
    _apply_style(ax)
    finish(fig, ax, plot_idx, xlim=(0, 3), ylim=(-0.5, 1.5),
           xticks=[0, 0.5, 1, 1.5, 2, 2.5, 3], yticks=[-0.5, 0, 0.5, 1, 1.5],
           title='Daubechies scaling function')
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 15 (Section 1.9):  cheb.gallery('zigzag')
#   gallery('zigzag') = cumsum(chebfun(sign(sin(100*t/(2-t))), 10000)), the
#   genuine ATAP "Six myths" example, length 10001.  Title shows the length.
# --------------------------------------------------------------------------
try:
    plot_idx += 1
    print(f"Plot {plot_idx}: Zigzag polynomial")
    f = gallery('zigzag')
    fig, ax = f.plot(n_pts=4000)
    finish(fig, ax, plot_idx, xlim=(-1, 1), ylim=(-0.1, 0.05),
           xticks=UNIT_TICKS, yticks=[-0.1, -0.05, 0.0, 0.05],
           title=f'zigzag, length = {len(f)}')
except Exception as e:
    print(f"  guide01_{plot_idx:02d}.png FAILED: {e}")


print("\nGuide 01 plot generation complete.")
