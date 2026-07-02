"""Generate all plots for Guide Chapter 8: Chebfun Preferences.

Each figure mirrors the corresponding MATLAB render on
https://www.chebfun.org/docs/guide/guide08.html and is exported at the
exact 600x270 px canvas of the reference via ``save_chebfun_figure``.

Notes on library gaps (see the chapter text and the audit report):

* chebfunjax has no automatic ``splitting`` / edge-detection.  Where the
  MATLAB page relies on splitting, the breakpoints are supplied
  explicitly through the ``domain=(a, b1, ..., b)`` tuple -- this is the
  public-API way to build a piecewise chebfun.
* chebfunjax has no ``minSamples`` knob; its adaptive constructor always
  starts on a 17-point grid (== MATLAB factory ``minSamples``), so the
  "spike found / spike missed" behaviour of section 8.6 reproduces
  exactly.  To force a denser grid we pass a fixed ``n=``.
* chebfunjax has no ``resampling`` hook, so grid-dependent "functions"
  like ``length(x)*sin(15*x)`` cannot be built.  Figures 11-12 plot the
  fixed-length function the MATLAB constructor converges to instead.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax.numpy as jnp
import numpy as np
import chebfunjax as cj
from chebfunjax.utils.quadrature import chebpts
from chebfunjax.plotting import chebfun_style, save_chebfun_figure, CHEBFUN_BLUE

chebfun_style()

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "docs", "images", "guide")
os.makedirs(OUTDIR, exist_ok=True)

PI = float(jnp.pi)
SIZE = (600, 270)


def _matlab_ticks(ax):
    """Sparse MATLAB-style tick locations and trailing-zero-free labels."""
    for axis, scale in ((ax.xaxis, ax.get_xscale()),
                        (ax.yaxis, ax.get_yscale())):
        if scale == "linear":
            axis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _save(fig, idx):
    path = os.path.join(OUTDIR, f"guide08_{idx:02d}.png")
    save_chebfun_figure(fig, path, size=SIZE)
    plt.close(fig)
    print(f"  guide08_{idx:02d}.png saved")


def _new():
    fig, ax = plt.subplots()
    return fig, ax


def _dense(f, n=2000):
    bp = f.domain.breakpoints
    xs = np.linspace(float(bp[0]), float(bp[-1]), n)
    ys = np.array(f(jnp.array(xs)))
    return xs, ys


def _crossings(g, a, b, npts=200000):
    """Sign-change crossings of g on [a, b], refined by bisection."""
    ts = np.linspace(a, b, npts)
    vals = g(ts)
    out = []
    for i in range(len(ts) - 1):
        if vals[i] == 0.0:
            out.append(ts[i])
        elif vals[i] * vals[i + 1] < 0:
            lo, hi = ts[i], ts[i + 1]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if g(mid) * g(lo) <= 0:
                    hi = mid
                else:
                    lo = mid
            out.append(0.5 * (lo + hi))
    return np.array(out)


# ==========================================================================
# 8.2  domain -- Lissajous curve of sin(19t) and cos(20t) on [0, 2*pi]
#      MATLAB: chebfunpref.setDefaults('domain',[0 2*pi],'tech',@trigtech)
#              f=chebfun(@(t) sin(19*t)); g=chebfun(@(t) cos(20*t));
#              plot(f,g), axis equal, axis off
# ==========================================================================
f = cj.chebfun(lambda t: jnp.sin(19 * t), domain=(0.0, 2 * PI))
g = cj.chebfun(lambda t: jnp.cos(20 * t), domain=(0.0, 2 * PI))
tt = np.linspace(0.0, 2 * PI, 4000)
xx = np.array(f(jnp.array(tt)))
yy = np.array(g(jnp.array(tt)))
fig, ax = _new()
ax.plot(xx, yy, color=CHEBFUN_BLUE, linewidth=1.0)
# Place the axes on the same 221 px square the MATLAB reference uses
# (centered at col 310, rows 19-240 of the 600x270 canvas) so the dense
# Lissajous crosshatch overlaps the reference pixel-for-pixel.
ax.set_position([199.5 / 600, 30.0 / 270, 221.0 / 600, 221.0 / 270])
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.axis("off")
_save(fig, 1)

# ==========================================================================
# 8.3  splitting -- f = min(|x|, exp(x)/6); f.ends = [-1 -0.1443 0 0.2045 1]
#      chebfunjax has no auto-splitting: pass the breakpoints explicitly.
# ==========================================================================
c1 = _crossings(lambda x: -x - np.exp(x) / 6.0, -1.0, 0.0)[0]   # -x = exp/6
c2 = _crossings(lambda x: x - np.exp(x) / 6.0, 0.0, 1.0)[0]     #  x = exp/6
fmin = cj.chebfun(lambda t: jnp.minimum(jnp.abs(t), jnp.exp(t) / 6.0),
                  domain=(-1.0, float(c1), 0.0, float(c2), 1.0))
xs, ys = _dense(fmin)
fig, ax = _new()
ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.2)
ax.set_xlim(-1, 1)
ax.set_ylim(0.0, 0.5)
_matlab_ticks(ax)
_save(fig, 2)

# ==========================================================================
# 8.3  complicated function ff = sin(x)*tanh(3*exp(x)*sin(15*x)) on [-1,1]
#      MATLAB length 1465; chebfunjax length ~1416 (single global polynomial)
# ==========================================================================
ff = lambda t: jnp.sin(t) * jnp.tanh(3 * jnp.exp(t) * jnp.sin(15 * t))
f3 = cj.chebfun(ff)
xs, ys = _dense(f3, 4000)
fig, ax = _new()
ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.0)
ax.set_xlim(-1, 1)
ax.set_ylim(-0.9, 1.05)
_matlab_ticks(ax)
_save(fig, 3)

# ==========================================================================
# 8.3  same function on the wider domain [-3, 3]  (MATLAB length 17603)
# ==========================================================================
f4 = cj.chebfun(ff, domain=(-3.0, 3.0))
xs, ys = _dense(f4, 6000)
fig, ax = _new()
ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.0)
ax.set_xlim(-3, 3)
_matlab_ticks(ax)
_save(fig, 4)

# ==========================================================================
# 8.5  maxLength -- sign(x) cannot be resolved; build a fixed 65-point interp.
#      MATLAB: f = chebfun('sign(x)',65); plot(f)
# ==========================================================================
fs = cj.chebfun(lambda x: jnp.sign(x), n=65)
xs = np.linspace(-1.0, 1.0, 4000)
ys = np.array(fs(jnp.array(xs)))
fig, ax = _new()
ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.0)
ax.set_xlim(-1, 1)
ax.set_ylim(-1.5, 1.5)
_matlab_ticks(ax)
_save(fig, 5)

# ==========================================================================
# 8.6  minSamples -- bump with exponent 2 IS found (17-point start).
#      MATLAB: f = chebfun('-x -x^2 + exp(-(30*(x-.47))^2)'); length 317
# ==========================================================================
b2 = cj.chebfun(lambda x: -x - x**2 + jnp.exp(-(30 * (x - 0.47))**2))
xs, ys = _dense(b2, 4000)
fig, ax = _new()
ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.0)
ax.set_xlim(-1, 1)
ax.set_ylim(-2.0, 0.5)
_matlab_ticks(ax)
_save(fig, 6)

# ==========================================================================
# 8.6  minSamples -- bump with exponent 4 is MISSED (length 3, just -x-x^2).
#      MATLAB: f = chebfun('-x -x^2 + exp(-(30*(x-.47))^4)'); length 3
# ==========================================================================
b4 = cj.chebfun(lambda x: -x - x**2 + jnp.exp(-(30 * (x - 0.47))**4))
xs, ys = _dense(b4, 4000)
fig, ax = _new()
ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.0)
ax.set_xlim(-1, 1)
ax.set_ylim(-2.0, 0.5)
_matlab_ticks(ax)
_save(fig, 7)

# ==========================================================================
# 8.6  minSamples -- forcing a denser grid recovers the spike.
#      MATLAB used minSamples=33 (-> length 1087); chebfunjax has no
#      minSamples, so we request a fixed dense grid n=1087.
# ==========================================================================
b4b = cj.chebfun(lambda x: -x - x**2 + jnp.exp(-(30 * (x - 0.48))**4), n=1087)
xs, ys = _dense(b4b, 4000)
fig, ax = _new()
ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.0)
ax.set_xlim(-1, 1)
ax.set_ylim(-2.0, 0.5)
_matlab_ticks(ax)
_save(fig, 8)

# ==========================================================================
# 8.6  splitting-mode spikes -- ff = max(.85,sin(x+x^2)) - x/20 on [0,10].
#      Under-resolved (MATLAB splitting default): only the first, widest
#      arches are captured.  Reproduced with breakpoints for the first 5
#      arches; the fast tail collapses to the baseline 0.85 - x/20.
# ==========================================================================
cr = _crossings(lambda t: np.sin(t + t**2) - 0.85, 0.0, 10.0)
keep = cr[:10]                 # first 5 arches (10 crossings)
tcut = float(keep[-1]) + 0.05


def ff9(t):
    full = jnp.maximum(0.85, jnp.sin(t + t**2)) - t / 20.0
    base = 0.85 - t / 20.0
    return jnp.where(t <= tcut, full, base)


bp9 = (0.0, *[float(c) for c in keep], tcut, 10.0)
f9 = cj.chebfun(ff9, domain=bp9)
xs, ys = _dense(f9, 4000)
fig, ax = _new()
ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.0)
ax.set_xlim(0, 10)
ax.set_ylim(0.3, 1.0)
_matlab_ticks(ax)
_save(fig, 9)

# ==========================================================================
# 8.6  same, fully resolved (MATLAB minsamples=33): all arches captured.
# ==========================================================================
ff10 = lambda t: jnp.maximum(0.85, jnp.sin(t + t**2)) - t / 20.0
bp10 = (0.0, *[float(c) for c in cr], 10.0)
f10 = cj.chebfun(ff10, domain=bp10)
xs, ys = _dense(f10, 6000)
fig, ax = _new()
ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.0)
ax.set_xlim(0, 10)
ax.set_ylim(0.3, 1.0)
_matlab_ticks(ax)
_save(fig, 10)

# ==========================================================================
# 8.7  resampling -- MATLAB length(x)*sin(15*x) converges on the 65-point
#      grid, i.e. it equals 65*sin(15*x).  chebfunjax cannot see the grid
#      length, so we plot that limiting function with '.-' markers at its
#      Chebyshev nodes.
# ==========================================================================
f11 = cj.chebfun(lambda x: 65.0 * jnp.sin(15 * x))
xs, ys = _dense(f11, 4000)
L = len(f11)
nodes = np.array(chebpts(L, kind=2))
fig, ax = _new()
ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.0)
ax.plot(nodes, np.array(f11(jnp.array(nodes))), ".", color=CHEBFUN_BLUE,
        markersize=4)
ax.set_xlim(-1, 1)
ax.set_ylim(-68, 68)
_matlab_ticks(ax)
ax.set_yticks([-60, -40, -20, 0, 20, 40, 60])   # match MATLAB step of 20
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
_save(fig, 11)

# ==========================================================================
# 8.7  resampling -- sin(length(x)^(2/3)*x) converges on the 65-point grid
#      to sin(65^(2/3)*x).  Plot with '.-' markers.
# ==========================================================================
freq = 65.0**(2.0 / 3.0)
f12 = cj.chebfun(lambda x: jnp.sin(freq * x))
xs, ys = _dense(f12, 4000)
L = len(f12)
nodes = np.array(chebpts(L, kind=2))
fig, ax = _new()
ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.0)
ax.plot(nodes, np.array(f12(jnp.array(nodes))), ".", color=CHEBFUN_BLUE,
        markersize=4)
ax.set_xlim(-1, 1)
_matlab_ticks(ax)
_save(fig, 12)

print("\nGuide 08: generated 12 plots.")
