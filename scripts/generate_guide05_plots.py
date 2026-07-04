"""Generate all 21 plots for Guide Chapter 5 (Complex Chebfuns).

Files are saved as docs/images/guide/guide05_NN.png in the order of the
figures on https://www.chebfun.org/docs/guide/guide05.html, each at the
reference render's exact pixel size (610x258).

Everything is drawn from genuine complex-valued chebfuns (native
complex support); piecewise paths use breakpoint domains (the
join-equivalent), and text figures use chebfunjax's scribble.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import chebfunjax as cj
from chebfunjax.plotting import CHEBFUN_BLUE, chebfun_style, save_chebfun_figure
from chebfunjax.utils.scribble import scribble

chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images", "guide")
os.makedirs(OUT, exist_ok=True)
SIZE = (610, 258)
plot_idx = 0

MAGENTA = (1.0, 0.0, 1.0)  # MATLAB 'm'


def save(fig):
    global plot_idx
    plot_idx += 1
    path = os.path.join(OUT, f"guide05_{plot_idx:02d}.png")
    save_chebfun_figure(fig, path, size=SIZE)
    plt.close(fig)
    print(f"  guide05_{plot_idx:02d}.png saved")


def _path_samples(f, m=150):
    """Sample a (possibly piecewise) complex chebfun per piece."""
    funs = getattr(f, "funs", None)
    if funs is not None and len(funs) > 1:
        chunks = []
        for p in funs:
            a, b = (float(v) for v in p.interval)
            ts = np.linspace(a, b, m)
            chunks.append(np.array(p(jnp.array(ts))))
        return np.concatenate(chunks)
    a, b = (float(v) for v in (f.domain.breakpoints[0], f.domain.breakpoints[-1]))
    return np.array(f(jnp.linspace(a, b, max(m, 200))))


def cplot(ax, f, color=CHEBFUN_BLUE, lw=1.2, n=200, **kw):
    """Plot a complex chebfun's image curve, per piece."""
    ys = _path_samples(f, m=n)
    ax.plot(np.real(ys), np.imag(ys), color=color, linewidth=lw, **kw)
    ax.set_aspect("equal")


def join_paths(segs):
    """join()-equivalent: piecewise complex chebfun; segment k, mapped
    from a local parameter in [0,1], occupies [k, k+1] globally."""

    def piecewise(t):
        val = segs[0](t - 0.0)
        for k in range(1, len(segs)):
            val = jnp.where(t > k, segs[k](t - k), val)
        return val

    return cj.chebfun(piecewise,
                      domain=[float(k) for k in range(len(segs) + 1)])


# --------------------------------------------------------------------------
# Fig 1: 20 points exp(1i*s) on upper half circle (dots)
# --------------------------------------------------------------------------
try:
    s20 = np.linspace(0, np.pi, 20)
    fig, ax = plt.subplots()
    pts = np.exp(1j * s20)
    ax.plot(np.real(pts), np.imag(pts), ".", color=CHEBFUN_BLUE,
            markersize=6)
    ax.set_aspect("equal")
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Fig 2: continuous curve exp(1i*s), s in [0, pi]
# --------------------------------------------------------------------------
f = None
try:
    f = cj.chebfun(lambda s: jnp.exp(1j * s), domain=[0.0, float(np.pi)])
    fig, ax = plt.subplots()
    cplot(ax, f)
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Fig 3: same curve with its Chebyshev data points, '.-'
# --------------------------------------------------------------------------
try:
    fig, ax = plt.subplots()
    cplot(ax, f)
    n = len(f)
    tt = 0.5 * float(np.pi) * (np.cos(np.pi * np.arange(n) / (n - 1)) + 1.0)
    zz = np.array(f(jnp.array(tt)))
    ax.plot(np.real(zz), np.imag(zz), ".", color=CHEBFUN_BLUE, markersize=6)
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Fig 4: g = s*exp(10i*s) and h = exp(2i*s)+.3*exp(20i*s)
# --------------------------------------------------------------------------
g = h = None
try:
    dom_pi = [0.0, float(np.pi)]
    g = cj.chebfun(lambda s: s * jnp.exp(10j * s), domain=dom_pi)
    h = cj.chebfun(
        lambda s: jnp.exp(2j * s) + 0.3 * jnp.exp(20j * s), domain=dom_pi
    )
    fig, (ax1, ax2) = plt.subplots(1, 2)
    cplot(ax1, g, n=600)
    cplot(ax2, h, n=800)
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Fig 5: g^2 and exp(h)
# --------------------------------------------------------------------------
try:
    exp_h = cj.chebfun(
        lambda s: jnp.exp(jnp.exp(2j * s) + 0.3 * jnp.exp(20j * s)),
        domain=dom_pi,
    )
    fig, (ax1, ax2) = plt.subplots(1, 2)
    cplot(ax1, g * g, n=800)
    cplot(ax2, exp_h, n=800)
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Fig 6: piecewise path z and z^2 (grid on)
# --------------------------------------------------------------------------
try:
    z = join_paths([
        lambda s: (1 + 0.5j) * s,
        lambda s: 1 + 0.5j - 2 * s,
    ])
    fig, (ax1, ax2) = plt.subplots(1, 2)
    cplot(ax1, z)
    ax1.grid(True, alpha=0.4, linewidth=0.4)
    cplot(ax2, z * z)
    ax2.grid(True, alpha=0.4, linewidth=0.4)
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Fig 7: rectangle R (blue) and cross X (red), left subplot
# --------------------------------------------------------------------------
R = X = None
try:
    R = join_paths([
        lambda s: 1 + s,
        lambda s: 2 + 2j * s,
        lambda s: 2 + 2j - s,
        lambda s: 1 + 2j - 2j * s,
    ])
    X = join_paths([
        lambda s: 1.3 + 1.5j + 0.4 * s,
        lambda s: 1.5 + 1.3j + 0.4j * s,
    ])
    fig, (ax1, ax2) = plt.subplots(1, 2)
    cplot(ax1, R, lw=2.2)
    cplot(ax1, X, color="r", lw=2.2)
    ax1.grid(True, alpha=0.4, linewidth=0.4)
    ax2.set_visible(False)
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Fig 8: R^2, X^2 and exp(R), exp(X)
# --------------------------------------------------------------------------
try:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    cplot(ax1, R * R, lw=1.5)
    cplot(ax1, X * X, color="r", lw=2.2)
    ax1.grid(True, alpha=0.4, linewidth=0.4)
    eR = np.exp(_path_samples(R, m=200))
    eX = np.exp(_path_samples(X, m=200))
    ax2.plot(np.real(eR), np.imag(eR), color=CHEBFUN_BLUE, linewidth=1.5)
    ax2.plot(np.real(eX), np.imag(eX), color="r", linewidth=2.2)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.4, linewidth=0.4)
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Collection S: grid of vertical/horizontal lines for figs 9-11
# --------------------------------------------------------------------------
S_lines = []
for d in np.arange(-1.0, 1.01, 0.2):
    dd = float(d)
    S_lines.append(cj.chebfun(lambda x, _d=dd: _d + 1j * x))
    S_lines.append(cj.chebfun(lambda x, _d=dd: 1j * _d + x))

_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _plot_lines(ax, fn=None, n=300, lw=1.0, shift=0.0):
    xs = np.linspace(-1.0, 1.0, n)
    for i, line in enumerate(S_lines):
        zz = np.array(line(jnp.array(xs)))
        if fn is not None:
            zz = fn(zz)
        zz = zz + shift
        ax.plot(np.real(zz), np.imag(zz),
                color=_CYCLE[i % len(_CYCLE)], linewidth=lw)
    ax.set_aspect("equal")


# Fig 9: the grid S itself (left subplot)
try:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    _plot_lines(ax1)
    ax2.set_visible(False)
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

# Fig 10: exp(S) and tan(S)
try:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    _plot_lines(ax1, fn=np.exp)
    _plot_lines(ax2, fn=np.tan)
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

# Fig 11: S, 1.6+exp(S), 6.6+tan(S) side by side, axis off
try:
    fig, ax = plt.subplots()
    _plot_lines(ax)
    _plot_lines(ax, fn=np.exp, shift=1.6)
    _plot_lines(ax, fn=np.tan, shift=6.6)
    ax.set_axis_off()
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Figs 12-13: Möbius iterations of a square: curves, then filled regions
# --------------------------------------------------------------------------
def _square_samples(m=150):
    sq = join_paths([
        lambda s: -0.5j + s,
        lambda s: 1 - 0.5j + 1j * s,
        lambda s: 1 + 0.5j - s,
        lambda s: 0.5j - 1j * s,
    ])
    return _path_samples(sq, m=m)


GOLD = (np.sqrt(5) - 1) / 2

try:
    fig, ax = plt.subplots()
    zz = _square_samples()
    curves = [zz]
    for _ in range(3):
        zz = 1.0 / (1.0 + zz)
        curves.append(zz)
    for i, c in enumerate(curves):
        ax.plot(np.real(c), np.imag(c), color=_CYCLE[i % len(_CYCLE)],
                linewidth=1.2)
    ax.plot([GOLD], [0.0], ".k", markersize=5)
    ax.set_aspect("equal")
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

try:
    fig, ax = plt.subplots()
    cur = _square_samples()
    ax.fill(np.real(cur), np.imag(cur), color=(0.5, 0.5, 1.0))
    for col in [(0.5, 1.0, 0.5), (1.0, 0.5, 0.5), (0.5, 1.0, 1.0)]:
        cur = 1.0 / (1.0 + cur)
        ax.fill(np.real(cur), np.imag(cur), color=col)
    ax.plot([GOLD], [0.0], ".k", markersize=5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Fig 14: keyhole contour
# --------------------------------------------------------------------------
try:
    c1, c2 = -2 + 0.05j, -0.2 + 0.05j
    c3, c4 = -0.2 - 0.05j, -2 - 0.05j
    # MATLAB's c2*c3.^s./c2.^s uses SEPARATE principal powers, which sends
    # the arcs the long way around the origin — that is what makes the
    # contour a keyhole. (c3/c2)**s would take the short arc across the
    # branch cut instead.
    L1, L2, L3, L4 = (np.log(c) for c in (c1, c2, c3, c4))
    z_key = join_paths([
        lambda s: c1 + s * (c2 - c1),
        lambda s: jnp.exp((1 - s) * L2 + s * L3),
        lambda s: c3 + s * (c4 - c3),
        lambda s: jnp.exp((1 - s) * L4 + s * L1),
    ])
    fig, ax = plt.subplots()
    cplot(ax, z_key, n=1200)
    ax.set_axis_off()
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Fig 15: scribble('Oxford University')
# --------------------------------------------------------------------------
f_text = None
try:
    f_text = scribble("Oxford University")
    fig, ax = plt.subplots()
    cplot(ax, f_text, lw=2.0, n=24)
    ax.set_xlim(-1.1, 1.1)
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Fig 16: exp(3i*f) of the text (magenta)
# --------------------------------------------------------------------------
try:
    zz = _path_samples(f_text, m=24)
    ww = np.exp(3j * zz)
    fig, ax = plt.subplots()
    ax.plot(np.real(ww), np.imag(ww), color=MAGENTA, linewidth=2.0)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Fig 17: text in a box
# --------------------------------------------------------------------------
box = None
try:
    box = join_paths([
        lambda s: -1.1 - 0.05j + 2.2 * s,
        lambda s: 1.1 - 0.05j + 0.22j * s,
        lambda s: 1.1 + 0.17j - 2.2 * s,
        lambda s: -1.1 + 0.17j - 0.22j * s,
    ])
    fig, ax = plt.subplots()
    cplot(ax, f_text, lw=2.0, n=24)
    cplot(ax, box, lw=2.0)
    ax.set_xlim(-1.2, 1.2)
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Figs 18-19: conformal images of the boxed text
# --------------------------------------------------------------------------
try:
    fig, ax = plt.subplots()
    for path in (f_text, box):
        zz = _path_samples(path, m=24)
        ww = np.exp((1 + 0.2j) * zz)
        ax.plot(np.real(ww), np.imag(ww), color=CHEBFUN_BLUE, linewidth=2.0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

try:
    fig, ax = plt.subplots()
    for path in (f_text, box):
        zz = _path_samples(path, m=24)
        ww = np.tan(zz)
        ax.plot(np.real(ww), np.imag(ww), color=CHEBFUN_BLUE, linewidth=2.0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Figs 20-21: Happy Birthday Pafnuty (mapped, red) with ellipse (blue)
# --------------------------------------------------------------------------
try:
    f_hb = scribble("Happy Birthday Pafnuty!")
    zz_t = _path_samples(f_hb, m=24)
    circle = 1.12 * np.exp(2j * np.pi * np.linspace(0.0, 1.0, 900))
    ellipse = (1.2 * (circle + 1.0 / circle) / 2
               + 1j * np.mean(np.imag(zz_t)))

    def g_map(w):
        return np.exp(-2.2j + (2.5j + 0.4) * w)

    fig, ax = plt.subplots()
    ax.plot(np.real(g_map(zz_t)), np.imag(g_map(zz_t)), "r", linewidth=2.0)
    ax.plot(np.real(g_map(ellipse)), np.imag(g_map(ellipse)), "b",
            linewidth=2.0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

try:
    fig, ax = plt.subplots()
    ax.plot(np.real(zz_t), np.imag(zz_t), "r", linewidth=2.0)
    ax.plot(np.real(ellipse), np.imag(ellipse), "b", linewidth=1.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide05_{plot_idx:02d}.png FAILED: {e}")

print(f"\nGuide 05: generated {plot_idx} plots.")
