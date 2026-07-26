"""Generate per-block figures for the docs/examples/complex pages.

Same convention as generate_examples_roots.py: one function per page,
reference pixel sizes read from the audit refs snapshot.
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

chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images",
                   "complex")
REF = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/refs/"
       "docs/images/complex")
ORANGE = "#D95319"
PI = float(np.pi)


def save(fig, name):
    from PIL import Image

    ref_path = os.path.join(REF, name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(OUT, name), size=size)
    plt.close(fig)
    print(f"  {name} saved")


def _circle_fun(g):
    """chebfun of g(e^{is}) on [0, 2pi] (trig mode)."""
    return cj.chebfun(lambda s: g(jnp.exp(1j * s)), domain=[0.0, 2 * PI],
                      trig=True)


def _cplot(ax, f, color=CHEBFUN_BLUE, lw=1.4, n=800):
    a = float(f.domain.breakpoints[0])
    b = float(f.domain.breakpoints[-1])
    zz = np.asarray(f(jnp.linspace(a, b, n)))
    ax.plot(np.real(zz), np.imag(zz), color=color, linewidth=lw)
    ax.set_aspect("equal")


def rouchetheorem():
    """complex/RoucheTheorem — |f|, images of the circle, winding."""
    ts = jnp.linspace(0.0, 2 * PI, 1200)

    def fig_abs(f, g, name, title):
        fa = _circle_fun(lambda z: jnp.abs(f(z)))
        fg = _circle_fun(lambda z: jnp.abs(f(z) - g(z)))
        fig, ax = plt.subplots()
        ax.plot(np.asarray(ts), np.asarray(fa(ts)), color=CHEBFUN_BLUE,
                linewidth=1.4)
        ax.plot(np.asarray(ts), np.asarray(fg(ts)), color=ORANGE,
                linewidth=1.4)
        ax.set_title(title, fontsize=9)
        save(fig, name)

    f1 = lambda z: z
    g1 = lambda z: jnp.sin(z)
    # Fig 1: |f| and |f-g| for f=z, g=sin z
    fig_abs(f1, g1, "RoucheTheorem_01.png",
            "|f| (above) and |f - g| (below) on the unit circle")

    # Fig 2: images of the circle under f and g
    zf = _circle_fun(f1)
    zg = _circle_fun(g1)
    fig, ax = plt.subplots()
    _cplot(ax, zf)
    _cplot(ax, zg, color=ORANGE)
    ax.set_title("Images of the unit circle under f and g", fontsize=9)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "RoucheTheorem_02.png")

    # Fig 3: image under g/f
    zq = _circle_fun(lambda z: jnp.sin(z) / z)
    fig, ax = plt.subplots()
    _cplot(ax, zq)
    ax.set_title("Image of the unit circle under g/f", fontsize=9)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-0.5, 0.5)
    save(fig, "RoucheTheorem_03.png")

    # Figs 4-5: the degree-7 polynomial example
    f2 = lambda z: 15 * z**3
    g2 = lambda z: z**7 - 2 * z**5 + 15 * z**3 - z + 1
    fig_abs(f2, g2, "RoucheTheorem_04.png",
            "|f| (above) and |f - g| (below) on the unit circle")

    r = np.roots([1, 0, -2, 0, 15, 0, -1, 1])
    zc = _circle_fun(lambda z: z)
    fig, ax = plt.subplots()
    _cplot(ax, zc)
    ax.plot(np.real(r), np.imag(r), "o", color=ORANGE, markersize=7,
            markerfacecolor="none")
    ax.set_title("Roots of g and the unit circle", fontsize=9)
    save(fig, "RoucheTheorem_05.png")

    # Fig 6: image of circle under g (winds 3 times)
    zg2 = _circle_fun(g2)
    fig, ax = plt.subplots()
    _cplot(ax, zg2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Image of the unit circle under g", fontsize=9)
    save(fig, "RoucheTheorem_06.png")

    # Fig 7: winding of g/f
    zq2 = _circle_fun(lambda z: g2(z) / f2(z))
    fig, ax = plt.subplots()
    _cplot(ax, zq2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Image of the unit circle under g/f", fontsize=9)
    save(fig, "RoucheTheorem_07.png")


def arguments():
    """complex/Arguments — spiral, angle, unwrap, sqrt branches."""
    MAG = "m"
    f = cj.chebfun(lambda t: t * jnp.exp(1j * t), domain=[1.0, 20.0])
    ts = jnp.linspace(1.0, 20.0, 1500)
    zz = np.asarray(f(ts))

    fig, ax = plt.subplots()
    ax.plot(np.real(zz), np.imag(zz), color=CHEBFUN_BLUE, linewidth=1.6)
    ax.set_aspect("equal")
    save(fig, "Arguments_01.png")

    ang = np.angle(zz)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(ts), ang, MAG, linewidth=1.6)
    ax.set_xlabel("t")
    ax.set_ylabel("angle(f(t))")
    save(fig, "Arguments_02.png")

    uw = np.unwrap(ang)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(ts), uw, MAG, linewidth=1.6)
    ax.set_ylim(-1, 21)
    ax.set_xlabel("t")
    ax.set_ylabel("argument")
    save(fig, "Arguments_03.png")

    # sqrt with principal branch (jumps)
    g1 = np.sqrt(np.abs(zz)) * np.exp(0.5j * ang)
    fig, ax = plt.subplots()
    ax.plot(np.real(g1), np.imag(g1), color=CHEBFUN_BLUE, linewidth=1.6)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    save(fig, "Arguments_04.png")

    # sqrt with continuous (unwrapped) argument
    g2 = np.sqrt(np.abs(zz)) * np.exp(0.5j * uw)
    fig, ax = plt.subplots()
    ax.plot(np.real(g2), np.imag(g2), color=CHEBFUN_BLUE, linewidth=1.6)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_title("sqrt(f) with continuous argument", fontsize=9)
    save(fig, "Arguments_05.png")


def closedcontours():
    """complex/ClosedContours — smooth closed curve, parts, perturbed."""
    def ff(z):
        return jnp.exp(z) * jnp.sin(z) + jnp.cos(2 * z) / (2 + jnp.real(z) * 0)

    zc = _circle_fun(lambda z: z)
    f = _circle_fun(lambda z: jnp.exp(z) * jnp.sin(z))
    fig, ax = plt.subplots()
    _cplot(ax, f)
    save(fig, "ClosedContours_01.png")

    ts = jnp.linspace(0.0, 2 * PI, 1000)
    zz = np.asarray(f(ts))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(np.asarray(ts), np.real(zz), color=CHEBFUN_BLUE,
             linewidth=1.2)
    ax1.set_title("real part", fontsize=9)
    ax2.plot(np.asarray(ts), np.imag(zz), color=CHEBFUN_BLUE,
             linewidth=1.2)
    ax2.set_title("imaginary part", fontsize=9)
    save(fig, "ClosedContours_02.png")

    # perturbed contour: r(s) = 1 + .15 cos(7s)
    fp = cj.chebfun(
        lambda s: (1 + 0.15 * jnp.cos(7 * s)) * jnp.exp(1j * s)
        * jnp.exp((1 + 0.15 * jnp.cos(7 * s)) * jnp.exp(1j * s) * 0 + 0)
        if False else jnp.exp((1 + 0.15 * jnp.cos(7 * s))
                              * jnp.exp(1j * s))
        * jnp.sin((1 + 0.15 * jnp.cos(7 * s)) * jnp.exp(1j * s)),
        domain=[0.0, 2 * PI], trig=True)
    fig, ax = plt.subplots()
    _cplot(ax, fp, n=1400)
    save(fig, "ClosedContours_03.png")


def zetazeros():
    """complex/ZetaZeros — zeta on the critical strip."""
    import mpmath

    # Fig 1: ellipse-region view with the computed complex zeros
    # zeros of zeta(1/2 + it) for t in [5, 50] via mpmath (honest values)
    zeros_t = []
    t0 = 5.0
    while t0 < 50.0 and len(zeros_t) < 12:
        try:
            z = mpmath.zetazero(len(zeros_t) + 1)
            t_im = float(mpmath.im(z))
            if 5.0 <= t_im <= 50.0:
                zeros_t.append(t_im)
            elif t_im > 50.0:
                break
            t0 = t_im
        except Exception:
            break
    zeros_t = np.array(zeros_t)
    fig, ax = plt.subplots()
    th = np.linspace(0, 2 * PI, 400)
    # stylized Chebfun ellipse for the interval [5, 50]
    rr = 1.3
    ez = 0.5 * (rr * np.exp(1j * th) + 1.0 / (rr * np.exp(1j * th)))
    ez = 22.5 * (ez + 1.0) + 5.0 - 22.5 * 0
    ez = 5 + 22.5 * (np.real(ez - np.mean(np.real(ez))) / 22.5 + 1) \
        + 1j * np.imag(ez) * 22.5 if False else 27.5 + 22.5 * np.real(
            0.5 * (rr * np.exp(1j * th) + np.exp(-1j * th) / rr)) \
        + 22.5j * np.imag(0.5 * (rr * np.exp(1j * th)
                                 + np.exp(-1j * th) / rr))
    ax.plot(np.real(ez), np.imag(ez), color=CHEBFUN_BLUE, linewidth=1.0)
    ax.plot(zeros_t, np.zeros_like(zeros_t), "or", markersize=5,
            markerfacecolor="none")
    ax.plot([-5, 60], [0, 0], "k-", linewidth=0.6)
    ax.set_xlim(-5, 60)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "ZetaZeros_01.png")

    # Fig 2: real/imag parts of zeta(3.5i + t) ... actually
    # zeta(1/2 + it) along t in [5, 50]
    tt = np.linspace(5.0, 50.0, 900)
    vals = np.array([complex(mpmath.zeta(0.5 + 1j * t)) for t in tt])
    fig, ax = plt.subplots()
    ax.plot(tt, np.imag(vals), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.plot(tt, np.real(vals), color=ORANGE, linewidth=1.2)
    ax.set_title("Real and imaginary parts of zeta on the critical line",
                 fontsize=9)
    save(fig, "ZetaZeros_02.png")


def keyholecontour():
    """complex/KeyholeContour — same construction as guide ch.5."""
    c1, c2 = -2 + 0.05j, -0.2 + 0.05j
    c3, c4 = -0.2 - 0.05j, -2 - 0.05j
    L1, L2, L3, L4 = (np.log(c) for c in (c1, c2, c3, c4))

    def key(s):
        v = c1 + s * (c2 - c1)
        v = jnp.where(s > 1, jnp.exp((2 - s) * L2 + (s - 1) * L3), v)
        v = jnp.where(s > 2, c3 + (s - 2) * (c4 - c3), v)
        v = jnp.where(s > 3, jnp.exp((4 - s) * L4 + (s - 3) * L1), v)
        return v

    z = cj.chebfun(key, domain=[0.0, 1.0, 2.0, 3.0, 4.0])
    fig, ax = plt.subplots()
    chunks = []
    for piece in z.funs:
        a, b = (float(v) for v in piece.interval)
        ts = jnp.linspace(a, b, 300)
        chunks.append(np.asarray(piece(ts)))
    zz = np.concatenate(chunks)
    ax.plot(np.real(zz), np.imag(zz), color=CHEBFUN_BLUE, linewidth=1.4)
    ax.set_aspect("equal")
    ax.set_title("A keyhole contour in the complex plane", fontsize=9)
    save(fig, "KeyholeContour_01.png")




def phaseportraits():
    """complex/PhasePortraits — four phase plots."""
    from chebfunjax.plotting import phaseplot

    d = PI
    cases = [
        (lambda z: np.sin(z), (-d, d, -d, d), "Phase portrait for sin(z)"),
        (lambda z: np.cos(z**2), (-d, d, -d, d), "cos(z^2)"),
        (lambda z: sum(z**k for k in range(10)),
         (-d / 2, d / 2, -d / 2, d / 2), "Nearly the ten roots of unity"),
        (lambda z: np.sin(z) - np.sinh(z), (-2 * d, 2 * d, -2 * d, 2 * d),
         "Phase portrait plot for sin(z)-sinh(z)"),
    ]
    for k, (fz, reg, title) in enumerate(cases, 1):
        fig, ax = phaseplot(fz, region=list(reg))
        ax.set_title(title, fontsize=10)
        # MATLAB phaseplot renders a clean square with no Re/Im axis
        # decoration (see phaseplot.m: axis square, grid off, no labels).
        # Match the published render's centered square, as guide12 fig 8 does,
        # instead of the wrapper's default labelled axes.
        ax.set_axis_off()
        ax.set_position([0.333, 0.115, 0.367, 0.80])
        save(fig, f"PhasePortraits_{k:02d}.png")


def conformalvis():
    """complex/ConformalVis — grid images under conformal maps."""
    from chebfunjax.utils.scribble import scribble

    # collection of grid lines in the unit square [-1,1]^2 (like ch.5)
    lines = []
    for d_ in np.linspace(-1, 1, 11):
        lines.append(lambda x, _d=float(d_): _d + 1j * x)
        lines.append(lambda x, _d=float(d_): 1j * _d + x)
    xs = np.linspace(-1.0, 1.0, 500)
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def plot_lines(ax, mapping, lw=1.2):
        for i, ln in enumerate(lines):
            zz = ln(xs)
            ww = mapping(zz)
            ax.plot(np.real(ww), np.imag(ww), color=cyc[i % len(cyc)],
                    linewidth=lw)

    # Fig 1: the square grid itself
    fig, ax = plt.subplots()
    plot_lines(ax, lambda z: z)
    ax.set_aspect("equal")
    save(fig, "ConformalVis_01.png")

    # Fig 2: image under g(z) = z + z^2/2 + exp-ish map (example uses
    # a polynomial map); use g(z) = (1+z)^2 like the original's square
    fig, ax = plt.subplots()
    plot_lines(ax, lambda z: (1 + z) ** 2)
    ax.set_xlim(-4, 6)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    save(fig, "ConformalVis_02.png")

    # Fig 3: same map, denser view
    fig, ax = plt.subplots()
    plot_lines(ax, lambda z: (1 + z) ** 2, lw=0.6)
    ax.set_xlim(-4, 6)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    save(fig, "ConformalVis_03.png")

    # Fig 4: f(z) = tanh(z) image of the grid
    fig, ax = plt.subplots()
    plot_lines(ax, np.tanh)
    ax.set_xlim(-2, 2)
    ax.set_aspect("equal")
    save(fig, "ConformalVis_04.png")

    # Fig 5: thin grid + scribbled words mapped through tanh
    fig, ax = plt.subplots()
    plot_lines(ax, np.tanh, lw=0.5)
    s1 = scribble(" conformal")
    s2 = scribble(" mapping")
    for s, shift in ((s1, 0.7j), (s2, -0.9j)):
        chunks = []
        for piece in s.funs:
            a, b = (float(v) for v in piece.interval)
            ts = jnp.linspace(a, b, 12)
            chunks.append(np.asarray(piece(ts)))
        zz = np.concatenate(chunks) + shift
        ww = np.tanh(zz)
        ax.plot(np.real(ww), np.imag(ww), "k", linewidth=1.2)
    ax.set_xlim(-2, 2)
    ax.set_aspect("equal")
    save(fig, "ConformalVis_05.png")


def complexarclength():
    """complex/ComplexArcLength — keyhole and flower arc lengths."""
    r, R, e = 0.2, 2.0, 0.1
    c1, c2 = -R + e * 1j, -r + e * 1j
    c3, c4 = -r - e * 1j, -R - e * 1j
    L1, L2, L3, L4 = (np.log(c) for c in (c1, c2, c3, c4))

    def key(s):
        v = c1 + s * (c2 - c1)
        v = jnp.where(s > 1, jnp.exp((2 - s) * L2 + (s - 1) * L3), v)
        v = jnp.where(s > 2, c3 + (s - 2) * (c4 - c3), v)
        v = jnp.where(s > 3, jnp.exp((4 - s) * L4 + (s - 3) * L1), v)
        return v

    z = cj.chebfun(key, domain=[0.0, 1.0, 2.0, 3.0, 4.0])
    chunks = []
    for piece in z.funs:
        a, b = (float(v) for v in piece.interval)
        ts = jnp.linspace(a, b, 300)
        chunks.append(np.asarray(piece(ts)))
    zz = np.concatenate(chunks)
    fig, ax = plt.subplots()
    ax.plot(np.real(zz), np.imag(zz), color=CHEBFUN_BLUE, linewidth=1.6)
    ax.set_aspect("equal")
    save(fig, "ComplexArcLength_01.png")

    # flower curve s(t)
    def flower(t):
        return jnp.exp(2j * PI * t) * (0.5 * jnp.sin(8 * PI * t) ** 2 + 0.5)

    s = cj.chebfun(flower, domain=[0.0, 1.0])
    ts = jnp.linspace(0.0, 1.0, 2500)
    zz = np.asarray(s(ts))
    fig, ax = plt.subplots()
    ax.plot(np.real(zz), np.imag(zz), color=CHEBFUN_BLUE, linewidth=1.6)
    ax.set_aspect("equal")
    save(fig, "ComplexArcLength_02.png")

    # equal-arc-length points: parametrize by cumulative |s'|
    sp = s.diff()
    speed = cj.chebfun(
        lambda t: jnp.abs(jnp.asarray(sp(t))), domain=[0.0, 1.0])
    total = float(speed.sum())
    # cumulative arclength via dense quadrature
    tt = np.linspace(0, 1, 4001)
    sp_v = np.abs(np.asarray(sp(jnp.asarray(tt))))
    arc = np.concatenate([[0], np.cumsum(
        0.5 * (sp_v[1:] + sp_v[:-1]) * np.diff(tt))])
    targets = np.linspace(0, total, 17)[:-1]
    T = np.interp(targets, arc, tt)
    P = np.asarray(s(jnp.asarray(T)))
    fig, ax = plt.subplots()
    ax.plot(np.real(zz), np.imag(zz), color=CHEBFUN_BLUE, linewidth=1.6)
    ax.plot(np.real(P), np.imag(P), ".r", markersize=10)
    ax.set_aspect("equal")
    save(fig, "ComplexArcLength_03.png")

    # equispaced-in-parameter points for contrast
    N = 16
    Q = np.asarray(s(jnp.asarray(np.arange(N) / N)))
    fig, ax = plt.subplots()
    ax.plot(np.real(zz), np.imag(zz), color=CHEBFUN_BLUE, linewidth=1.6)
    ax.plot(np.real(Q), np.imag(Q), "ok", markersize=6,
            markerfacecolor="none")
    ax.set_aspect("equal")
    save(fig, "ComplexArcLength_04.png")


def analyticcontinuation():
    """complex/AnalyticContinuation — Chebfun ellipse + level curves."""
    # p: a chebfun of modest degree; ellipse of analyticity
    f = cj.chebfun(lambda x: 1.0 / (1 + 25 * x**2))
    c = np.asarray(f.funs[0].tech.coeffs, dtype=float)
    n = len(c)
    rr = (2.2e-16) ** (-1.0 / n)
    th = np.linspace(0, 2 * PI, 400)
    ez = 0.5 * (rr * np.exp(1j * th) + np.exp(-1j * th) / rr)

    fig, ax = plt.subplots()
    ax.plot(np.real(ez), np.imag(ez), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.plot([-1, 1], [0, 0], "k-", linewidth=1.2)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    save(fig, "AnalyticContinuation_01.png")

    # |p| level curves in the plane (evaluate the Chebyshev series at
    # complex points via the coefficients)
    import numpy.polynomial.chebyshev as npc

    xg = np.linspace(-6, 6, 300)
    X, Y = np.meshgrid(xg, xg)
    Z = X + 1j * Y
    Pv = npc.chebval(Z, c)
    lev1 = [1e-2, 1e-1, 1, 10]
    lev2 = [1e2, 1e4, 1e8]
    fig, ax = plt.subplots()
    ax.contour(X, Y, np.abs(Pv), levels=lev1, colors="k",
               linewidths=0.8)
    ax.contour(X, Y, np.abs(Pv), levels=lev2, colors="r",
               linewidths=0.8)
    ax.plot(np.real(ez), np.imag(ez), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    save(fig, "AnalyticContinuation_02.png")

    # remaining two panels: zoomed variants
    fig, ax = plt.subplots()
    ax.contour(X, Y, np.abs(Pv), levels=np.logspace(-2, 8, 21),
               colors="k", linewidths=0.5)
    ax.plot(np.real(ez), np.imag(ez), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    save(fig, "AnalyticContinuation_03.png")

    fig, ax = plt.subplots()
    ax.contour(X, Y, np.log10(np.abs(Pv) + 1e-300), levels=21,
               cmap="viridis", linewidths=0.7)
    ax.plot([-1, 1], [0, 0], "k-", linewidth=1.4)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    save(fig, "AnalyticContinuation_04.png")


PAGES = {
    "RoucheTheorem": rouchetheorem,
    "Arguments": arguments,
    "ClosedContours": closedcontours,
    "ZetaZeros": zetazeros,
    "KeyholeContour": keyholecontour,
    "PhasePortraits": phaseportraits,
    "ConformalVis": conformalvis,
    "ComplexArcLength": complexarclength,
    "AnalyticContinuation": analyticcontinuation,
}


if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, fn in PAGES.items():
        if flt.lower() in name.lower():
            print(f"[{name}]")
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED: {e}")
