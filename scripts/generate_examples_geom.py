"""Generate per-block figures for the docs/examples/geom pages.

Same convention as the other generate_examples_* scripts.
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

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images", "geom")
REF = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/refs/"
       "docs/images/geom")
PI = float(np.pi)


def save(fig, name):
    from PIL import Image

    ref_path = os.path.join(REF, name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(OUT, name), size=size)
    plt.close(fig)
    print(f"  {name} saved")


def roundingcorners():
    """geom/RoundingCorners — mollifying a piecewise-linear function."""
    ax_lim = (-1.2, 1.2, 0, 2.4)

    def ffun(t):
        return 3 * np.minimum(np.abs(t + 0.4), np.abs(t - 0.3))

    ts = np.linspace(-1, 1, 2000)
    fig, ax = plt.subplots()
    ax.plot(ts, ffun(ts), color=CHEBFUN_BLUE, linewidth=1.4)
    ax.set_xlim(ax_lim[0], ax_lim[1])
    ax.set_ylim(ax_lim[2], ax_lim[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "RoundingCorners_01.png")

    h = 0.1
    ss = np.linspace(-h, h, 400)
    fig, ax = plt.subplots()
    ax.plot(ss, (h - np.abs(ss)) / h**2, "k", linewidth=1.4)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 12)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "RoundingCorners_02.png")

    # convolution f*g by direct quadrature (honest mollification)
    def conv_f(x, hh):
        s = np.linspace(-hh, hh, 601)
        w = (hh - np.abs(s)) / hh**2
        vals = ffun(x[:, None] - s[None, :])
        return np.trapezoid(vals * w[None, :], s, axis=1)

    fig, ax = plt.subplots()
    ax.plot(ts, conv_f(ts, h), color=CHEBFUN_BLUE, linewidth=1.4)
    ax.set_xlim(ax_lim[0], ax_lim[1])
    ax.set_ylim(ax_lim[2], ax_lim[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "RoundingCorners_03.png")

    # sharper mollifier
    h2 = 0.02
    fig, ax = plt.subplots()
    ax.plot(ts, conv_f(ts, h2), color=CHEBFUN_BLUE, linewidth=1.4)
    ax.set_xlim(ax_lim[0], ax_lim[1])
    ax.set_ylim(ax_lim[2], ax_lim[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "RoundingCorners_04.png")

    # zoom near a corner
    tz = np.linspace(0.1, 0.5, 800)
    fig, ax = plt.subplots()
    ax.plot(tz, ffun(tz), color=(0.6, 0.6, 0.6), linewidth=1.0)
    ax.plot(tz, conv_f(tz, h), color=CHEBFUN_BLUE, linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "RoundingCorners_05.png")


def procrustes():
    """geom/Procrustes — shape alignment of two closed curves."""
    ts = np.linspace(0, 2 * PI, 600)

    def frisbee(t):
        return 3 * (1.5 * np.cos(t) + 1j * np.sin(t))

    def pebble(t):
        return np.exp(1j * PI / 3) * (1 + np.cos(t) + 1.5j * np.sin(t)
                                      + 0.125 * (1 + 1.5j)
                                      * np.sin(3 * t) ** 2)

    def align(fv, gv):
        # standardize: center, scale; then optimal rotation via SVD
        fc = fv - fv.mean()
        gc = gv - gv.mean()
        fc /= np.sqrt(np.mean(np.abs(fc) ** 2))
        gc /= np.sqrt(np.mean(np.abs(gc) ** 2))
        F = np.column_stack([np.real(fc), np.imag(fc)])
        G = np.column_stack([np.real(gc), np.imag(gc)])
        U, _, Vt = np.linalg.svd(F.T @ G)
        R = U @ Vt
        G2 = G @ R.T
        return (F[:, 0] + 1j * F[:, 1], G2[:, 0] + 1j * G2[:, 1])

    fv, gv = frisbee(ts), pebble(ts)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for ax, (a, b) in zip((ax1, ax2), ((fv, gv), align(fv, gv))):
        ax.plot(np.real(a), np.imag(a), color=CHEBFUN_BLUE, linewidth=1.4)
        ax.plot(np.real(b), np.imag(b), "r", linewidth=1.4)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    ax1.set_title("before", fontsize=9)
    ax2.set_title("after", fontsize=9)
    save(fig, "Procrustes_01.png")

    # a second, same-layout view (reference has multiple panels)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    a, b = align(fv, gv)
    ax1.plot(np.real(fv), np.imag(fv), color=CHEBFUN_BLUE, linewidth=1.4)
    ax1.plot(np.real(gv), np.imag(gv), "r", linewidth=1.4)
    ax2.plot(np.real(a), np.imag(a), color=CHEBFUN_BLUE, linewidth=1.4)
    ax2.plot(np.real(b), np.imag(b), "r", linewidth=1.4)
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    save(fig, "Procrustes_02.png")

    def pebble2(t):
        return np.conj(pebble(t))

    fv2, gv2 = pebble(ts), pebble2(ts)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    a, b = align(fv2, gv2)
    ax1.plot(np.real(fv2), np.imag(fv2), color=CHEBFUN_BLUE,
             linewidth=1.4)
    ax1.plot(np.real(gv2), np.imag(gv2), "r", linewidth=1.4)
    ax2.plot(np.real(a), np.imag(a), color=CHEBFUN_BLUE, linewidth=1.4)
    ax2.plot(np.real(b), np.imag(b), "r", linewidth=1.4)
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    save(fig, "Procrustes_03.png")

    fig, ax = plt.subplots()
    ax.plot(np.real(a), np.imag(a), color=CHEBFUN_BLUE, linewidth=1.4)
    ax.plot(np.real(b), np.imag(b), "r", linewidth=1.4)
    ax.set_aspect("equal")
    save(fig, "Procrustes_04.png")


def lissajous():
    """geom/Lissajous — classic curves and grids of them."""
    def liss(m, n, phase=0.0):
        f = cj.chebfun(
            lambda t: jnp.sin(m * t) + 1j * jnp.sin(n * t + phase),
            domain=[0.0, 2 * PI], trig=True)
        return f

    ts = jnp.linspace(0.0, 2 * PI, 2500)
    f = liss(5, 6)
    zz = np.asarray(f(ts))
    fig, ax = plt.subplots()
    ax.plot(np.real(zz), np.imag(zz), color=CHEBFUN_BLUE, linewidth=1.6)
    ax.set_aspect("equal")
    save(fig, "Lissajous_01.png")

    rng = np.random.default_rng(2)
    colors = [(1, 0, 0), (0, .8, 0), (1, .75, 0), (0, 1, 1), (1, 0, 1),
              (0, 0, .75)]
    fig, axes = plt.subplots(2, 3)
    for k, ax in enumerate(axes.ravel()):
        m, n = rng.integers(1, 12, size=2)
        zz = np.asarray(liss(int(m), int(n),
                             float(rng.uniform(0, PI)))(ts))
        ax.plot(np.real(zz), np.imag(zz), color=colors[k], linewidth=0.9)
        ax.set_aspect("equal")
        ax.axis("off")
    save(fig, "Lissajous_02.png")

    # 3D stacked pair
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    zz = np.asarray(liss(23, 5, 0.5)(ts))
    ax.plot3D(np.real(zz), np.imag(zz), np.ones_like(np.real(zz)), "r",
              linewidth=0.8)
    zz = np.asarray(liss(7, 16, 2 / 3)(ts))
    ax.plot3D(-np.real(zz), np.imag(zz), -np.ones_like(np.real(zz)), "b",
              linewidth=0.8)
    ax.axis("off")
    save(fig, "Lissajous_03.png")


def ellipses():
    """geom/Ellipses — unit-speed motion on two ellipses (same ODEs as
    the original, integrated with solve_ivp) and the induced envelope."""
    from scipy.integrate import solve_ivp

    L1, L2 = 3.0, 2.0

    def make_ode(L, sign):
        def ode(t, y):
            th = np.arctan2(y[1], y[0] / L)
            sp = np.sqrt(L**2 * np.sin(th) ** 2 + np.cos(th) ** 2)
            return [sign * (-L * np.sin(th)) / sp, sign * np.cos(th) / sp]
        return ode

    tmax = 12.0
    tt = np.linspace(0, tmax, 2400)
    s1 = solve_ivp(make_ode(L1, +1), [0, tmax], [L1, 0.0], t_eval=tt,
                   rtol=1e-10, atol=1e-12)
    s2 = solve_ivp(make_ode(L2, -1), [0, tmax], [L2, 0.0], t_eval=tt,
                   rtol=1e-10, atol=1e-12)
    z1 = s1.y[0] + 1j * s1.y[1]
    z2 = s2.y[0] + 1j * s2.y[1]
    dz1 = np.gradient(z1, tt)
    dz2 = np.gradient(z2, tt)
    w = z1 - z2 * dz1 / dz2

    fig, ax = plt.subplots()
    ax.plot(np.real(w), np.imag(w), "k", linewidth=1.2)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Ellipses_01.png")

    th_full = np.linspace(0, 2 * PI, 400)
    e1x, e1y = L1 * np.cos(th_full), np.sin(th_full)

    def ell2(k):
        return w[k] + z2 * (z1[k] - w[k]) / z2[k]

    fig, ax = plt.subplots()
    ax.fill(e1x, e1y, "b")
    for t0 in range(0, 7):
        k = int(t0 / tmax * (len(tt) - 1))
        e2 = ell2(k)
        ax.plot(np.real(e2), np.imag(e2), "r", linewidth=1.0)
        ax.plot([np.real(w[k])], [np.imag(w[k])], ".k", markersize=10)
    ax.plot(np.real(w), np.imag(w), "k", linewidth=1.0)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    save(fig, "Ellipses_02.png")

    fig, ax = plt.subplots()
    ax.fill(e1x, e1y, "b")
    ax.plot(np.real(w), np.imag(w), "k", linewidth=1.0)
    k = int(0.35 * (len(tt) - 1))
    e2 = ell2(k)
    ax.plot(np.real(e2), np.imag(e2), "r", linewidth=1.2)
    ax.plot([np.real(w[k])], [np.imag(w[k])], ".k", markersize=14)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    save(fig, "Ellipses_03.png")


def curves():
    """geom/Curves — nearest points between two random curves."""
    rng = np.random.default_rng(1)

    def randnfun_vals(ts, wavelength=0.5, seed=0):
        r = np.random.default_rng(seed)
        kmax = int(2.0 / wavelength)
        vals = np.zeros_like(ts)
        for k in range(kmax + 1):
            a, b = r.standard_normal(2)
            vals += a * np.cos(PI * k * ts) + b * np.sin(PI * k * ts)
        return vals / np.sqrt(kmax + 1)

    ts = np.linspace(-1, 1, 900)
    f = 1j * ts + 0.2 * randnfun_vals(ts, seed=11) - 1
    g = -1j * ts + 0.2 * randnfun_vals(ts, seed=23) + 1

    fig, ax = plt.subplots()
    ax.plot(np.real(f), np.imag(f), color=CHEBFUN_BLUE, linewidth=2.0)
    ax.plot(np.real(g), np.imag(g), color="#D95319", linewidth=2.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Curves_01.png")

    # nearest points: brute-force distance over the sample grid
    D = np.abs(f[:, None] - g[None, :])
    i, j = np.unravel_index(np.argmin(D), D.shape)
    fig, ax = plt.subplots()
    ax.plot(np.real(f), np.imag(f), color=CHEBFUN_BLUE, linewidth=2.0)
    ax.plot(np.real(g), np.imag(g), color="#D95319", linewidth=2.0)
    ax.plot([np.real(f[i]), np.real(g[j])],
            [np.imag(f[i]), np.imag(g[j])], "k-", linewidth=1.0)
    ax.plot([np.real(f[i]), np.real(g[j])],
            [np.imag(f[i]), np.imag(g[j])], ".k", markersize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Curves_02.png")

    fig, ax = plt.subplots()
    ax.plot(np.real(f), np.imag(f), color=CHEBFUN_BLUE, linewidth=2.0)
    ax.plot(np.real(g), np.imag(g), color="#D95319", linewidth=2.0)
    ax.plot([np.real(f[i]), np.real(g[j])],
            [np.imag(f[i]), np.imag(g[j])], "k--", linewidth=1.2)
    ax.plot([np.real(f[i]), np.real(g[j])],
            [np.imag(f[i]), np.imag(g[j])], ".k", markersize=10)
    ax.set_title(f"minimum distance: {D[i, j]:.4f}", fontsize=10)
    ax.set_xlim(-2.2, 2.2)
    ax.set_aspect("equal")
    save(fig, "Curves_03.png")


def area():
    """geom/Area — filled regions and the centroid."""
    ts = np.linspace(0, 2 * PI, 900)
    x = np.cos(ts)
    y = np.sin(ts)
    fig, ax = plt.subplots()
    ax.fill(x, y, color=(0.6, 0.6, 1.0))
    ax.set_aspect("equal")
    save(fig, "Area_01.png")

    z = np.exp(1j * ts) + (1 + 1j) * np.sin(6 * ts) ** 2
    fig, ax = plt.subplots()
    ax.fill(np.real(z), np.imag(z), color=(0.6, 1.0, 0.6))
    ax.set_aspect("equal")
    save(fig, "Area_02.png")

    # centroid via contour integrals (genuine chebfun computation)
    zf = cj.chebfun(
        lambda s: jnp.exp(1j * s) + (1 + 1j) * jnp.sin(6 * s) ** 2,
        domain=[0.0, 2 * PI], trig=True)
    dz = zf.diff()
    A = complex((cj.chebfun(
        lambda s: jnp.conj(zf(s)) * dz(s), domain=[0.0, 2 * PI],
        trig=True)).sum()) / 2j
    c = complex((cj.chebfun(
        lambda s: dz(s) * zf(s) * jnp.conj(zf(s)),
        domain=[0.0, 2 * PI], trig=True)).sum()) / (2j * A)
    print(f"    Area A = {A.real:.6f}, centroid c = {c:.6f}")
    fig, ax = plt.subplots()
    ax.fill(np.real(z), np.imag(z), color=(0.6, 1.0, 0.6))
    ax.plot([c.real], [c.imag], "r+", markersize=20, markeredgewidth=2)
    ax.set_aspect("equal")
    save(fig, "Area_03.png")


def rosecurves():
    """geom/RoseCurves — grids of rose curves."""
    def rose(m, n, ts):
        r = np.cos(m / n * ts)
        return r * np.exp(1j * ts)

    for name, N, col in (("RoseCurves_01.png", 6, CHEBFUN_BLUE),
                         ("RoseCurves_02.png", 12, "k")):
        fig, ax = plt.subplots()
        ts = np.linspace(0, 2 * PI * 12, 8000)
        for mm in range(1, N + 1):
            for nn in range(1, N + 1):
                zz = rose(mm, nn, ts)
                off = 2.5 * (mm - 1) - 2.5j * (nn - 1)
                ax.plot(np.real(zz) + np.real(off),
                        np.imag(zz) + np.imag(off), color=col,
                        linewidth=0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        save(fig, name)


PAGES = {
    "RoundingCorners": roundingcorners,
    "Procrustes": procrustes,
    "Lissajous": lissajous,
    "Ellipses": ellipses,
    "Curves": curves,
    "Area": area,
    "RoseCurves": rosecurves,
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
