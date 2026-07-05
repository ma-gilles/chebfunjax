"""Generate per-block figures for docs/examples/approx pages, tranche 4:
Pushnitski, Inpainting1D, FiltersCF, EdgeDetection, BSplineConv.
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
                   "approx")
REF = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/refs/"
       "docs/images/approx")
ORANGE = "#D95319"
PI = float(np.pi)


def save(fig, name):
    from PIL import Image

    ref_path = os.path.join(REF, name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(OUT, name), size=size)
    plt.close(fig)
    print(f"  {name} saved")


def _chebcoeffs(vals):
    n = len(vals)
    ext = np.concatenate([vals[::-1], vals[1:-1]])
    c = np.real(np.fft.fft(ext)) / (n - 1)
    c = c[:n]
    c[0] /= 2
    c[-1] /= 2
    return c


def pushnitski():
    """approx/Pushnitski — slow log-decay of -1/log(x) coefficients."""
    def f_pos(x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        m = x > 0
        out[m] = -1.0 / np.log(x[m] / 1.0)
        return out

    xs = np.linspace(-0.1, 0.1, 3000)

    fig, ax = plt.subplots()
    ax.plot(xs, f_pos(xs), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("-heaviside(x)/log(x)", fontsize=10)
    save(fig, "Pushnitski_01.png")

    # coefficients of the degree-1000 interpolant on [-0.1, 0.1]
    n = 1000
    xc = 0.1 * np.cos(PI * np.arange(n) / (n - 1))
    c = np.abs(_chebcoeffs(f_pos(xc[::-1])[::-1]))
    fig, ax = plt.subplots()
    kk = np.arange(1, len(c))
    ax.loglog(kk, np.maximum(c[1:], 1e-18), ".", color=CHEBFUN_BLUE,
              markersize=3)
    ax.set_xlim(1, 500)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    ax.set_title("Chebyshev coefficients (loglog)", fontsize=10)
    save(fig, "Pushnitski_02.png")

    # coefficients against 1/(k log^2 k) reference lines
    fig, ax = plt.subplots()
    ax.loglog(kk, np.maximum(c[1:], 1e-18), ".", color=CHEBFUN_BLUE,
              markersize=3)
    ref = 1.0 / (kk * np.log(np.maximum(kk, 2)) ** 2)
    ax.loglog(kk, ref * c[1] / ref[0], "r", linewidth=1.0)
    ax.set_xlim(1, 500)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    ax.set_title("coefficients vs C/(k log^2 k)", fontsize=10)
    save(fig, "Pushnitski_03.png")

    # errors of truncations (very slow convergence)
    ns = (2 ** np.arange(3, 10)).astype(int)
    errs = []
    xe = np.linspace(-0.1, 0.1, 5000)
    fe = f_pos(xe)
    for m in ns:
        xm = 0.1 * np.cos(PI * np.arange(m) / (m - 1))
        cm = _chebcoeffs(f_pos(xm[::-1])[::-1])
        vm = np.polynomial.chebyshev.chebval(xe / 0.1, cm)
        errs.append(np.max(np.abs(fe - vm)))
    fig, ax = plt.subplots()
    ax.loglog(ns, errs, ".-", color=CHEBFUN_BLUE, markersize=8,
              linewidth=1.0)
    ax.loglog(ns, 1.0 / np.log(ns) ** 1, "r--", linewidth=0.8)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    ax.set_xlabel("n")
    ax.set_ylabel("max error")
    save(fig, "Pushnitski_04.png")

    fig, ax = plt.subplots()
    ax.semilogy(xe, np.maximum(np.abs(
        fe - np.polynomial.chebyshev.chebval(xe / 0.1, _chebcoeffs(
            f_pos((0.1 * np.cos(PI * np.arange(256) / 255))[::-1])
            [::-1]))), 1e-18), "k", linewidth=0.5)
    ax.set_title("pointwise error, n = 256", fontsize=10)
    save(fig, "Pushnitski_05.png")


def inpainting1d():
    """approx/Inpainting1D — L1 fitting removes localized corruption."""
    rng = np.random.default_rng(1)

    xs = np.linspace(-1, 1, 2500)

    def smooth_np(x):
        return 0.3 + np.asarray(x) ** 2 + np.exp(0.3 * np.asarray(x))

    # band-limited noise ~ randnfun(.1): wavelength 0.1 -> ~20 modes
    kmax = 20
    coefs = rng.standard_normal((kmax + 1, 2))
    def noise_np(x):
        x = np.asarray(x)
        out = np.zeros_like(x)
        for k in range(kmax + 1):
            out += coefs[k, 0] * np.cos(PI * k * x) \
                + coefs[k, 1] * np.sin(PI * k * x)
        return out / np.sqrt(kmax + 1)

    corrupted = np.maximum(smooth_np(xs), noise_np(xs))
    fig, ax = plt.subplots()
    ax.plot(xs, corrupted, color=CHEBFUN_BLUE, linewidth=1.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("corrupted smooth function", fontsize=10)
    save(fig, "Inpainting1D_01.png")

    deg = 20

    # L1 fit by IRLS
    V = np.polynomial.chebyshev.chebvander(xs, deg)
    w = np.ones_like(xs)
    for _ in range(80):
        W = np.sqrt(w)[:, None]
        c1, *_ = np.linalg.lstsq(V * W, corrupted * W[:, 0], rcond=None)
        r = corrupted - V @ c1
        w = 1.0 / np.maximum(np.abs(r), 1e-9)
    p1 = V @ c1
    fig, ax = plt.subplots()
    ax.plot(xs, p1, color=CHEBFUN_BLUE, linewidth=1.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    err1 = np.max(np.abs(p1 - smooth_np(xs)))
    ax.set_title(f"L1 fit (err {err1:.2e})", fontsize=10)
    save(fig, "Inpainting1D_02.png")

    # L2 fit
    c2, *_ = np.linalg.lstsq(V, corrupted, rcond=None)
    p2 = V @ c2
    fig, ax = plt.subplots()
    ax.plot(xs, p2, color=CHEBFUN_BLUE, linewidth=1.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    err2 = np.max(np.abs(p2 - smooth_np(xs)))
    ax.set_title(f"L2 fit (err {err2:.2e})", fontsize=10)
    save(fig, "Inpainting1D_03.png")

    # Linf fit via LP
    from scipy.optimize import linprog

    A = np.block([[V, -np.ones((len(xs), 1))],
                  [-V, -np.ones((len(xs), 1))]])
    b = np.concatenate([corrupted, -corrupted])
    cv = np.zeros(deg + 2)
    cv[-1] = 1.0
    lp = linprog(cv, A_ub=A, b_ub=b,
                 bounds=[(None, None)] * (deg + 1) + [(0, None)],
                 method="highs")
    pinf = V @ lp.x[:deg + 1]
    fig, ax = plt.subplots()
    ax.plot(xs, pinf, color=CHEBFUN_BLUE, linewidth=1.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    errinf = np.max(np.abs(pinf - smooth_np(xs)))
    ax.set_title(f"Linf fit (err {errinf:.2e})", fontsize=10)
    save(fig, "Inpainting1D_04.png")

    fig, ax = plt.subplots()
    ax.plot(xs, smooth_np(xs), "k", linewidth=1.2, label="smooth")
    ax.plot(xs, p1, color=CHEBFUN_BLUE, linewidth=0.9, label="L1")
    ax.plot(xs, p2, color=ORANGE, linewidth=0.9, label="L2")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Inpainting1D_05.png")
    print(f"    L1 err {err1:.2e} << L2 err {err2:.2e}")


def filterscf():
    """approx/FiltersCF — square-wave filters and near-best approx."""
    def f_np(x):
        x = np.asarray(x)
        return ((np.abs(x) < 0.3).astype(float)
                + (np.abs(x - 0.7) < 0.1).astype(float)
                + (np.abs(x + 0.65) < 0.2).astype(float))

    xs = np.linspace(-1, 1, 4000)
    fig, ax = plt.subplots()
    ax.plot(xs, f_np(xs), "k", linewidth=1.2)
    ax.set_ylim(-0.2, 1.4)
    save(fig, "FiltersCF_01.png")

    # mollified square wave: conv with triangular phi of half-width .02
    h = 0.02
    ss = np.linspace(-h, h, 301)
    phi = 50 - 50**2 * np.abs(ss)

    def f2_np(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        vals = f_np(x[:, None] - ss[None, :])
        return np.trapezoid(vals * phi[None, :], ss, axis=1)

    f2v = f2_np(xs)
    fig, ax = plt.subplots()
    ax.plot(xs, f2v, color=CHEBFUN_BLUE, linewidth=1.0)
    ax.set_ylim(-0.2, 1.4)
    save(fig, "FiltersCF_02.png")

    # near-best polynomial approx (LP grid-minimax; cf() is a backlog)
    from scipy.optimize import linprog

    xg = np.linspace(-1, 1, 1500)
    f2g = f2_np(xg)
    errs = {}
    for m in (100, 200):
        V = np.polynomial.chebyshev.chebvander(xg, m)
        A = np.block([[V, -np.ones((len(xg), 1))],
                      [-V, -np.ones((len(xg), 1))]])
        b = np.concatenate([f2g, -f2g])
        cv = np.zeros(m + 2)
        cv[-1] = 1.0
        lp = linprog(cv, A_ub=A, b_ub=b,
                     bounds=[(None, None)] * (m + 1) + [(0, None)],
                     method="highs")
        errs[m] = (lp.x[:m + 1], lp.x[-1])

    fig, ax = plt.subplots()
    ax.plot(xs, f2v, "k", linewidth=0.8)
    for m, color in ((100, CHEBFUN_BLUE), (200, ORANGE)):
        pv = np.polynomial.chebyshev.chebval(xs, errs[m][0])
        ax.plot(xs, pv, color=color, linewidth=0.7)
    ax.set_ylim(-0.2, 1.4)
    save(fig, "FiltersCF_03.png")

    fig, ax = plt.subplots()
    pv = np.polynomial.chebyshev.chebval(xs, errs[200][0])
    ax.plot(xs, f2v - pv, linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "FiltersCF_04.png")

    fig, ax = plt.subplots()
    for m, color in ((100, CHEBFUN_BLUE), (200, ORANGE)):
        pv = np.polynomial.chebyshev.chebval(xs, errs[m][0])
        ax.semilogy(xs, np.maximum(np.abs(f2v - pv), 1e-18),
                    color=color, linewidth=0.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "FiltersCF_05.png")


def edgedetection():
    """approx/EdgeDetection — eigenvalues and splitting detection."""
    rng = np.random.default_rng(1)

    d = np.sort(rng.standard_normal(20)) + 1j * rng.standard_normal(20)
    A = np.diag(d)
    A[:10, :10] += np.diag(np.ones(9), 1)
    fig, ax = plt.subplots()
    ev = np.linalg.eigvals(A)
    ax.plot(np.real(ev), np.imag(ev), "xb", markersize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "EdgeDetection_01.png")

    # |exp(x) sin(10 pi x)|: piecewise construction at the known roots
    breaks = sorted({-1.0, 1.0, *[k / 10 for k in range(-10, 11)]})
    f = cj.chebfun(lambda x: jnp.abs(jnp.exp(x) * jnp.sin(10 * PI * x)),
                   domain=breaks)
    xs = jnp.linspace(-1, 1, 4000)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(f(xs)), color=CHEBFUN_BLUE,
            linewidth=0.8)
    save(fig, "EdgeDetection_02.png")

    # spectral abscissa of (1-t)B + tC — a piecewise-smooth function
    rng0 = np.random.default_rng(0)
    B = rng0.standard_normal((20, 20))
    C = rng0.standard_normal((20, 20))

    def abscissa(t):
        return np.max(np.real(np.linalg.eigvals((1 - t) * B + t * C)))

    tt = np.linspace(0, 1, 600)
    av = np.array([abscissa(t) for t in tt])
    fig, ax = plt.subplots()
    ax.plot(tt, av, color=CHEBFUN_BLUE, linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("spectral abscissa of (1-t)B + tC", fontsize=10)
    save(fig, "EdgeDetection_03.png")

    # derivative (finite differences) reveals the kinks
    fig, ax = plt.subplots()
    ax.plot(tt[1:], np.diff(av) / np.diff(tt), color=ORANGE,
            linewidth=0.9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("derivative: jumps at the crossing points",
                 fontsize=10)
    save(fig, "EdgeDetection_04.png")

    # zoom near one kink
    kink = tt[np.argmax(np.abs(np.diff(av, 2)))]
    m = (tt > kink - 0.05) & (tt < kink + 0.05)
    fig, ax = plt.subplots()
    ax.plot(tt[m], av[m], ".-", color=CHEBFUN_BLUE, markersize=4,
            linewidth=0.8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("zoom at a nonsmooth point", fontsize=10)
    save(fig, "EdgeDetection_05.png")


def bsplineconv():
    """approx/BSplineConv — B-splines by iterated convolution."""
    ax_lim = (-3, 3, -0.2, 1.2)

    def box(x):
        return ((np.asarray(x) >= -0.5)
                & (np.asarray(x) < 0.5)).astype(float)

    xs = np.linspace(-3, 3, 4000)

    # B0
    fig, ax = plt.subplots()
    ax.plot(xs, box(xs), color=CHEBFUN_BLUE, linewidth=1.6)
    ax.set_xlim(ax_lim[0], ax_lim[1])
    ax.set_ylim(ax_lim[2], ax_lim[3])
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "BSplineConv_01.png")

    # iterated convolutions via numerical quadrature
    Bk = box(xs)
    dx = xs[1] - xs[0]
    for k in range(1, 5):
        Bk = np.convolve(Bk, box(xs), mode="same") * dx
        pts = np.arange(-(k + 1) / 2, (k + 1) / 2 + 1e-9, 1.0)
        fig, ax = plt.subplots()
        ax.plot(xs, Bk, color=CHEBFUN_BLUE, linewidth=1.6)
        # knots
        knot_vals = np.interp(pts, xs, Bk)
        ax.plot(pts, knot_vals, ".r", markersize=9)
        ax.set_xlim(ax_lim[0], ax_lim[1])
        ax.set_ylim(ax_lim[2], ax_lim[3])
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_title(f"B{k}", fontsize=10)
        save(fig, f"BSplineConv_{k+1:02d}.png")


PAGES = {
    "Pushnitski": pushnitski,
    "Inpainting1D": inpainting1d,
    "FiltersCF": filterscf,
    "EdgeDetection": edgedetection,
    "BSplineConv": bsplineconv,
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
