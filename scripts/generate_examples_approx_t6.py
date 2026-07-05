"""Generate per-block figures for docs/examples/approx pages, tranche 6:
Splines, SmoothCompact, RationalInterp, Local, LebesgueConst, CF30,
AAASpline.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np

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


def splines():
    """approx/Splines — cubic spline through samples of a chebfun."""
    from scipy.interpolate import CubicSpline

    def f_np(x):
        return np.sin(np.asarray(x) + 0.25 * np.asarray(x) ** 2)

    xs = np.linspace(0, 10, 3000)
    fig, ax = plt.subplots()
    ax.plot(xs, f_np(xs), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.2, 1.2)
    save(fig, "Splines_01.png")

    knots = np.arange(11.0)
    sp = CubicSpline(knots, f_np(knots), bc_type="not-a-knot")
    fig, ax = plt.subplots()
    ax.plot(xs, f_np(xs), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.plot(xs, sp(xs), "r", linewidth=1.0)
    ax.plot(knots, f_np(knots), ".r", markersize=10)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.2, 1.2)
    save(fig, "Splines_02.png")

    fig, ax = plt.subplots()
    ax.plot(xs, f_np(xs) - sp(xs), "k", linewidth=0.9)
    ax.plot(knots, np.zeros_like(knots), ".r", markersize=9)
    ax.set_xlim(0, 10)
    ax.set_title("spline interpolation error", fontsize=10)
    save(fig, "Splines_03.png")


def smoothcompact():
    """approx/SmoothCompact — C-infinity compactly supported bumps."""
    # f = box conv box(2^-3) conv box(2^-4) conv box(2^-5)
    xs = np.linspace(-1.5, 1.5, 6000)
    dx = xs[1] - xs[0]

    def box(width):
        return ((np.abs(xs) <= width / 2).astype(float)) / width

    f = box(1.0)
    for k in range(3, 6):
        f = np.convolve(f, box(2.0 ** -k), mode="same") * dx

    fig, ax = plt.subplots()
    ax.plot(xs, f, color=CHEBFUN_BLUE, linewidth=1.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "SmoothCompact_01.png")

    # translated copies on a wider interval
    fig, ax = plt.subplots()
    ax.plot(xs, f, color=CHEBFUN_BLUE, linewidth=1.2)
    ax.plot(xs + 1.0, f, color=ORANGE, linewidth=1.2)
    ax.set_xlim(-1, 2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "SmoothCompact_02.png")

    # sum of the two = partition-of-unity-like plot
    f2 = np.interp(xs, xs + 1.0, f, left=0, right=0)
    fig, ax = plt.subplots()
    ax.plot(xs, f + f2, "k", linewidth=1.2)
    ax.plot(xs, f, color=CHEBFUN_BLUE, linewidth=0.8)
    ax.plot(xs, f2, color=ORANGE, linewidth=0.8)
    ax.set_xlim(-1, 2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "SmoothCompact_03.png")


def rationalinterp():
    """approx/RationalInterp — ratinterp in floating point."""
    # f = (x^4 - 3) / ((x+2) (x - 1.5e-1i? )): use the classic
    # example f = (x-0.2)*(1+x)/ (1 + ep noise): panels for two eps
    def f_np(x):
        x = np.asarray(x)
        return np.cos(PI * x) / (1 + 25 * x**2)

    xs = np.linspace(-1, 1, 2500)
    rng = np.random.default_rng(0)

    def ratinterp_grid(fv_nodes, nodes, m, n):
        """Linearized rational interpolation (p - f q = 0 on nodes)."""
        Vp = np.polynomial.chebyshev.chebvander(nodes, m)
        Vq = np.polynomial.chebyshev.chebvander(nodes, n)
        A = np.hstack([Vp, -fv_nodes[:, None] * Vq])
        _, _, Vt = np.linalg.svd(A)
        sol = Vt[-1]
        pc, qc = sol[:m + 1], sol[m + 1:]
        return (lambda x: np.polynomial.chebyshev.chebval(x, pc)
                / np.polynomial.chebyshev.chebval(x, qc))

    fig, axes = plt.subplots(2, 1)
    for j, ep in enumerate((1e-1, 1e-3)):
        nodes = np.cos(PI * np.arange(15) / 14)
        fv = f_np(nodes) * (1 + ep * rng.standard_normal(len(nodes)))
        r = ratinterp_grid(fv, nodes, 7, 7)
        axes[j].plot(xs, f_np(xs), "k", linewidth=0.8)
        axes[j].plot(xs, r(xs), color=CHEBFUN_BLUE, linewidth=0.8)
        axes[j].set_ylim(-1.5, 1.5)
        axes[j].set_title(f"noise level {ep:g}", fontsize=8)
        axes[j].tick_params(labelsize=6)
    save(fig, "RationalInterp_01.png")

    # clean type (3,3) interpolation in Chebyshev points
    nodes7 = np.cos(PI * np.arange(7) / 6)
    r33 = ratinterp_grid(f_np(nodes7), nodes7, 3, 3)
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(xs, r33(xs), color=CHEBFUN_BLUE, linewidth=1.2)
    axes[0].plot(nodes7, f_np(nodes7), ".r", markersize=8)
    axes[0].set_title("type (3,3) rational interpolant", fontsize=8)
    axes[1].plot(xs, f_np(xs) - r33(xs), "k", linewidth=0.8)
    axes[1].set_title("error", fontsize=8)
    for a in axes:
        a.tick_params(labelsize=6)
    save(fig, "RationalInterp_02.png")

    # degree sweep of max error for (n, n) interpolation
    ns = np.arange(2, 16)
    errs = []
    for n in ns:
        nodesn = np.cos(PI * np.arange(2 * n + 1) / (2 * n))
        rn = ratinterp_grid(f_np(nodesn), nodesn, int(n), int(n))
        with np.errstate(all="ignore"):
            vals = rn(xs)
        vals[~np.isfinite(vals)] = 0
        errs.append(np.max(np.abs(f_np(xs) - vals)))
    fig, ax = plt.subplots()
    ax.semilogy(ns, errs, ".-", color=CHEBFUN_BLUE, markersize=8,
                linewidth=0.9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("n")
    ax.set_ylabel("max error of (n,n) interpolant")
    save(fig, "RationalInterp_03.png")


def local():
    """approx/Local — local complexity scan of a function."""
    def f_np(x):
        x = np.asarray(x)
        return np.sin(100 * x) * np.exp(-((x - 0.5) ** 2) * 30) \
            + np.tanh(5 * x)

    xs = np.linspace(-1, 1, 4000)

    # top: the function; bottom: local polynomial degree needed on
    # sliding windows of width d
    d = 0.2
    centers = np.linspace(-1 + d / 2, 1 - d / 2, 60)
    degs = []
    for c0 in centers:
        a, b = c0 - d / 2, c0 + d / 2
        n = 96
        xc = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(
            PI * np.arange(n) / (n - 1))
        vals = f_np(xc[::-1])[::-1]
        ext = np.concatenate([vals[::-1], vals[1:-1]])
        cn = np.abs(np.real(np.fft.fft(ext))[:n] / (n - 1))
        tail = np.nonzero(cn > 1e-10 * cn.max())[0]
        degs.append(tail[-1] if len(tail) else 0)

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(xs, f_np(xs), color=CHEBFUN_BLUE, linewidth=0.8)
    axes[0].set_title("f", fontsize=8)
    axes[1].plot(centers, degs, ".-", color=ORANGE, markersize=5,
                 linewidth=0.8)
    axes[1].set_title("local degree needed (window 0.2)", fontsize=8)
    for a in axes:
        a.tick_params(labelsize=6)
        a.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Local_01.png")

    # two more scans with different windows
    for k, d in enumerate((0.1, 0.4), 2):
        centers = np.linspace(-1 + d / 2, 1 - d / 2, 60)
        degs = []
        for c0 in centers:
            a, b = c0 - d / 2, c0 + d / 2
            n = 128
            xc = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(
                PI * np.arange(n) / (n - 1))
            vals = f_np(xc[::-1])[::-1]
            ext = np.concatenate([vals[::-1], vals[1:-1]])
            cn = np.abs(np.real(np.fft.fft(ext))[:n] / (n - 1))
            tail = np.nonzero(cn > 1e-10 * cn.max())[0]
            degs.append(tail[-1] if len(tail) else 0)
        fig, ax = plt.subplots()
        ax.plot(centers, degs, ".-", color=ORANGE, markersize=5,
                linewidth=0.8)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_title(f"local degree, window {d:g}", fontsize=10)
        save(fig, f"Local_{k:02d}.png")


def lebesgueconst():
    """approx/LebesgueConst — Lebesgue functions and constants."""
    def lebesgue_fn(nodes, xx):
        nodes = np.asarray(nodes, dtype=float)
        wj = np.ones(len(nodes))
        for j in range(len(nodes)):
            wj[j] = 1.0 / np.prod(nodes[j] - np.delete(nodes, j))
        L = np.zeros_like(xx)
        for i, x in enumerate(xx):
            dd = x - nodes
            hit = np.argmin(np.abs(dd))
            if abs(dd[hit]) < 1e-14:
                L[i] = 1.0
                continue
            terms = wj / dd
            L[i] = np.sum(np.abs(terms)) / np.abs(np.sum(terms))
        return L

    xx = np.linspace(-1, 1, 4000)

    cheb10 = np.cos(PI * np.arange(10) / 9)
    L10 = lebesgue_fn(cheb10, xx)
    fig, ax = plt.subplots()
    ax.plot(xx, L10, color=CHEBFUN_BLUE, linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title(f"Lebesgue function, 10 Chebyshev points "
                 f"(Lambda = {L10.max():.3f})", fontsize=9)
    save(fig, "LebesgueConst_01.png")

    cheb40 = np.cos(PI * np.arange(40) / 39)
    L40 = lebesgue_fn(cheb40, xx)
    equi40 = np.linspace(-1, 1, 40)
    Le40 = lebesgue_fn(equi40, xx)
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(xx, L40, color=CHEBFUN_BLUE, linewidth=0.8)
    axes[0].set_title(f"40 Chebyshev points   Lambda = "
                      f"{L40.max():.2f}", fontsize=8)
    axes[0].grid(True, alpha=0.4, linewidth=0.4)
    axes[1].semilogy(xx, Le40, color=CHEBFUN_BLUE, linewidth=0.8)
    axes[1].set_title(f"40 equispaced points   Lambda = "
                      f"{Le40.max():.2e}", fontsize=8)
    axes[1].grid(True, alpha=0.4, linewidth=0.4)
    for a in axes:
        a.tick_params(labelsize=6)
    save(fig, "LebesgueConst_02.png")

    equi10 = np.linspace(-1, 1, 10)
    Le = lebesgue_fn(equi10, xx)
    fig, ax = plt.subplots()
    ax.semilogy(xx, Le, "r", linewidth=0.9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title(f"10 equispaced points (Lambda = {Le.max():.1f})",
                 fontsize=9)
    save(fig, "LebesgueConst_03.png")


def cf30():
    """approx/CF30 — Caratheodory-Fejer-style near-best approx."""
    # CF approx via SVD of the Hankel matrix of Chebyshev coefficients
    def f_np(x):
        return np.exp(np.asarray(x))

    N = 60
    xc = np.cos(PI * np.arange(N) / (N - 1))
    vals = f_np(xc[::-1])[::-1]
    ext = np.concatenate([vals[::-1], vals[1:-1]])
    c = np.real(np.fft.fft(ext))[:N] / (N - 1)
    c[0] /= 2
    c[-1] /= 2

    def cf_poly(m):
        """CF approximation of degree m from the coefficient tail."""
        a = c.copy()
        tail = a[m + 1:]
        H = np.array([[tail[i + j] if i + j < len(tail) else 0.0
                       for j in range(len(tail))]
                      for i in range(len(tail))])
        U, S, Vt = np.linalg.svd(H)
        # leading singular value ~ CF error; near-best = truncation +
        # correction. For the figure, use truncation + s1 reference.
        return a[:m + 1], S[0] if len(S) else 0.0

    xs = np.linspace(-1, 1, 3000)
    fv = f_np(xs)

    cm, s1 = cf_poly(3)
    pv = np.polynomial.chebyshev.chebval(xs, cm)
    fig, ax = plt.subplots()
    ax.plot(xs, fv - pv, color=CHEBFUN_BLUE, linewidth=1.6)
    ax.axhline(s1, color="k", linewidth=0.6, linestyle="--")
    ax.axhline(-s1, color="k", linewidth=0.6, linestyle="--")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("truncation error vs CF singular value, m = 3",
                 fontsize=9)
    save(fig, "CF30_01.png")

    # (3,0) LP-minimax comparison
    from scipy.optimize import linprog

    xg = np.linspace(-1, 1, 1200)
    fg = f_np(xg)
    V = np.polynomial.chebyshev.chebvander(xg, 3)
    A = np.block([[V, -np.ones((len(fg), 1))],
                  [-V, -np.ones((len(fg), 1))]])
    b = np.concatenate([fg, -fg])
    cv = np.zeros(5)
    cv[-1] = 1.0
    lp = linprog(cv, A_ub=A, b_ub=b,
                 bounds=[(None, None)] * 4 + [(0, None)],
                 method="highs")
    pbest = np.polynomial.chebyshev.chebval(xs, lp.x[:4])
    fig, ax = plt.subplots()
    ax.plot(xs, fv - pbest, "m", linewidth=1.6)
    ax.set_ylim(-0.02, 0.02)
    ax.axhline(lp.x[-1], color="k", linewidth=0.6, linestyle="--")
    ax.axhline(-lp.x[-1], color="k", linewidth=0.6, linestyle="--")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("best (minimax) error, degree 3", fontsize=9)
    save(fig, "CF30_02.png")

    fig, ax = plt.subplots()
    ax.plot(xs, fv - pv, color=CHEBFUN_BLUE, linewidth=1.1,
            label="truncation")
    ax.plot(xs, fv - pbest, "m", linewidth=1.1, label="best")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "CF30_03.png")


def aaaspline():
    """approx/AAASpline — AAA applied to a spline (poles at knots)."""
    from scipy.interpolate import CubicSpline

    from chebfunjax.utils.aaa import aaa

    def f_np(x):
        return np.sin(np.asarray(x) + 0.25 * np.asarray(x) ** 2)

    knots = np.arange(11.0)
    sp = CubicSpline(knots, f_np(knots), bc_type="not-a-knot")

    X = np.linspace(0, 10, 1000)
    r, poles, *_ = aaa(sp(X), X, mmax=200, tol=1e-10)

    fig, ax = plt.subplots()
    ax.plot(np.real(poles), np.imag(poles), ".r", markersize=6)
    ax.set_xlim(-1, 11)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("poles of the AAA approximant to a spline",
                 fontsize=9)
    save(fig, "AAASpline_01.png")

    fig, ax = plt.subplots()
    ax.plot(np.real(poles), np.imag(poles), ".r", markersize=9)
    ax.set_xlim(3.9, 4.1)
    ax.set_ylim(-0.8, 0.8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("poles near x = 4", fontsize=9)
    save(fig, "AAASpline_02.png")

    fig, ax = plt.subplots()
    ax.plot(X, sp(X), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.plot(knots, sp(knots), ".k", markersize=8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("the spline being approximated", fontsize=10)
    save(fig, "AAASpline_03.png")


PAGES = {
    "Splines": splines,
    "SmoothCompact": smoothcompact,
    "RationalInterp": rationalinterp,
    "Local": local,
    "LebesgueConst": lebesgueconst,
    "CF30": cf30,
    "AAASpline": aaaspline,
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
