"""Generate per-block figures for docs/examples/approx pages, tranche 3:
OscError, MinimaxSqrt, Galleries, FermiDirac.
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
from scipy.optimize import linprog

from chebfunjax.plotting import CHEBFUN_BLUE, chebfun_style, save_chebfun_figure
from chebfunjax.utils.aaa import aaa
from chebfunjax.utils.minimax import minimax

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


def _lp_minimax(f_vals, xg, deg, dom=(-1.0, 1.0)):
    """Best degree-`deg` polynomial on the grid via linear programming.

    Returns (eval_fn, max_error)."""
    xh = 2 * (xg - dom[0]) / (dom[1] - dom[0]) - 1
    V = np.polynomial.chebyshev.chebvander(xh, deg)
    nc = V.shape[1]
    A = np.block([[V, -np.ones((len(f_vals), 1))],
                  [-V, -np.ones((len(f_vals), 1))]])
    b = np.concatenate([f_vals, -f_vals])
    cv = np.zeros(nc + 1)
    cv[-1] = 1.0
    lp = linprog(cv, A_ub=A, b_ub=b,
                 bounds=[(None, None)] * nc + [(0, None)],
                 method="highs")
    c = lp.x[:nc]

    def ev(x):
        xhh = 2 * (np.asarray(x) - dom[0]) / (dom[1] - dom[0]) - 1
        return np.polynomial.chebyshev.chebval(xhh, c)

    return ev, float(lp.x[-1])


def _dc_rational(f_vals, xg, m, dom=(-1.0, 1.0), iters=30):
    """Best type-(m,m) rational on the grid via differential correction.

    Returns (eval_fn, max_error)."""
    xh = 2 * (xg - dom[0]) / (dom[1] - dom[0]) - 1
    V = np.polynomial.chebyshev.chebvander(xh, m)
    nv = V.shape[1]
    q = np.ones_like(xg)
    pc = qc = None
    delta = None
    for _ in range(iters):
        A = np.block([
            [-V, f_vals[:, None] * V, -q[:, None]],
            [V, -f_vals[:, None] * V, -q[:, None]],
            [np.zeros_like(V), -V, np.zeros((len(xg), 1))],
        ])
        b = np.concatenate([np.zeros(2 * len(xg)),
                            -0.05 * np.ones(len(xg))])
        cv = np.zeros(2 * nv + 1)
        cv[-1] = 1.0
        A_eq = np.zeros((1, 2 * nv + 1))
        A_eq[0, nv] = 1.0
        lp = linprog(cv, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=[1.0],
                     bounds=[(None, None)] * (2 * nv) + [(0, None)],
                     method="highs")
        if not lp.success:
            break
        pc, qc = lp.x[:nv], lp.x[nv:2 * nv]
        q = np.polynomial.chebyshev.chebval(xh, qc)
        if delta is not None and abs(lp.x[-1] - delta) < 1e-15:
            break
        delta = lp.x[-1]

    def ev(x):
        xhh = 2 * (np.asarray(x) - dom[0]) / (dom[1] - dom[0]) - 1
        return (np.polynomial.chebyshev.chebval(xhh, pc)
                / np.polynomial.chebyshev.chebval(xhh, qc))

    err = float(np.max(np.abs(f_vals - ev(xg)))) if pc is not None \
        else np.nan
    return ev, err


def oscerror():
    """approx/OscError — oscillatory part of the best-approx error."""
    def f_np(x):
        return np.exp(np.asarray(x)) + 0.5 * np.sin(
            2 * PI * np.asarray(x))

    xs = np.linspace(-1, 1, 3000)
    fv = f_np(xs)

    fig, ax = plt.subplots()
    ax.plot(xs, fv, color=CHEBFUN_BLUE, linewidth=2.0)
    save(fig, "OscError_01.png")

    xg = np.linspace(-1, 1, 1500)
    ev4, err4 = _lp_minimax(f_np(xg), xg, 4)
    fig, ax = plt.subplots()
    ax.plot(xs, fv, color=CHEBFUN_BLUE, linewidth=2.0)
    ax.plot(xs, ev4(xs), "r-.", linewidth=2.0)
    ax.set_title("Function and its best approximation", fontsize=10)
    save(fig, "OscError_02.png")

    fig, ax = plt.subplots()
    ax.plot(xs, fv - ev4(xs), "r", linewidth=2.0)
    ax.axhline(err4, color="k", linewidth=0.7, linestyle="--")
    ax.axhline(-err4, color="k", linewidth=0.7, linestyle="--")
    ax.set_title("Approximation error", fontsize=10)
    save(fig, "OscError_03.png")

    # errors for a sweep of degrees
    ns = np.arange(0, 21, 2)
    errs = [_lp_minimax(f_np(xg), xg, int(n))[1] for n in ns]
    fig, ax = plt.subplots()
    ax.semilogy(ns, errs, ".-", color=CHEBFUN_BLUE, markersize=8,
                linewidth=1.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("degree n")
    ax.set_ylabel("minimax error")
    save(fig, "OscError_04.png")

    # error curves at two higher degrees
    for k, n in enumerate((8, 12), 5):
        evn, errn = _lp_minimax(f_np(xg), xg, n)
        fig, ax = plt.subplots()
        ax.plot(xs, fv - evn(xs), "r", linewidth=1.4)
        ax.axhline(errn, color="k", linewidth=0.6, linestyle="--")
        ax.axhline(-errn, color="k", linewidth=0.6, linestyle="--")
        ax.set_title(f"error, degree {n}", fontsize=10)
        save(fig, f"OscError_{k:02d}.png")


def _num2str(a):
    """Mimic MATLAB num2str for the a-values used on this page."""
    if a == 0:
        return "0"
    if a == 1e-5:
        return "1e-05"
    return f"{a:g}"


def _rat_fit_grid(a, N=4000):
    """Sample grid clustered near the left endpoint (where root-type
    functions are steepest) for the AAA rational fit."""
    lo = a if a > 0 else 1e-12
    g = np.unique(np.concatenate([
        lo * (1.0 / lo) ** np.linspace(0.0, 1.0, N // 2),
        np.linspace(lo, 1.0, N // 2),
    ]))
    if a == 0:
        g = np.concatenate([[0.0], g])
    return g


def _poly_err(fn, n, a):
    """Best degree-``n`` minimax polynomial error of ``fn`` on [a, 1]."""
    try:
        return float(minimax(fn, int(n), domain=(a, 1.0)).err)
    except Exception:  # noqa: BLE001 — fall back to grid LP if Remez fails
        xg = _rat_fit_grid(a, 2000)
        return _lp_minimax(fn(xg), xg, int(n), dom=(a, 1.0))[1]


def _rat_err(fn, n, a):
    """Near-minimax type-(n/2, n/2) rational error of ``fn`` on [a, 1].

    Combines two near-best rational fits and keeps the smaller (better)
    error at each degree:

    * AAA (adaptive Antoulas-Anderson) with ``n/2 + 1`` support points —
      captures the root-exponential convergence at higher degree that
      differential correction on a fixed grid stalls short of for the
      near-singular cases (a -> 0);
    * differential correction on a left-clustered grid — closer to the
      true minimax at very low degree, where AAA's greedy interpolation
      overshoots.
    """
    m = int(n) // 2
    ze = np.linspace(a, 1.0, 20000)
    fze = fn(ze)
    zf = _rat_fit_grid(a)
    r = aaa(fn(zf), zf, mmax=m + 1, tol=1e-14, cleanup=False)[0]
    err = float(np.max(np.abs(fze - np.real(r(ze)))))
    # DC on a small left-clustered grid is closer to minimax at low
    # degree; only worth it there (its LP is O(grid^3) and it stalls at
    # higher degree anyway).
    if m <= 4:
        try:
            zc = _rat_fit_grid(a, 400)
            _, dc = _dc_rational(fn(zc), zc, m, dom=(a, 1.0), iters=25)
            if np.isfinite(dc):
                err = min(err, float(dc))
        except Exception:  # noqa: BLE001
            pass
    return err


def _minimax_sweep(fn, a, ns):
    perrs = np.array([_poly_err(fn, n, a) for n in ns])
    rerrs = np.array([_rat_err(fn, n, a) for n in ns])
    # The best degree-n approximation error is monotone non-increasing in
    # n (a higher type/degree can always reproduce a lower one). AAA is
    # interpolatory and can overshoot this at individual degrees, so take
    # the running minimum to recover the true (monotone) minimax envelope.
    return np.minimum.accumulate(perrs), np.minimum.accumulate(rerrs)


def minimaxsqrt():
    """approx/MinimaxSqrt — polynomial vs rational best approximation.

    Faithful port of the chebfun.org approx/MinimaxSqrt example:
    convergence (max error vs degrees-of-freedom) of best polynomial
    (blue) and best type-(n/2, n/2) rational (red) approximation of
    sqrt(x) on [a, 1] as a -> 0 (figures 1-4), an a=0 vs a=1e-5 overlay
    (figure 5), and the same study for the fifth root of x (figure 6).
    The rational approximant converges root-exponentially while the
    polynomial one stalls, the point of the example.
    """
    sqrt_fn = lambda x: np.sqrt(np.asarray(x, dtype=float))  # noqa: E731

    def draw(ax, fn, a, ns, xpad, pstyle, rstyle, label_dy=1.0):
        """Add one (poly, rational) convergence pair + text labels."""
        perrs, rerrs = _minimax_sweep(fn, a, ns)
        astr = _num2str(a)
        ax.semilogy(ns, perrs, pstyle, markersize=7, linewidth=1.0)
        ax.text(ns[-1] + xpad, perrs[-1] * label_dy, f"poly a={astr}",
                color="b", fontsize=9)
        ax.semilogy(ns, rerrs, rstyle, markersize=7, linewidth=1.0)
        ax.text(ns[-1] + xpad, rerrs[-1], f"rat a={astr}",
                color="r", fontsize=9)
        ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
        ax.set_xlim(0, ns[-1] + (2 if ns[-1] <= 8 else 7))
        ax.set_xlabel("DOF")

    ns_short = np.arange(2, 9, 2)
    ns_long = np.arange(2, 21, 2)

    # Figures 1-2: sqrt on [a,1], a = 0.8, 0.1 (short DOF range)
    for a, name in ((0.8, "MinimaxSqrt_01.png"), (0.1, "MinimaxSqrt_02.png")):
        fig, ax = plt.subplots()
        draw(ax, sqrt_fn, a, ns_short, 0.25, "b*-", "r*-")
        ax.set_title(f"sqrt(x) on [a,1], a = {_num2str(a)}", fontsize=10)
        save(fig, name)

    # Figure 3: a = 0.001 (long DOF range)
    fig, ax = plt.subplots()
    draw(ax, sqrt_fn, 1e-3, ns_long, 0.5, "b*-", "r*-")
    ax.set_title("sqrt(x) on [a,1], a = 0.001", fontsize=10)
    save(fig, "MinimaxSqrt_03.png")

    # Figure 4: a = 1e-5, then figure 5 overlays a = 0 (dashed) on the
    # SAME axes (MATLAB omits `hold off` between them).
    fig, ax = plt.subplots()
    draw(ax, sqrt_fn, 1e-5, ns_long, 0.5, "b*-", "r*-")
    ax.set_title("sqrt(x) on [a,1], a = 1e-05", fontsize=10)
    save(fig, "MinimaxSqrt_04.png")

    draw(ax, sqrt_fn, 0.0, ns_long, 0.5, "bo--", "ro--", label_dy=1.3)
    ax.set_title("sqrt(x) on [a,1], a = 0", fontsize=10)
    save(fig, "MinimaxSqrt_05.png")

    # Figure 6: fifth root of x, a = 1e-5 (solid) and a = 0 (dashed)
    fifth = lambda x: np.asarray(x, dtype=float) ** (1.0 / 5.0)  # noqa: E731
    fig, ax = plt.subplots()
    draw(ax, fifth, 1e-5, ns_long, 0.2, "b*-", "r*-")
    draw(ax, fifth, 0.0, ns_long, 0.2, "bo--", "ro--")
    ax.set_title("fifth root of x on [a,1], a = 0 and 1e-5", fontsize=10)
    save(fig, "MinimaxSqrt_06.png")


def galleries():
    """approx/Galleries — a tour of cheb.gallery."""
    # rose curve (MATLAB gallery('rose'): cos(5t/4) e^{it} on [0, 8pi])
    import matplotlib.colors as mcolors
    from scipy.special import airy as sp_airy

    from chebfunjax.utils.gallery import gallery
    from chebfunjax.utils.scribble import scribble

    ts = np.linspace(0, 8 * PI, 6000)
    zz = np.cos(5 / 4 * ts) * np.exp(1j * ts)
    # even-odd fill (MATLAB's rule) by crossing-number rasterization
    poly = np.column_stack([np.real(zz), np.imag(zz)])
    g = np.linspace(-1.2, 1.2, 900)
    X, Y = np.meshgrid(g, g)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    x0, y0 = poly[:-1, 0], poly[:-1, 1]
    x1, y1 = poly[1:, 0], poly[1:, 1]
    inside = np.zeros(len(pts), dtype=int)
    for i in range(len(x0)):
        cond = (y0[i] > pts[:, 1]) != (y1[i] > pts[:, 1])
        xin = (x1[i] - x0[i]) * (pts[:, 1] - y0[i]) \
            / (y1[i] - y0[i] + 1e-300) + x0[i]
        inside += (cond & (pts[:, 0] < xin)).astype(int)
    mask = (inside % 2 == 1).reshape(X.shape)
    fig = plt.figure()
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(np.where(mask, 1.0, np.nan),
              extent=(-1.2, 1.2, -1.2, 1.2), origin="lower",
              cmap=mcolors.ListedColormap(["red"]),
              interpolation="nearest")
    ax.plot(np.real(zz), np.imag(zz), "k", linewidth=1.6)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "Galleries_01.png")

    f2 = gallery("wiggly")
    xs = jnp.linspace(-1, 1, 1500)

    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(f2(xs)), color=CHEBFUN_BLUE,
            linewidth=1.0)
    save(fig, "Galleries_02.png")

    # airy on [-40, 40] (library gallery lacks 'airy' — task #13)
    xa = np.linspace(-40, 40, 4000)
    fig, ax = plt.subplots()
    ax.plot(xa, sp_airy(xa)[0], color=CHEBFUN_BLUE, linewidth=0.9)
    ax.set_ylim(-0.6, 0.6)
    ax.set_title("Airy function", fontsize=10)
    save(fig, "Galleries_03.png")

    # zigzag: degree-10000 polynomial
    fz = gallery("zigzag")
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(fz(xs)), "m", linewidth=1.0)
    ax.set_ylim(-0.13, 0.09)
    ax.set_title("polynomial of degree 10,000", fontsize=10)
    save(fig, "Galleries_04.png")

    # motto: exp(3i * scribble(...))
    s = scribble("there is no fun like chebfun")
    fig, ax = plt.subplots()
    for piece in s.funs:
        a, b = (float(v) for v in piece.interval)
        ts = jnp.linspace(a, b, 24)
        zz = np.exp(3j * np.asarray(piece(ts)))
        ax.plot(np.real(zz), np.imag(zz), "k", linewidth=1.6)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "Galleries_05.png")

    # seismograph
    fs = gallery("seismograph")
    xs2 = jnp.linspace(-1, 1, 6000)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs2), np.asarray(fs(xs2)), color=CHEBFUN_BLUE,
            linewidth=0.5)
    save(fig, "Galleries_06.png")


def fermidirac():
    """approx/FermiDirac — the Fermi-Dirac function and its approx."""
    L = 20

    def f_np(x):
        return 1.0 / (1.0 + np.exp(np.asarray(x) - L))

    xs = np.linspace(0, 80, 3000)
    fig, ax = plt.subplots()
    ax.plot(xs, f_np(xs), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.set_ylim(-1, 2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("physical domain", fontsize=10)
    save(fig, "FermiDirac_01.png")

    # transplantation x(s) = 40(s+1) to [-1, 1]
    ss = np.linspace(-1, 1, 3000)

    def x_of_s(s):
        return 40 * (np.asarray(s) + 1)

    fig, ax = plt.subplots()
    ax.plot(ss, f_np(x_of_s(ss)), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.set_ylim(-1, 2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("transplantation to [-1,1]", fontsize=10)
    save(fig, "FermiDirac_02.png")

    # minimax approximations at several degrees (LP on the s-domain)
    sg = np.linspace(-1, 1, 1500)
    gv = f_np(x_of_s(sg))
    for k, n in enumerate((10, 20), 3):
        evn, errn = _lp_minimax(gv, sg, n)
        fig, ax = plt.subplots()
        ax.plot(ss, gv[np.searchsorted(sg, ss).clip(0, len(sg) - 1)]
                if False else f_np(x_of_s(ss)) - evn(ss),
                "r", linewidth=0.8)
        ax.axhline(errn, color="k", linewidth=0.5, linestyle="--")
        ax.axhline(-errn, color="k", linewidth=0.5, linestyle="--")
        ax.set_title(f"minimax error, degree {n} "
                     f"(err = {errn:.2e})", fontsize=9)
        save(fig, f"FermiDirac_{k:02d}.png")

    # error vs degree for L = 20 and L = 50
    ns = np.arange(4, 41, 4)
    fig, ax = plt.subplots()
    for L_, style in ((20, ".-"), (50, ".--")):
        gvL = 1.0 / (1.0 + np.exp(x_of_s(sg) - L_))
        errs = [_lp_minimax(gvL, sg, int(n))[1] for n in ns]
        ax.semilogy(ns, errs, style, markersize=7, linewidth=0.9,
                    label=f"L = {L_}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("degree")
    save(fig, "FermiDirac_05.png")

    # rational approximation comparison at L = 20
    evr, rerr = _dc_rational(gv, sg, 8)
    fig, ax = plt.subplots()
    ax.plot(ss, f_np(x_of_s(ss)) - evr(ss), color=CHEBFUN_BLUE,
            linewidth=0.8)
    ax.set_title(f"rational type (8,8) error (err = {rerr:.2e})",
                 fontsize=9)
    save(fig, "FermiDirac_06.png")


PAGES = {
    "OscError": oscerror,
    "MinimaxSqrt": minimaxsqrt,
    "Galleries": galleries,
    "FermiDirac": fermidirac,
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
