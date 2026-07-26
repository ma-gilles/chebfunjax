"""Generate per-block figures for the docs/examples/approx pages.

First tranche: GreedyInterp, BestL1, OddEven, AAAApprox.
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
GREEN = (0.0, 0.7, 0.0)
PI = float(np.pi)


def save(fig, name):
    from PIL import Image

    ref_path = os.path.join(REF, name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(OUT, name), size=size)
    plt.close(fig)
    print(f"  {name} saved")


def _minimax_eval(res, x):
    """Evaluate a MinimaxResult's polynomial at points x."""
    a, b = res.domain
    xh = 2 * (np.asarray(x, dtype=float) - a) / (b - a) - 1
    return np.polynomial.chebyshev.chebval(xh, np.asarray(res.coeffs))


def greedyinterp():
    """approx/GreedyInterp — greedy interpolation nodes for |x| + wiggle."""
    from chebfunjax.chebfun1d.chebfun import Chebfun

    f = cj.chebfun(lambda x: jnp.abs(x)).abs() \
        if False else cj.chebfun(lambda x: x).abs()
    f = f + cj.chebfun(lambda x: jnp.sin(4 * x) / 4)
    xs = jnp.linspace(-1, 1, 2000)
    fv = np.asarray(f(xs))

    # greedy loop: add the max-error location each step
    s = [float(xs[np.argmax(np.abs(fv))])]
    errs = []
    snapshots = {}
    for n in range(1, 129):
        sx = np.array(sorted(set(s)))
        sy = np.asarray(f(jnp.asarray(sx)))
        p = Chebfun.interp1(jnp.asarray(sx), jnp.asarray(sy),
                            domain=(-1.0, 1.0))
        err = fv - np.asarray(p(xs))
        k = int(np.argmax(np.abs(err)))
        errs.append(float(np.abs(err[k])))
        if n in (8, 20, 50):
            snapshots[n] = (sx.copy(), np.asarray(p(xs)).copy())
        s.append(float(xs[k]))

    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), fv, color=CHEBFUN_BLUE, linewidth=1.2)
    save(fig, "GreedyInterp_01.png")

    k = 2
    for n, (sx, pv) in snapshots.items():
        fig, ax = plt.subplots()
        ax.plot(np.asarray(xs), fv, color=CHEBFUN_BLUE, linewidth=1.0)
        ax.plot(np.asarray(xs), pv, color=ORANGE, linewidth=1.0)
        ax.plot(sx, np.asarray(f(jnp.asarray(sx))), ".k", markersize=7)
        ax.set_title(f"n = {n}", fontsize=9)
        save(fig, f"GreedyInterp_{k:02d}.png")
        k += 1

        fig, ax = plt.subplots()
        ax.plot(np.asarray(xs), fv - pv, color=GREEN, linewidth=1.0)
        ax.set_title(f"error, n = {n}", fontsize=9)
        save(fig, f"GreedyInterp_{k:02d}.png")
        k += 1

    # node distribution vs chebpts
    from chebfunjax.utils.quadrature import chebpts

    sx = np.array(sorted(s[:64]))
    fig, ax = plt.subplots()
    ax.plot(np.sort(sx), ".k", markersize=7)
    scheb = np.asarray(chebpts(len(sx), kind=2))
    ax.plot(np.arange(len(scheb)), np.sort(scheb), "or", markersize=4,
            markerfacecolor="none")
    save(fig, f"GreedyInterp_{k:02d}.png")
    k += 1

    # convergence
    fig, ax = plt.subplots()
    ax.semilogy(np.arange(1, len(errs) + 1), errs, ".-",
                color=CHEBFUN_BLUE, markersize=4, linewidth=0.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("number of greedy nodes")
    ax.set_ylabel("max error")
    save(fig, f"GreedyInterp_{k:02d}.png")
    k += 1

    if k == 11:
        fig, ax = plt.subplots()
        ax.plot(np.asarray(xs), fv, color=CHEBFUN_BLUE, linewidth=1.0)
        sx = np.array(sorted(s))
        ax.plot(sx, np.asarray(f(jnp.asarray(sx))), ".k", markersize=5)
        ax.set_title(f"{len(sx)} greedy nodes", fontsize=9)
        save(fig, "GreedyInterp_11.png")

    # Lebesgue functions: greedy nodes (black) vs Chebyshev points (red)
    def lebesgue(nodes, xx):
        nodes = np.asarray(nodes, dtype=float)
        # barycentric weights
        wj = np.ones(len(nodes))
        for j in range(len(nodes)):
            wj[j] = 1.0 / np.prod(nodes[j] - np.delete(nodes, j))
        L = np.zeros_like(xx)
        for i, x in enumerate(xx):
            d = x - nodes
            hit = np.argmin(np.abs(d))
            if abs(d[hit]) < 1e-14:
                L[i] = 1.0
                continue
            terms = wj / d
            L[i] = np.sum(np.abs(terms)) / np.abs(np.sum(terms))
        return L

    sx = np.array(sorted(set(s)))
    xf = np.linspace(-1, 1, 6000)
    Lg = lebesgue(sx, xf)
    Lc = lebesgue(np.asarray(chebpts(len(sx), kind=2)), xf)
    fig, ax = plt.subplots()
    ax.semilogy(xf, Lg, "k", linewidth=0.6)
    ax.semilogy(xf, Lc, "r", linewidth=0.6)
    ax.set_ylim(1, 100)
    ax.set_xlim(-1, 1)
    save(fig, "GreedyInterp_12.png")


def bestl1():
    """approx/BestL1 — Linf vs L2 vs L1 approximation errors."""

    dom = (0.0, 14.0)
    deg = 40  # published example uses 100; 40 shows the same structure
    f = cj.chebfun(lambda x: jnp.sin(x) ** 2 + jnp.sin(x**2), domain=list(dom))
    xs = jnp.linspace(dom[0], dom[1], 3000)
    fv = np.asarray(f(xs))

    # grid-minimax via LP (the deg-40 Remez on [0,14] is slow here)
    from scipy.optimize import linprog

    xg0 = np.linspace(dom[0], dom[1], 1500)
    fg0 = np.asarray(f(jnp.asarray(xg0)))
    V0 = np.polynomial.chebyshev.chebvander(
        2 * (xg0 - dom[0]) / (dom[1] - dom[0]) - 1, deg)
    nc0 = V0.shape[1]
    A0 = np.block([[V0, -np.ones((len(fg0), 1))],
                   [-V0, -np.ones((len(fg0), 1))]])
    b0 = np.concatenate([fg0, -fg0])
    cv0 = np.zeros(nc0 + 1)
    cv0[-1] = 1.0
    lp0 = linprog(cv0, A_ub=A0, b_ub=b0,
                  bounds=[(None, None)] * nc0 + [(0, None)],
                  method="highs")
    c_inf = lp0.x[:nc0]

    def eval_p(_unused, x):
        xh = 2 * (np.asarray(x) - dom[0]) / (dom[1] - dom[0]) - 1
        return np.polynomial.chebyshev.chebval(xh, c_inf)

    pinf = None

    # The published example opens with f overlaid on its L-infinity
    # approximant, then the error; the port previously skipped straight to
    # the error, shifting every subsequent figure by one against the
    # reference numbering (and duplicating the L2 error later to compensate).
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), fv, color=CHEBFUN_BLUE, linewidth=0.9)
    ax.plot(np.asarray(xs), eval_p(pinf, np.asarray(xs)), color=ORANGE,
            linewidth=0.9)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("f and Linfty approximant", fontsize=9)
    save(fig, "BestL1_01.png")

    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), fv - eval_p(pinf, np.asarray(xs)), "k",
            linewidth=0.9)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("error of Linfty approximant", fontsize=9)
    save(fig, "BestL1_02.png")

    p2 = f.polyfit(deg)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), fv, color=CHEBFUN_BLUE, linewidth=0.9)
    ax.plot(np.asarray(xs), np.asarray(p2(xs)), color=ORANGE,
            linewidth=0.9)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("f and L2 approximant", fontsize=9)
    save(fig, "BestL1_03.png")

    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), fv - np.asarray(p2(xs)), "k", linewidth=0.9)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("error of L2 approximant", fontsize=9)
    save(fig, "BestL1_04.png")

    # L1 approximation via iteratively reweighted least squares
    xg = np.linspace(dom[0], dom[1], 1200)
    fg = np.asarray(f(jnp.asarray(xg)))
    V = np.polynomial.chebyshev.chebvander(
        2 * (xg - dom[0]) / (dom[1] - dom[0]) - 1, deg)
    w = np.ones_like(xg)
    for _ in range(60):
        W = np.sqrt(w)[:, None]
        c, *_ = np.linalg.lstsq(V * W, fg * W[:, 0], rcond=None)
        r = fg - V @ c
        w = 1.0 / np.maximum(np.abs(r), 1e-8)
    p1v_g = V @ c
    p1v = np.interp(np.asarray(xs), xg, p1v_g)

    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), fv, color=CHEBFUN_BLUE, linewidth=0.9)
    ax.plot(np.asarray(xs), p1v, color=ORANGE, linewidth=0.9)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("f and L1 approximant", fontsize=9)
    save(fig, "BestL1_05.png")

    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), fv - p1v, "k", linewidth=0.9)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("error of L1 approximant", fontsize=9)
    save(fig, "BestL1_06.png")

    # Second example: f = |x - 1/4| on [-1, 1], degree 80
    x1 = np.linspace(-1, 1, 4000)
    fabs = np.abs(x1 - 0.25)
    # deg-80 Remez on a kink is slow; solve the grid-minimax problem
    # exactly instead via linear programming (min t s.t. |f - Vc| <= t)
    from scipy.optimize import linprog

    Vm = np.polynomial.chebyshev.chebvander(x1[::4], 80)
    fm = fabs[::4]
    nc = Vm.shape[1]
    # variables: [c (nc), t (1)]
    A_ub = np.block([[Vm, -np.ones((len(fm), 1))],
                     [-Vm, -np.ones((len(fm), 1))]])
    b_ub = np.concatenate([fm, -fm])
    cvec = np.zeros(nc + 1)
    cvec[-1] = 1.0
    lp = linprog(cvec, A_ub=A_ub, b_ub=b_ub,
                 bounds=[(None, None)] * nc + [(0, None)],
                 method="highs")
    pinf_v = np.polynomial.chebyshev.chebval(x1, lp.x[:nc])
    fig, ax = plt.subplots()
    ax.plot(x1, fabs - pinf_v, "k", linewidth=0.7)
    ax.set_ylim(-1e-2, 1e-2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Linf error", fontsize=9)
    save(fig, "BestL1_07.png")

    # L2 projection onto degree-80 polynomials by Gauss-Legendre
    # quadrature (constructing chebfun(|x-1/4|) hits the unhappy path)
    gl_x, gl_w = np.polynomial.legendre.leggauss(2000)
    fgl = np.abs(gl_x - 0.25)
    cleg = np.array([
        (2 * k + 1) / 2
        * np.sum(gl_w * fgl * np.polynomial.legendre.legval(
            gl_x, np.eye(81)[k]))
        for k in range(81)])
    p2a_v = np.polynomial.legendre.legval(x1, cleg)
    fig, ax = plt.subplots()
    ax.plot(x1, fabs - p2a_v, "k", linewidth=0.7)
    ax.set_ylim(-1e-2, 1e-2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("L2 error", fontsize=9)
    save(fig, "BestL1_08.png")

    # L1 fit of |x - 1/4| (IRLS on [-1, 1])
    V1 = np.polynomial.chebyshev.chebvander(x1, 80)
    w1 = np.ones_like(x1)
    for _ in range(60):
        W1 = np.sqrt(w1)[:, None]
        c1, *_ = np.linalg.lstsq(V1 * W1, fabs * W1[:, 0], rcond=None)
        r1 = fabs - V1 @ c1
        w1 = 1.0 / np.maximum(np.abs(r1), 1e-10)
    p1a = V1 @ c1
    fig, ax = plt.subplots()
    ax.plot(x1, fabs - p1a, "k", linewidth=0.7)
    ax.set_ylim(-1e-2, 1e-2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("L1 error", fontsize=9)
    save(fig, "BestL1_09.png")

    fig, ax = plt.subplots()
    ax.plot(x1, fabs - p1a, "k", linewidth=0.7)
    ax.set_ylim(-1e-4, 1e-4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("closeup", fontsize=9)
    save(fig, "BestL1_10.png")


def oddeven():
    """approx/OddEven — even/odd decomposition and best constants."""
    from chebfunjax.utils.minimax import minimax

    def f_np(x):
        return np.exp(-150 * (np.asarray(x) - 0.5) ** 2)

    xs = np.linspace(-1, 1, 2000)
    fv = f_np(xs)
    res = minimax(f_np, 0)
    p0 = res
    err0 = res.err

    def eval_p(r, x):
        return _minimax_eval(r, x)

    ax_lim = (-1, 1, -0.6, 1.1)
    fig, ax = plt.subplots()
    ax.plot(xs, fv, color=CHEBFUN_BLUE, linewidth=1.0)
    ax.plot(xs, eval_p(p0, xs), color=GREEN, linewidth=1.0)
    ax.set_xlim(ax_lim[0], ax_lim[1])
    ax.set_ylim(ax_lim[2], ax_lim[3])
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "OddEven_01.png")

    fig, ax = plt.subplots()
    ax.plot(xs, eval_p(p0, xs) - fv, color=GREEN, linewidth=1.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title(f"error = {float(err0):.6f}", fontsize=9)
    save(fig, "OddEven_02.png")

    feven = (f_np(xs) + f_np(-xs)) / 2
    fodd = (f_np(xs) - f_np(-xs)) / 2

    def const_best(vals):
        return (vals.max() + vals.min()) / 2, (vals.max()
                                               - vals.min()) / 2

    ce, ee = const_best(feven)
    co, eo = const_best(fodd)

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(xs, feven, color=CHEBFUN_BLUE, linewidth=0.9)
    axes[0, 0].axhline(ce, color=GREEN, linewidth=0.9)
    axes[0, 0].set_title(f"even part, err {ee:.4f}", fontsize=7)
    axes[0, 1].plot(xs, feven - ce, color=GREEN, linewidth=0.9)
    axes[1, 0].plot(xs, fodd, color=CHEBFUN_BLUE, linewidth=0.9)
    axes[1, 0].axhline(co, color=GREEN, linewidth=0.9)
    axes[1, 0].set_title(f"odd part, err {eo:.4f}", fontsize=7)
    axes[1, 1].plot(xs, fodd - co, color=GREEN, linewidth=0.9)
    for a in axes.ravel():
        a.tick_params(labelsize=6)
    save(fig, "OddEven_03.png")

    # degree sweep: best errors for f, feven, fodd
    degs = list(range(0, 22, 2))
    ef, ee_l, eo_l = [], [], []
    for d in degs:
        r = minimax(f_np, d)
        ef.append(float(r.err))
        r = minimax(lambda x: (f_np(x) + f_np(-np.asarray(x))) / 2, d)
        ee_l.append(float(r.err))
        r = minimax(lambda x: (f_np(x) - f_np(-np.asarray(x))) / 2, d)
        eo_l.append(float(r.err))

    fig, ax = plt.subplots()
    ax.semilogy(degs, ef, ".-", color=CHEBFUN_BLUE, markersize=7,
                linewidth=0.8, label="f")
    ax.semilogy(degs, ee_l, ".-", color=ORANGE, markersize=7,
                linewidth=0.8, label="even part")
    ax.semilogy(degs, eo_l, ".-", color=GREEN, markersize=7,
                linewidth=0.8, label="odd part")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "OddEven_04.png")

    # per-degree error curves for a few degrees
    for k, d in enumerate((4, 8, 12, 16, 20), 5):
        r = minimax(f_np, d)
        fig, ax = plt.subplots()
        ax.plot(xs, eval_p(r, xs) - fv, color=GREEN, linewidth=0.9)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_title(f"degree {d} error", fontsize=9)
        save(fig, f"OddEven_{k:02d}.png")


def aaaapprox():
    """approx/AAAApprox — AAA rational approximation gallery."""
    # gamma on [-2, 2] (sampled avoiding the poles at 0, -1, -2)
    from scipy.special import gamma as G

    from chebfunjax.utils.aaa import aaa

    Z = np.linspace(-1.99, 1.99, 4000)
    Z = Z[np.min(np.abs(Z[:, None] - np.array([0.0, -1.0])[None, :]),
                 axis=1) > 1e-3]
    F = G(Z)
    r, pol, res, zer, z_sup, f_sup, w = aaa(F, Z, tol=1e-8, mmax=30)

    xs = np.linspace(-3, 3, 3000)
    xs = xs[np.min(np.abs(xs[:, None] - np.real(pol)[None, :]),
                   axis=1) > 5e-3]
    fig, ax = plt.subplots()
    ax.plot(xs, np.real(r(xs)), color=CHEBFUN_BLUE, linewidth=0.9)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-8, 8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "AAAApprox_01.png")

    print("    gamma poles (real parts):",
          np.round(np.sort(np.real(pol))[:4], 4))

    fig, ax = plt.subplots()
    ax.plot(np.real(pol), np.imag(pol), "xr", markersize=8)
    ax.plot(np.real(zer), np.imag(zer), "ob", markersize=5,
            markerfacecolor="none")
    ax.set_xlim(-6, 4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "AAAApprox_02.png")

    # f = sin(20x)/(1+25x^2) on [-1, 2]: function and AAA error
    xg = np.linspace(-1, 2, 3000)
    fg = np.sin(20 * xg) / (1 + 25 * xg**2)
    r2, pol2, *_ = aaa(fg, xg, tol=1e-10, mmax=40)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(xg, fg, color=CHEBFUN_BLUE, linewidth=0.8)
    ax1.set_ylim(-1, 1)
    ax1.grid(True, alpha=0.4, linewidth=0.4)
    ax1.tick_params(labelsize=7)
    ax2.semilogy(xg, np.maximum(np.abs(fg - np.real(r2(xg))), 1e-18),
                 color=ORANGE, linewidth=0.6)
    ax2.tick_params(labelsize=7)
    save(fig, "AAAApprox_03.png")

    # poles in the complex plane
    fig, ax = plt.subplots()
    ax.plot(np.real(pol2), np.imag(pol2), "xr", markersize=8)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "AAAApprox_04.png")

    # exp(x): AAA type-(3,3) vs best type-(3,3) error curves (block 9).
    # Best rational approx by classical differential correction (iterated
    # LP); library rational minimax is a backlog item (#14).
    from scipy.optimize import linprog

    xg2 = np.linspace(-1, 1, 4000)
    fexp = np.exp(xg2)
    r4, *_ = aaa(fexp, xg2, tol=0, mmax=4)

    xl = np.linspace(-1, 1, 400)
    fl = np.exp(xl)
    Vp = np.polynomial.chebyshev.chebvander(xl, 3)
    q = np.ones_like(xl)
    delta = None
    for _ in range(25):
        # variables: [p (4), qc (4), t]; constraints |f q - p| <= t q_old
        Vq = np.polynomial.chebyshev.chebvander(xl, 3)
        A = np.block([
            [-Vp, fl[:, None] * Vq, -q[:, None]],
            [Vp, -fl[:, None] * Vq, -q[:, None]],
            [np.zeros_like(Vp), -Vq, np.zeros((len(xl), 1))],
        ])
        b = np.concatenate([np.zeros(2 * len(xl)),
                            -0.1 * np.ones(len(xl))])
        cvec = np.zeros(9)
        cvec[-1] = 1.0
        # normalize q(0) = 1
        A_eq = np.zeros((1, 9))
        A_eq[0, 4] = 1.0
        lp = linprog(cvec, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=[1.0],
                     bounds=[(None, None)] * 8 + [(0, None)],
                     method="highs")
        if not lp.success:
            break
        pc, qc = lp.x[:4], lp.x[4:8]
        qnew = np.polynomial.chebyshev.chebval(xl, qc)
        if delta is not None and abs(lp.x[-1] - delta) < 1e-14:
            q = qnew
            break
        delta = lp.x[-1]
        q = qnew
    pbest = np.polynomial.chebyshev.chebval(xg2, pc) \
        / np.polynomial.chebyshev.chebval(xg2, qc)

    fig, ax = plt.subplots()
    ax.plot(xg2, fexp - np.real(r4(xg2)), color=CHEBFUN_BLUE,
            linewidth=0.9)
    ax.plot(xg2, fexp - pbest, color=ORANGE, linewidth=0.9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("AAA and best type (3,3) approximants to exp(x)",
                 fontsize=9)
    save(fig, "AAAApprox_05.png")

    # |x|: AAA error at machine scale (block 10)
    fabs2 = np.abs(xg2)
    ra, *_ = aaa(fabs2, xg2, tol=1e-13, mmax=60)
    fig, ax = plt.subplots()
    ax.plot(xg2, fabs2 - np.real(ra(xg2)), color=CHEBFUN_BLUE,
            linewidth=0.7)
    ax.set_ylim(-5e-14, 5e-14)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("abs(x)-r(x)", fontsize=9)
    save(fig, "AAAApprox_06.png")

    # scattered-data AAA: point cloud + poles (block 11)
    rng = np.random.default_rng(0)
    npts = 2000
    X = 8 * rng.random(npts) - 4
    Y = 2 * rng.random(npts) - 1 + X**3 / 16
    Zc = X + 1j * Y

    def ffz(z):
        return np.sqrt(2 + z**2) / (z - 4)

    rz, polz, *_ = aaa(ffz(Zc), Zc, tol=1e-10, mmax=40)
    fig, ax = plt.subplots()
    ax.plot(np.real(Zc), np.imag(Zc), ".k", markersize=2)
    ax.plot(np.real(polz), np.imag(polz), ".r", markersize=8)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-5, 5)
    save(fig, "AAAApprox_07.png")


PAGES = {
    "GreedyInterp": greedyinterp,
    "BestL1": bestl1,
    "OddEven": oddeven,
    "AAAApprox": aaaapprox,
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
