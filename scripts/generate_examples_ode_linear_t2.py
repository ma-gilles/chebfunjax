"""Generate per-block figures for the ode-linear example category,
tranche 2: fourteen smaller pages.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from chebfunjax.plotting import CHEBFUN_BLUE, chebfun_style, save_chebfun_figure

chebfun_style()

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
REFROOT = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/"
           "refs/docs/images")
ORANGE = "#D95319"
PI = float(np.pi)


def save(fig, name):
    from PIL import Image

    ref_path = os.path.join(REFROOT, "ode-linear", name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(DOCS, "ode-linear", name),
                        size=size)
    plt.close(fig)
    print(f"  ode-linear/{name} saved")


def diffmat(x):
    N = len(x)
    c = np.ones(N)
    c[0] = c[-1] = 2.0
    c *= (-1.0) ** np.arange(N)
    X = x[:, None] - x[None, :]
    D = (c[:, None] / c[None, :]) / (X + np.eye(N))
    return D - np.diag(D.sum(axis=1))


def chebgrid(n, a=-1.0, b=1.0):
    xs = np.cos(PI * np.arange(n) / (n - 1))[::-1]
    return 0.5 * (a + b) + 0.5 * (b - a) * xs


def cheb_ops(n, dom):
    xs = chebgrid(n, *dom)
    D = diffmat(np.cos(PI * np.arange(n) / (n - 1))[::-1]) \
        * (2.0 / (dom[1] - dom[0]))
    return xs, D, D @ D


def regions():
    """ode-linear/Regions — stability regions of ODE formulas."""
    th = np.linspace(0, 2 * PI, 400)
    z = np.exp(1j * th)

    # Adams-Bashforth boundaries: lam h = (z-1)/rho(z) style loci
    fig, ax = plt.subplots()
    ax.axhline(0, color="k", linewidth=0.6)
    ax.axvline(0, color="k", linewidth=0.6)
    for s, style in ((1, "-"), (2, "-"), (3, "-")):
        if s == 1:
            w = z - 1
        elif s == 2:
            w = (z - 1) / ((3 - 1 / z) / 2)
        else:
            w = (z - 1) / ((23 - 16 / z + 5 / z**2) / 12)
        ax.plot(np.real(w), np.imag(w), style, linewidth=1.2)
    ax.set_aspect("equal")
    ax.set_title("Adams-Bashforth stability regions", fontsize=9)
    save(fig, "Regions_01.png")

    # Adams-Moulton
    fig, ax = plt.subplots()
    ax.axhline(0, color="k", linewidth=0.6)
    ax.axvline(0, color="k", linewidth=0.6)
    for s in (3, 4, 5):
        if s == 3:
            w = 12 * (z**2 - z) / (5 * z**2 + 8 * z - 1)
        elif s == 4:
            w = 24 * (z**3 - z**2) / (9 * z**3 + 19 * z**2
                                      - 5 * z + 1)
        else:
            w = 720 * (z**4 - z**3) / (251 * z**4 + 646 * z**3
                                       - 264 * z**2 + 106 * z - 19)
        ax.plot(np.real(w), np.imag(w), linewidth=1.2)
    ax.set_aspect("equal")
    ax.set_title("Adams-Moulton stability regions", fontsize=9)
    save(fig, "Regions_02.png")

    # BDF (backward differentiation): exteriors
    fig, ax = plt.subplots()
    ax.axhline(0, color="k", linewidth=0.6)
    ax.axvline(0, color="k", linewidth=0.6)
    for s in (1, 2, 3):
        w = np.zeros_like(z)
        for j in range(1, s + 1):
            w = w + (1 - 1 / z) ** j / j
        ax.plot(np.real(w), np.imag(w), linewidth=1.2)
    ax.set_aspect("equal")
    ax.set_title("BDF stability boundaries", fontsize=9)
    save(fig, "Regions_03.png")

    # Runge-Kutta: |1 + w + ... + w^s/s!| = 1 level curves
    fig, ax = plt.subplots()
    xg = np.linspace(-5, 2, 400)
    yg = np.linspace(-3.5, 3.5, 400)
    X, Y = np.meshgrid(xg, yg)
    Z = X + 1j * Y
    ax.axhline(0, color="k", linewidth=0.6)
    ax.axvline(0, color="k", linewidth=0.6)
    R = np.ones_like(Z)
    fact = 1.0
    for s in range(1, 5):
        fact *= s
        R = R + Z**s / fact
        ax.contour(X, Y, np.abs(R), levels=[1], linewidths=1.2)
    ax.set_aspect("equal")
    ax.set_title("Runge-Kutta stability regions, s = 1..4",
                 fontsize=9)
    save(fig, "Regions_04.png")


def nonstandardbcs():
    """ode-linear/NonstandardBCs — integral and interior conditions."""
    dom = (-1.0, 1.0)
    n = 400
    xs, D, D2 = cheb_ops(n, dom)
    w = np.zeros(n)  # trapezoid weights on the Chebyshev grid
    w[1:-1] = (xs[2:] - xs[:-2]) / 2
    w[0] = (xs[1] - xs[0]) / 2
    w[-1] = (xs[-1] - xs[-2]) / 2

    def solve_with_rows(rows, vals, rhs=1.0):
        L = 0.01 * D2 + np.diag(xs) @ D + np.eye(n)
        b = np.full(n, rhs)
        for r, (row, val) in enumerate(zip(rows, vals)):
            L[0 if r == 0 else -1] = row
            b[0 if r == 0 else -1] = val
        return np.linalg.solve(L, b)

    # 1: integral condition sum(u) = 0 + u(1) = 0
    u = solve_with_rows([w, np.eye(n)[-1]], [0.0, 0.0])
    fig, ax = plt.subplots()
    ax.plot(xs, u, color=CHEBFUN_BLUE, linewidth=1.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("BCs: sum(u) = 0 and u(1) = 0", fontsize=9)
    save(fig, "NonstandardBCs_01.png")

    # 2: interior point condition u(0) = 1 + u(1) = 0
    mid = np.zeros(n)
    mid[np.argmin(np.abs(xs))] = 1.0
    u = solve_with_rows([mid, np.eye(n)[-1]], [1.0, 0.0])
    fig, ax = plt.subplots()
    ax.plot(xs, u, color=CHEBFUN_BLUE, linewidth=1.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("BCs: u(0) = 1 and u(1) = 0", fontsize=9)
    save(fig, "NonstandardBCs_02.png")

    # 3: u(-1) = u(1) (periodic-like) + u'(−1) = 0
    per = np.eye(n)[0] - np.eye(n)[-1]
    u = solve_with_rows([per, D[0]], [0.0, 0.0])
    fig, ax = plt.subplots()
    ax.plot(xs, u, color=CHEBFUN_BLUE, linewidth=1.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("BCs: u(-1) = u(1) and u'(-1) = 0", fontsize=9)
    save(fig, "NonstandardBCs_03.png")

    # 4: max-type surrogate — normalization u'(0) = 5
    dmid = D[np.argmin(np.abs(xs))]
    u = solve_with_rows([dmid, np.eye(n)[-1]], [5.0, 0.0])
    fig, ax = plt.subplots()
    ax.plot(xs, u, color=CHEBFUN_BLUE, linewidth=1.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("BCs: u'(0) = 5 and u(1) = 0", fontsize=9)
    save(fig, "NonstandardBCs_04.png")


def fouriercollocation():
    """ode-linear/FourierCollocation — periodic ODE solves."""
    dom = (0.0, 2 * PI)
    n = 256
    xg = np.linspace(*dom, n, endpoint=False)
    dx = xg[1] - xg[0]
    # spectral periodic differentiation via FFT matrix (dense here)
    col = np.zeros(n)
    kk = np.arange(1, n)
    col[1:] = 0.5 * (-1.0) ** kk / np.tan(kk * dx / 2)
    Dp = np.zeros((n, n))
    idx = np.arange(n)
    for k in range(n):
        Dp[k] = np.roll(np.concatenate([[0], col[1:]]), k)[
            np.argsort(idx)] if False else 0
    # simpler: circulant from column
    c0 = np.concatenate([[0.0], col[1:]])
    for k in range(n):
        Dp[:, k] = np.roll(c0, k)
    Dp = Dp.T * (-1) if False else -Dp

    def solve_periodic(a_np, f_np):
        L = Dp + np.diag(a_np(xg))
        return np.linalg.solve(L, f_np(xg))

    u = solve_periodic(lambda x: 1 + np.sin(np.cos(10 * x)),
                       lambda x: np.exp(np.sin(x)))
    fig, ax = plt.subplots()
    ax.plot(xg, u, color=CHEBFUN_BLUE, linewidth=1.2)
    ax.set_xlim(*dom)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("u' + (1 + sin(cos 10x)) u = exp(sin x), periodic",
                 fontsize=9)
    save(fig, "FourierCollocation_01.png")

    # convergence with n
    ns = 2 ** np.arange(4, 9)
    xf = np.linspace(*dom, 512, endpoint=False)

    def solve_at(nn):
        xgn = np.linspace(*dom, nn, endpoint=False)
        dxn = xgn[1] - xgn[0]
        coln = np.zeros(nn)
        kkn = np.arange(1, nn)
        coln[1:] = 0.5 * (-1.0) ** kkn / np.tan(kkn * dxn / 2)
        Dn = np.zeros((nn, nn))
        c0n = np.concatenate([[0.0], coln[1:]])
        for k in range(nn):
            Dn[:, k] = np.roll(c0n, k)
        Dn = -Dn
        L = Dn + np.diag(1 + np.sin(np.cos(10 * xgn)))
        un = np.linalg.solve(L, np.exp(np.sin(xgn)))
        return xgn, un

    x_ref, u_ref = solve_at(512)
    errs = []
    for nn in ns:
        xgn, un = solve_at(int(nn))
        errs.append(np.max(np.abs(np.interp(xgn, x_ref, u_ref) - un)))
    fig, ax = plt.subplots()
    ax.semilogy(ns, np.maximum(errs, 1e-17), ".-", markersize=8,
                linewidth=1.0, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("n")
    ax.set_title("spectral convergence of Fourier collocation",
                 fontsize=9)
    save(fig, "FourierCollocation_02.png")

    # second-order periodic problem
    D2p = Dp @ Dp
    L2 = D2p + np.diag(2 + np.cos(xg))
    u2 = np.linalg.solve(L2, np.sin(3 * xg))
    fig, ax = plt.subplots()
    ax.plot(xg, u2, color=ORANGE, linewidth=1.2)
    ax.set_xlim(*dom)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("u'' + (2 + cos x) u = sin 3x, periodic", fontsize=9)
    save(fig, "FourierCollocation_03.png")

    # eigenvalues of the periodic operator
    ev = np.sort(np.linalg.eigvals(L2).real)[:20]
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, 21), ev, ".", markersize=8,
            color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("lowest eigenvalues of the periodic operator",
                 fontsize=9)
    save(fig, "FourierCollocation_04.png")


def wikiode():
    """ode-linear/WikiODE — the Wikipedia examples."""
    # y' + 2y = sin t, y(0) = 1 (closed form)
    ts = np.linspace(0, 10, 800)
    y1 = (np.sin(ts) * 2 - np.cos(ts)) / 5 + 6 / 5 * np.exp(-2 * ts)
    fig, ax = plt.subplots()
    ax.plot(ts, y1, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "WikiODE_01.png")

    # y'' + 4y' + 5y = 0 damped oscillation
    sol = solve_ivp(lambda t, y: [y[1], -4 * y[1] - 5 * y[0]],
                    (0, 5), [1.0, 0.0], t_eval=np.linspace(0, 5, 500))
    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], color=CHEBFUN_BLUE, linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "WikiODE_02.png")

    # x^2 y'' + x y' + (x^2 - 1) y = 0: Bessel J1
    from scipy.special import jv

    xs = np.linspace(0.01, 20, 800)
    fig, ax = plt.subplots()
    ax.plot(xs, jv(1, xs), color=CHEBFUN_BLUE, linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Bessel equation solution J_1", fontsize=9)
    save(fig, "WikiODE_03.png")


def matchedasymp():
    """ode-linear/MatchedAsymp — layers for four epsilons."""
    dom = (0.0, 1.0)

    def layer(ep, n=900):
        xs, D, D2 = cheb_ops(n, dom)
        L = ep * D2 + (1 + xs) @ np.eye(n) * 0 + np.diag(1 + xs) @ D \
            + np.eye(n)
        b = np.zeros(n)
        L[0] = 0.0
        L[0, 0] = 1.0
        b[0] = 0.0
        L[-1] = 0.0
        L[-1, -1] = 1.0
        b[-1] = 1.0
        return xs, np.linalg.solve(L, b)

    fig, axes = plt.subplots(2, 2)
    for j, ax in enumerate(axes.ravel(), 1):
        ep = 10.0 ** (-j)
        xs, y = layer(ep)
        ax.plot(xs, y, color=CHEBFUN_BLUE, linewidth=1.0)
        ax.set_title(f"eps = {ep:g}", fontsize=7)
        ax.tick_params(labelsize=6)
    save(fig, "MatchedAsymp_01.png")

    # inner/outer asymptotics vs the eps = 1e-3 solution
    xs, y = layer(1e-3)
    outer = 2.0 / (1 + xs)
    inner_scale = xs / 1e-3
    fig, ax = plt.subplots()
    ax.plot(xs, y, color=CHEBFUN_BLUE, linewidth=1.4,
            label="solution")
    ax.plot(xs, outer * (1 - np.exp(-inner_scale)), "r--",
            linewidth=1.0, label="matched asymptotic")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "MatchedAsymp_02.png")

    fig, ax = plt.subplots()
    ax.semilogy(xs, np.maximum(np.abs(
        y - outer * (1 - np.exp(-inner_scale))), 1e-18), "k",
        linewidth=0.8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("error of the matched asymptotic", fontsize=9)
    save(fig, "MatchedAsymp_03.png")


def krylov():
    """ode-linear/Krylov — MINRES on a differential operator."""
    from scipy.sparse.linalg import minres

    dom = (-1.0, 1.0)
    n = 500
    xs, D, D2 = cheb_ops(n, dom)
    L = -D2 + 20 * np.eye(n)
    L[0] = 0.0
    L[0, 0] = 1.0
    L[-1] = 0.0
    L[-1, -1] = 1.0
    Lsym = (L + L.T) / 2  # symmetrize for MINRES demo
    f = np.sin(13 * PI * np.abs(xs))
    f[0] = f[-1] = 0.0

    u_direct = np.linalg.solve(L, f)
    resids = []

    def cb(xk):
        resids.append(np.linalg.norm(Lsym @ xk - f))

    u_min, info = minres(Lsym, f, rtol=1e-10, maxiter=400,
                         callback=cb)
    fig, ax = plt.subplots()
    ax.plot(xs, u_direct, color=CHEBFUN_BLUE, linewidth=1.4,
            label="collocation \\\\")
    ax.plot(xs, u_min, "--", color=ORANGE, linewidth=1.0,
            label="MINRES")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Krylov_01.png")

    fig, ax = plt.subplots()
    ax.semilogy(np.maximum(resids, 1e-18), color=CHEBFUN_BLUE,
                linewidth=1.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("iteration")
    ax.set_title("MINRES residual history", fontsize=9)
    save(fig, "Krylov_02.png")

    fig, ax = plt.subplots()
    ax.semilogy(xs, np.maximum(np.abs(u_direct - u_min), 1e-18), "k",
                linewidth=0.7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("pointwise difference", fontsize=9)
    save(fig, "Krylov_03.png")


def contourexpm():
    """ode-linear/ContourExpm — expm via contour quadrature."""
    dom = (-1.0, 1.0)
    n = 200
    xs, D, D2 = cheb_ops(n, dom)
    A = D2[1:-1, 1:-1]
    xi = xs[1:-1]
    u0 = np.exp(-30 * xi**2)
    t = 0.05

    # exact via dense expm
    import scipy.linalg as sla

    u_exact = sla.expm(t * A) @ u0

    # contour quadrature: parabolic Talbot-type contour
    N = 32
    th = (-N / 2 + 0.5 + np.arange(N)) * PI / (N / 2)
    z = (N / t) * (0.1309 - 0.1194 * th**2 + 0.2500j * th)
    dz = (N / t) * (-2 * 0.1194 * th + 0.2500j)
    u_c = np.zeros_like(u0, dtype=complex)
    for k in range(N):
        w = np.linalg.solve(z[k] * np.eye(len(xi)) - A, u0)
        u_c += np.exp(z[k] * t) * w * dz[k]
    u_c = np.real(u_c) * (PI / (N / 2)) / (2 * PI) * 2 \
        if False else np.real(u_c / (2j * PI) * 2j * (PI / (N / 2))) \
        * 1
    u_c = np.real(u_c)

    fig, ax = plt.subplots()
    ax.plot(np.real(z) * t / N, np.imag(z) * t / N, ".-",
            markersize=6, linewidth=0.8, color=CHEBFUN_BLUE)
    ax.set_title("Talbot contour (scaled)", fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "ContourExpm_01.png")

    fig, ax = plt.subplots()
    ax.plot(xi, u0, color=(0.6, 0.6, 0.6), linewidth=1.0,
            label="u(0)")
    ax.plot(xi, u_exact, color=CHEBFUN_BLUE, linewidth=1.4,
            label="exp(tA) u0 (dense)")
    ax.plot(xi, u_c, "--", color=ORANGE, linewidth=1.1,
            label="contour quadrature")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "ContourExpm_02.png")

    fig, ax = plt.subplots()
    ax.semilogy(xi, np.maximum(np.abs(u_exact - u_c), 1e-18), "k",
                linewidth=0.8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("contour-quadrature error", fontsize=9)
    save(fig, "ContourExpm_03.png")


def advdiffjump():
    """ode-linear/AdvDiffJump — advection-diffusion with a jump."""
    # 0.2 u'' - u' with jump conditions at 0 on [-10, 10]
    n1 = n2 = 300
    a, mid, b = -10.0, 0.0, 10.0

    def build_piece(n, lo, hi):
        base = diffmat(np.cos(PI * np.arange(n) / (n - 1))[::-1])
        D = base * (2.0 / (hi - lo))
        return chebgrid(n, lo, hi), D

    x1, D1 = build_piece(n1, a, mid)
    x2, D2m = build_piece(n2, mid, b)
    L1 = 0.2 * D1 @ D1 - D1
    L2 = 0.2 * D2m @ D2m - D2m
    N = n1 + n2
    L = np.zeros((N, N))
    bb = np.zeros(N)
    L[:n1, :n1] = L1
    L[n1:, n1:] = L2
    L[0] = 0.0
    L[0, 0] = 1.0
    L[-1] = 0.0
    L[-1, -1] = 1.0
    # jump conditions: [u] = 2, [u'] = 0
    L[n1 - 1] = 0.0
    L[n1 - 1, n1 - 1] = -1.0
    L[n1 - 1, n1] = 1.0
    bb[n1 - 1] = 2.0
    L[n1] = 0.0
    L[n1, :n1] = -D1[-1]
    L[n1, n1:] = D2m[0]
    u = np.linalg.solve(L, bb + (np.abs(np.concatenate([x1, x2]))
                                 < 3).astype(float) * 0)
    xs = np.concatenate([x1, x2])
    fig, ax = plt.subplots()
    ax.plot(x1, u[:n1], color=CHEBFUN_BLUE, linewidth=1.6)
    ax.plot(x2, u[n1:], color=CHEBFUN_BLUE, linewidth=1.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("homogeneous solve with jump [u] = 2 at x = 0",
                 fontsize=9)
    save(fig, "AdvDiffJump_01.png")

    # with a source
    bb2 = np.exp(-np.concatenate([x1, x2]) ** 2)
    bb2[0] = bb2[-1] = 0.0
    bb2[n1 - 1] = 2.0
    bb2[n1] = 0.0
    u2 = np.linalg.solve(L, bb2)
    fig, ax = plt.subplots()
    ax.plot(x1, u2[:n1], color=ORANGE, linewidth=1.6)
    ax.plot(x2, u2[n1:], color=ORANGE, linewidth=1.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Gaussian source with the same jump", fontsize=9)
    save(fig, "AdvDiffJump_02.png")

    # derivative showing continuity of u'
    du = np.concatenate([D1 @ u2[:n1], D2m @ u2[n1:]])
    fig, ax = plt.subplots()
    ax.plot(x1, du[:n1], "k", linewidth=1.1)
    ax.plot(x2, du[n1:], "k", linewidth=1.1)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("u' is continuous across the interface", fontsize=9)
    save(fig, "AdvDiffJump_03.png")


def spectraldisc():
    """ode-linear/SpectralDisc — rectangular spectral matrices."""
    n = 85
    dom = (-20.0, 10.0)
    xs, D, D2 = cheb_ops(n, dom)
    L = D2 - np.diag(xs)  # Airy operator
    # spy-style plot of the operator with BC rows
    Lb = L.copy()
    Lb[0] = 0.0
    Lb[0, 0] = 1.0
    Lb[-1] = 0.0
    Lb[-1, -1] = 1.0
    fig, ax = plt.subplots()
    ii, jj = np.nonzero(np.abs(Lb) > 1e-8 * np.abs(Lb).max())
    ax.plot(jj, ii, ".", color=CHEBFUN_BLUE, markersize=1.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("spy of the Airy collocation matrix", fontsize=9)
    save(fig, "SpectralDisc_01.png")

    b = np.zeros(n)
    b[0] = 1.0
    u = np.linalg.solve(Lb, b)
    fig, ax = plt.subplots()
    ax.plot(xs, u, color=CHEBFUN_BLUE, linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Airy solution u'' - x u = 0, u(-20)=1, u(10)=0",
                 fontsize=9)
    save(fig, "SpectralDisc_02.png")


def periodicsystem():
    """ode-linear/PeriodicSystem — coupled periodic system."""
    dom = (0.0, 2 * PI)
    n = 200
    xg = np.linspace(*dom, n, endpoint=False)
    dx = xg[1] - xg[0]
    col = np.zeros(n)
    kk = np.arange(1, n)
    col[1:] = 0.5 * (-1.0) ** kk / np.tan(kk * dx / 2)
    Dp = np.zeros((n, n))
    c0 = np.concatenate([[0.0], col[1:]])
    for k in range(n):
        Dp[:, k] = np.roll(c0, k)
    Dp = -Dp

    # u' + v = cos x ; v' - u = sin 2x, periodic
    Z = np.zeros((n, n))
    Iden = np.eye(n)
    L = np.block([[Dp, Iden], [-Iden, Dp]])
    b = np.concatenate([np.cos(xg), np.sin(2 * xg)])
    uv = np.linalg.solve(L, b)
    u, v = uv[:n], uv[n:]
    fig, ax = plt.subplots()
    ax.plot(xg, u, linewidth=2.0, label="u")
    ax.plot(xg, v, linewidth=2.0, label="v")
    ax.legend(fontsize=8)
    ax.set_xlim(*dom)
    ax.set_title("Solutions u and v", fontsize=10)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "PeriodicSystem_01.png")

    fig, ax = plt.subplots()
    ax.plot(u, v, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("phase plot (u, v)", fontsize=10)
    save(fig, "PeriodicSystem_02.png")


def floquet():
    """ode-linear/Floquet — fundamental matrix of a periodic system."""
    # Mathieu-type: u'' + (2 + 2cos t) u = 0; monodromy over [0, 2pi]
    def rhs(t, y):
        return [y[1], -(2 + 2 * np.cos(t)) * y[0]]

    ts = np.linspace(0, 2 * PI, 600)
    P = np.zeros((2, 2, len(ts)))
    for j, ic in enumerate((np.array([1.0, 0.0]),
                            np.array([0.0, 1.0]))):
        sol = solve_ivp(rhs, (0, 2 * PI), ic, t_eval=ts, rtol=1e-10)
        P[:, j] = sol.y

    fig, axes = plt.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            axes[i, j].plot(ts, P[i, j], linewidth=1.6,
                            color=CHEBFUN_BLUE)
            axes[i, j].tick_params(labelsize=6)
    save(fig, "Floquet_01.png")

    M = P[:, :, -1]
    mults = np.linalg.eigvals(M)
    print(f"    Floquet multipliers: {mults}")
    th = np.linspace(0, 2 * PI, 200)
    fig, ax = plt.subplots()
    ax.plot(np.cos(th), np.sin(th), "k", linewidth=0.7)
    ax.plot(np.real(mults), np.imag(mults), "xr", markersize=11,
            markeredgewidth=2)
    ax.set_aspect("equal")
    ax.set_title("Floquet multipliers vs the unit circle",
                 fontsize=9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Floquet_02.png")


def dawsonintegral():
    """ode-linear/DawsonIntegral — f' + 2xf = 1."""
    from scipy.special import dawsn

    W = 5.0
    xs = np.linspace(-W, W, 900)
    fig, ax = plt.subplots()
    ax.plot(xs, dawsn(xs), color=CHEBFUN_BLUE, linewidth=1.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("the Dawson function", fontsize=10)
    save(fig, "DawsonIntegral_01.png")

    # solve the ODE by collocation and compare
    n = 400
    xs_c, D, _ = cheb_ops(n, (-W, W))
    L = D + np.diag(2 * xs_c)
    b = np.ones(n)
    L[0] = 0.0
    L[0, 0] = 1.0
    b[0] = dawsn(-W)
    u = np.linalg.solve(L, b)
    fig, ax = plt.subplots()
    ax.semilogy(xs_c, np.maximum(np.abs(u - dawsn(xs_c)), 1e-18),
                "k", linewidth=0.8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("ODE-solve error against scipy dawsn", fontsize=9)
    save(fig, "DawsonIntegral_02.png")


def resonantvandal():
    """ode-linear/ResonantVandal — resonance d'' + d = 1 - cos t."""
    sol = solve_ivp(lambda t, y: [y[1], -y[0] + (1 - np.cos(t))],
                    (0, 50), [2.0, 0.0],
                    t_eval=np.linspace(0, 50, 2000), rtol=1e-10)
    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], color=CHEBFUN_BLUE, linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("t")
    ax.set_title("resonant growth: d'' + d = 1 - cos t", fontsize=9)
    save(fig, "ResonantVandal_01.png")


def orderstars():
    """ode-linear/OrderStars — |r(z) e^-z| = 1 for Pade r."""
    # type (2,2) Pade approximant of exp(z)
    xg = np.linspace(-6, 6, 700)
    X, Y = np.meshgrid(xg, xg)
    Z = X + 1j * Y
    r = (1 + Z / 2 + Z**2 / 12) / (1 - Z / 2 + Z**2 / 12)
    F = np.abs(r * np.exp(-Z))
    fig, ax = plt.subplots()
    ax.contour(X, Y, F, levels=[1.0], colors="k", linewidths=1.4)
    ax.contourf(X, Y, F, levels=[0, 1.0], colors=[(0.85, 0.9, 1.0)])
    ax.axhline(0, color=(0.6, 0.6, 0.6), linewidth=0.5)
    ax.axvline(0, color=(0.6, 0.6, 0.6), linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_title("order star of the (2,2) Pade approximant",
                 fontsize=9)
    save(fig, "OrderStars_01.png")


PAGES = {
    "Regions": regions,
    "NonstandardBCs": nonstandardbcs,
    "FourierCollocation": fouriercollocation,
    "WikiODE": wikiode,
    "MatchedAsymp": matchedasymp,
    "Krylov": krylov,
    "ContourExpm": contourexpm,
    "AdvDiffJump": advdiffjump,
    "SpectralDisc": spectraldisc,
    "PeriodicSystem": periodicsystem,
    "Floquet": floquet,
    "DawsonIntegral": dawsonintegral,
    "ResonantVandal": resonantvandal,
    "OrderStars": orderstars,
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
