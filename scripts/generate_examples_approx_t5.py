"""Generate per-block figures for docs/examples/approx pages, tranche 5:
Prolate, NoisyNonsmooth, NearestOrthFun, GammaFun, EightShades,
BestL2Approximation.
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


def _chebcoeffs(vals):
    n = len(vals)
    ext = np.concatenate([vals[::-1], vals[1:-1]])
    c = np.real(np.fft.fft(ext)) / (n - 1)
    c = c[:n]
    c[0] /= 2
    c[-1] /= 2
    return c


def prolate():
    """approx/Prolate — the kernel exp(i c x t) and its eigenvalues."""
    from chebfunjax.plotting import PARULA

    def kernel_image(N):
        c = N * PI
        g = np.linspace(-1, 1, 400)
        X, T = np.meshgrid(g, g, indexing="ij")
        return np.real(np.exp(1j * c * X * T)), g

    K10img, g = kernel_image(10)
    fig, ax = plt.subplots()
    ax.pcolormesh(g, g, K10img.T, cmap=PARULA, shading="auto")
    xx = np.arange(-10, 10) / 10
    XX, KK = np.meshgrid(xx, xx)
    ax.plot(XX.ravel(), KK.ravel(), ".w", markersize=3)
    ax.set_aspect("equal")
    ax.set_title("Re(K(x,t))", fontsize=10)
    save(fig, "Prolate_01.png")

    # eigenvalues of the integral operator with kernel exp(i c x t):
    # Nystrom discretization with Gauss-Legendre quadrature
    def op_eigs(N, nq=300):
        c = N * PI
        xq, wq = np.polynomial.legendre.leggauss(nq)
        Kmat = np.exp(1j * c * xq[:, None] * xq[None, :]) * wq[None, :]
        return np.sort(np.abs(np.linalg.eigvals(Kmat)))[::-1]

    fig, ax = plt.subplots()
    lam10 = op_eigs(10)
    ax.semilogy(np.arange(1, 41), lam10[:40], ".", markersize=9,
                color=CHEBFUN_BLUE)
    ax.set_ylim(1e-15, 100)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("eigenvalues, c = 10 pi", fontsize=10)
    save(fig, "Prolate_02.png")

    fig, ax = plt.subplots()
    for N in (10, 20):
        lam = op_eigs(N)
        ax.semilogy(np.arange(1, 81), lam[:80], ".", markersize=8)
    ax.set_ylim(1e-15, 100)
    ax.text(20, 4, "c = 10 pi", ha="center", fontsize=8)
    ax.text(45, 4, "c = 20 pi", ha="center", fontsize=8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Prolate_03.png")

    fig, ax = plt.subplots()
    lam4 = op_eigs(4)
    ax.semilogy(np.arange(1, 41), lam4[:40], ".", markersize=9,
                color=CHEBFUN_BLUE)
    ax.set_ylim(1e-15, 100)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("eigenvalues, c = 4 pi", fontsize=10)
    save(fig, "Prolate_04.png")


def noisynonsmooth():
    """approx/NoisyNonsmooth — splitting plus noise floors."""
    rng = np.random.default_rng(0)

    def ff(x):
        x = np.asarray(x)
        return np.abs(x - 0.1) + 1e-8 * rng.standard_normal(np.shape(x))

    xs = np.linspace(-1, 1, 2500)
    fig, ax = plt.subplots()
    ax.plot(xs, np.abs(xs - 0.1), color=CHEBFUN_BLUE, linewidth=1.2)
    ax.set_title("|x - 0.1| + 1e-8 noise", fontsize=10)
    save(fig, "NoisyNonsmooth_01.png")

    # coefficients of the two pieces with the noise plateau
    fig, ax = plt.subplots()
    for (a, b), color in (((-1.0, 0.1), CHEBFUN_BLUE),
                          ((0.1, 1.0), ORANGE)):
        n = 120
        xc = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(
            PI * np.arange(n) / (n - 1))
        cn = np.abs(_chebcoeffs(ff(xc[::-1])[::-1]))
        ax.semilogy(np.arange(len(cn)), np.maximum(cn, 1e-18), ".-",
                    color=color, markersize=5, linewidth=0.5)
    ax.set_title("Chebyshev coefficients of the two pieces",
                 fontsize=10)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "NoisyNonsmooth_02.png")

    # truncated-at-plateau reconstruction error
    n = 120
    a, b = 0.1, 1.0
    xc = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(
        PI * np.arange(n) / (n - 1))
    cn = _chebcoeffs(ff(xc[::-1])[::-1])
    cut = 30
    ct = cn.copy()
    ct[cut:] = 0
    xr = np.linspace(a, b, 1000)
    vr = np.polynomial.chebyshev.chebval(
        2 * (xr - a) / (b - a) - 1, ct)
    fig, ax = plt.subplots()
    ax.plot(xr, vr - np.abs(xr - 0.1), "k", linewidth=0.8)
    ax.set_title("error of plateau-truncated right piece", fontsize=10)
    save(fig, "NoisyNonsmooth_03.png")

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(len(cn)), np.maximum(np.abs(cn), 1e-18),
                ".", color=CHEBFUN_BLUE, markersize=4)
    ax.axhline(1e-8, color="r", linewidth=0.8, linestyle="--")
    ax.set_title("plateau at the noise level 1e-8", fontsize=10)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "NoisyNonsmooth_04.png")


def nearestorthfun():
    """approx/NearestOrthFun — QR vs optimal orthonormalization."""
    xs = np.linspace(-1, 1, 1600)
    w = np.full(len(xs), xs[1] - xs[0])
    W = np.sqrt(w)[:, None]
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def panels(A, name, xdom=None):
        xd = xs if xdom is None else xdom
        wd = np.full(len(xd), xd[1] - xd[0])
        Wd = np.sqrt(wd)[:, None]
        U, S, Vt = np.linalg.svd(A * Wd, full_matrices=False)
        Q = (U / Wd) @ Vt
        Q2, _ = np.linalg.qr(A * Wd)
        Q2 = Q2 / Wd
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for j in range(A.shape[1]):
            ax1.plot(xd, Q2[:, j], color=cyc[j % len(cyc)],
                     linewidth=0.8)
            ax2.plot(xd, Q[:, j], color=cyc[j % len(cyc)],
                     linewidth=0.8)
        for a, t in ((ax1, "QR orthonormalization"),
                     (ax2, "Optimal orthonormalization")):
            v = np.max(np.abs(a.get_ylim()))
            a.set_ylim(-v, v)
            a.grid(True, alpha=0.4, linewidth=0.4)
            a.set_title(t, fontsize=8)
            a.tick_params(labelsize=6)
        save(fig, name)

    # 1: monomial Vandermonde [1, x, ..., x^5]
    A1 = np.column_stack([xs**k for k in range(6)])
    panels(A1, "NearestOrthFun_01.png")

    # 2: Chebyshev Vandermonde restricted to [0, 1]
    x01 = np.linspace(0, 1, 1600)
    A2 = np.column_stack([np.cos(k * np.arccos(
        np.clip(x01, -1, 1))) for k in range(6)])
    panels(A2, "NearestOrthFun_02.png", xdom=x01)

    # 3: [1, cos x, sin x^2, x^3, x^4, x^5]
    A3 = np.column_stack([np.ones_like(xs), np.cos(xs),
                          np.sin(xs**2), xs**3, xs**4, xs**5])
    panels(A3, "NearestOrthFun_03.png")

    # 4: oscillatory gallery trio (stegosaurus/wiggly/blasius stand-ins
    # built from the same recipes on [0, 10])
    x10 = np.linspace(0, 10, 2000)
    A4 = np.column_stack([
        np.cos(x10) + 0.6 * np.sin(x10 / 2),
        np.exp(x10 / 10 - 1) * np.sin(10 * x10) * 0.8,
        x10 / 15 + 0.05 * np.sin(x10**2 / 3),
    ])
    panels(A4, "NearestOrthFun_04.png", xdom=x10)


def gammafun():
    """approx/GammaFun — the gamma function with poles."""
    from scipy.special import gamma as G

    def clipped(x):
        v = G(np.asarray(x))
        v[np.abs(v) > 20] = np.nan
        return v

    xs = np.linspace(-4, 4, 6000)
    fig, ax = plt.subplots()
    ax.plot(xs, clipped(xs), color=CHEBFUN_BLUE, linewidth=1.0)
    ax.set_ylim(-10, 10)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Gamma function", fontsize=10)
    save(fig, "GammaFun_01.png")

    # 1/gamma is entire
    fig, ax = plt.subplots()
    ax.plot(xs, 1.0 / G(xs), color=CHEBFUN_BLUE, linewidth=1.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("1/Gamma is entire", fontsize=10)
    save(fig, "GammaFun_02.png")

    # gamma * prod(x+k) removes the poles on [-4, 4]
    reg = G(xs)
    for k in range(0, 4):
        reg = reg * (xs + k)
    fig, ax = plt.subplots()
    ax.plot(xs, reg, color=CHEBFUN_BLUE, linewidth=1.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("gamma(x) x(x+1)(x+2)(x+3): poles removed",
                 fontsize=10)
    save(fig, "GammaFun_03.png")

    fig, ax = plt.subplots()
    ax.semilogy(xs[xs > 0], G(xs[xs > 0]), color=CHEBFUN_BLUE,
                linewidth=1.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("gamma on (0, 4], log scale", fontsize=10)
    save(fig, "GammaFun_04.png")


def eightshades():
    """approx/EightShades — 8 flavors of degree-8 approximation."""
    def f_np(x):
        return np.exp(-50 * (np.asarray(x) - 0.1) ** 2)

    m = 8
    xs = np.linspace(-1, 1, 2000)
    fv = f_np(xs)
    yl = (-0.3, 1.1)

    def cheb_interp(n):
        xc = np.cos(PI * np.arange(n) / (n - 1))
        c = _chebcoeffs(f_np(xc[::-1])[::-1])
        return np.polynomial.chebyshev.chebval(xs, c)

    def cheb_trunc(n):
        N = 400
        xc = np.cos(PI * np.arange(N) / (N - 1))
        c = _chebcoeffs(f_np(xc[::-1])[::-1])[:n]
        return np.polynomial.chebyshev.chebval(xs, c)

    def trig_interp(n):
        xg = np.linspace(-1, 1, n, endpoint=False)
        ck = np.fft.fft(f_np(xg)) / n
        k = np.fft.fftfreq(n, d=2.0 / n) * 2 * PI / 2
        out = np.zeros_like(xs, dtype=complex)
        for j in range(n):
            out += ck[j] * np.exp(1j * PI * np.fft.fftfreq(n)[j] * n
                                  * xs)
        return np.real(out)

    def trig_trunc(n):
        N = 512
        xg = np.linspace(-1, 1, N, endpoint=False)
        ck = np.fft.fft(f_np(xg)) / N
        keep = np.zeros(N, dtype=bool)
        keep[:n // 2 + 1] = True
        keep[-(n // 2):] = True
        ckt = np.where(keep, ck, 0)
        out = np.zeros_like(xs, dtype=complex)
        for j in np.nonzero(keep)[0]:
            kk = j if j <= N // 2 else j - N
            out += ckt[j] * np.exp(1j * PI * kk * xs)
        return np.real(out)

    # figure 1: 2x2 Chebyshev panels (interp / trunc / their errors)
    ci = cheb_interp(m + 1)
    ct = cheb_trunc(m + 1)
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(xs, fv, "k", linewidth=0.8)
    axes[0, 0].plot(xs, ci, color=CHEBFUN_BLUE, linewidth=0.8)
    axes[0, 0].set_title("interpolation", fontsize=7)
    axes[0, 1].plot(xs, fv, "k", linewidth=0.8)
    axes[0, 1].plot(xs, ct, color=CHEBFUN_BLUE, linewidth=0.8)
    axes[0, 1].set_title("truncation", fontsize=7)
    axes[1, 0].plot(xs, fv - ci, color=ORANGE, linewidth=0.8)
    axes[1, 1].plot(xs, fv - ct, color=ORANGE, linewidth=0.8)
    for a in axes[0]:
        a.set_ylim(*yl)
    for a in axes.ravel():
        a.tick_params(labelsize=6)
    save(fig, "EightShades_01.png")

    ti = trig_interp(m + 1)
    tt = trig_trunc(m)
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(xs, fv, "k", linewidth=0.8)
    axes[0, 0].plot(xs, ti, "b", linewidth=0.8)
    axes[0, 0].set_title("trig interpolation", fontsize=7)
    axes[0, 1].plot(xs, fv, "k", linewidth=0.8)
    axes[0, 1].plot(xs, tt, "b", linewidth=0.8)
    axes[0, 1].set_title("trig truncation", fontsize=7)
    axes[1, 0].plot(xs, fv - ti, color=ORANGE, linewidth=0.8)
    axes[1, 1].plot(xs, fv - tt, color=ORANGE, linewidth=0.8)
    for a in axes[0]:
        a.set_ylim(*yl)
    for a in axes.ravel():
        a.tick_params(labelsize=6)
    save(fig, "EightShades_02.png")

    # best (minimax) and least-squares panels
    from scipy.optimize import linprog

    xg = np.linspace(-1, 1, 1200)
    fg = f_np(xg)
    V = np.polynomial.chebyshev.chebvander(xg, m)
    A = np.block([[V, -np.ones((len(fg), 1))],
                  [-V, -np.ones((len(fg), 1))]])
    b = np.concatenate([fg, -fg])
    cv = np.zeros(m + 2)
    cv[-1] = 1.0
    lp = linprog(cv, A_ub=A, b_ub=b,
                 bounds=[(None, None)] * (m + 1) + [(0, None)],
                 method="highs")
    pbest = np.polynomial.chebyshev.chebval(xs, lp.x[:m + 1])
    cls, *_ = np.linalg.lstsq(V, fg, rcond=None)
    pls = np.polynomial.chebyshev.chebval(xs, cls)

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(xs, fv, "k", linewidth=0.8)
    axes[0, 0].plot(xs, pbest, "g", linewidth=0.8)
    axes[0, 0].set_title("best (minimax)", fontsize=7)
    axes[0, 1].plot(xs, fv, "k", linewidth=0.8)
    axes[0, 1].plot(xs, pls, "g", linewidth=0.8)
    axes[0, 1].set_title("least squares", fontsize=7)
    axes[1, 0].plot(xs, fv - pbest, color=ORANGE, linewidth=0.8)
    axes[1, 1].plot(xs, fv - pls, color=ORANGE, linewidth=0.8)
    for a in axes[0]:
        a.set_ylim(*yl)
    for a in axes.ravel():
        a.tick_params(labelsize=6)
    save(fig, "EightShades_03.png")

    # all eight error curves overlaid
    fig, ax = plt.subplots()
    for vals, color, lbl in ((ci, CHEBFUN_BLUE, "cheb interp"),
                             (ct, ORANGE, "cheb trunc"),
                             (ti, "b", "trig interp"),
                             (tt, "m", "trig trunc"),
                             (pbest, "g", "minimax"),
                             (pls, "c", "least sq")):
        ax.semilogy(xs, np.maximum(np.abs(fv - vals), 1e-18),
                    color=color, linewidth=0.6, label=lbl)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "EightShades_04.png")


def bestl2approximation():
    """approx/BestL2Approximation — Legendre projections."""
    from numpy.polynomial import legendre as npleg

    xs = np.linspace(-1, 1, 3000)

    def leg_proj(f_vals_fn, n):
        gl_x, gl_w = np.polynomial.legendre.leggauss(600)
        fg = f_vals_fn(gl_x)
        c = np.array([
            (2 * k + 1) / 2 * np.sum(gl_w * fg * npleg.legval(
                gl_x, np.eye(n + 1)[k])) for k in range(n + 1)])
        return npleg.legval(xs, c)

    fabs = np.abs(xs)
    p5 = leg_proj(np.abs, 5)
    fig, ax = plt.subplots()
    ax.plot(xs, fabs, "k", linewidth=1.1)
    ax.plot(xs, p5, color=CHEBFUN_BLUE, linewidth=1.1)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("|x| and its best L2 approximation, n = 5",
                 fontsize=10)
    save(fig, "BestL2Approximation_01.png")

    def runge(x):
        return 1.0 / (1 + 25 * np.asarray(x) ** 2)

    p10 = leg_proj(runge, 10)
    fig, ax = plt.subplots()
    ax.plot(xs, runge(xs), "k", linewidth=1.1)
    ax.plot(xs, p10, color=CHEBFUN_BLUE, linewidth=1.1)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Runge function, best L2, n = 10", fontsize=10)
    save(fig, "BestL2Approximation_02.png")

    fig, ax = plt.subplots()
    ax.plot(xs, runge(xs) - p10, color=ORANGE, linewidth=0.9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("error of the L2 projection", fontsize=10)
    save(fig, "BestL2Approximation_03.png")

    # L2 vs Linf errors as n grows for |x|
    from scipy.optimize import linprog

    ns = np.arange(2, 30, 2)
    e2, einf = [], []
    xg = np.linspace(-1, 1, 1200)
    fg = np.abs(xg)
    for n in ns:
        pn = leg_proj(np.abs, int(n))
        e2.append(np.max(np.abs(fabs - pn)))
        V = np.polynomial.chebyshev.chebvander(xg, int(n))
        A = np.block([[V, -np.ones((len(fg), 1))],
                      [-V, -np.ones((len(fg), 1))]])
        b = np.concatenate([fg, -fg])
        cv = np.zeros(int(n) + 2)
        cv[-1] = 1.0
        lp = linprog(cv, A_ub=A, b_ub=b,
                     bounds=[(None, None)] * (int(n) + 1)
                     + [(0, None)], method="highs")
        einf.append(lp.x[-1])
    fig, ax = plt.subplots()
    ax.loglog(ns, e2, ".-", color=CHEBFUN_BLUE, markersize=7,
              linewidth=0.9, label="L2 projection (max err)")
    ax.loglog(ns, einf, ".-r", markersize=7, linewidth=0.9,
              label="minimax")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    save(fig, "BestL2Approximation_04.png")


PAGES = {
    "Prolate": prolate,
    "NoisyNonsmooth": noisynonsmooth,
    "NearestOrthFun": nearestorthfun,
    "GammaFun": gammafun,
    "EightShades": eightshades,
    "BestL2Approximation": bestl2approximation,
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
