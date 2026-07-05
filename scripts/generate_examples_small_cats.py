"""Generate per-block figures for the small example categories:
linalg (CondVandermonde, ResolventNorm), integro (WikiIntegroDiff;
FracCalc figures are byte-shared with temp and copied), quad
(GaussClenCurt, SpikeIntegral, QuadratureConvergence, SymbolicNumeric,
HermiteQuad).
"""

import os
import shutil

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

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
REFROOT = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/"
           "refs/docs/images")
ORANGE = "#D95319"
PI = float(np.pi)


def save(fig, cat, name):
    from PIL import Image

    ref_path = os.path.join(REFROOT, cat, name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(DOCS, cat, name), size=size)
    plt.close(fig)
    print(f"  {cat}/{name} saved")


def condvandermonde():
    """linalg/CondVandermonde — cond of [1, x, ..., x^n]."""
    xs = np.linspace(-1, 1, 2000)
    w = np.full(len(xs), xs[1] - xs[0])
    W = np.sqrt(w)[:, None]
    conds = []
    for n in range(1, 21):
        A = np.column_stack([xs**k for k in range(n + 1)])
        s = np.linalg.svd(A * W, compute_uv=False)
        conds.append(s[0] / s[-1])
    rhoc = 1 + np.sqrt(2)
    fig, ax = plt.subplots()
    ax.semilogy(np.arange(1, 21), conds, ".-", color=CHEBFUN_BLUE,
                markersize=7, linewidth=0.9)
    ax.semilogy(np.arange(1, 21), rhoc ** np.arange(1, 21), ".-",
                color=ORANGE, markersize=7, linewidth=0.9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("cond of the Vandermonde quasimatrix vs (1+sqrt 2)^n",
                 fontsize=9)
    save(fig, "linalg", "CondVandermonde_01.png")


def resolventnorm():
    """linalg/ResolventNorm — ||(iyI - A)^-1|| along the imag axis."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((60, 60)) - 3.0 * np.eye(60)

    def nr(A_, y):
        return 1.0 / np.min(np.linalg.svd(1j * y * np.eye(len(A_)) - A_,
                                          compute_uv=False))

    ys = np.linspace(-25, 25, 800)
    f = np.array([nr(A, y) for y in ys])
    fig, ax = plt.subplots()
    ax.plot(ys, f, color=CHEBFUN_BLUE, linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title(f"maximum = {f.max():.4f}", fontsize=10)
    save(fig, "linalg", "ResolventNorm_01.png")

    B = A - 1.5 * np.eye(60)
    fB = np.array([nr(B, y) for y in ys])
    fig, ax = plt.subplots()
    ax.plot(ys, fB, color=CHEBFUN_BLUE, linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title(f"maximum = {fB.max():.4f}", fontsize=10)
    save(fig, "linalg", "ResolventNorm_02.png")

    C = rng.standard_normal((60, 60))
    C = C - (np.max(np.real(np.linalg.eigvals(C))) + 0.5) * np.eye(60)
    fC = np.array([nr(C, y) for y in ys])
    fig, ax = plt.subplots()
    ax.plot(ys, fC, color=CHEBFUN_BLUE, linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title(f"maximum = {fC.max():.4f}", fontsize=10)
    save(fig, "linalg", "ResolventNorm_03.png")


def wikiintegrodiff():
    """integro/WikiIntegroDiff — u' + 2u + 5 int u = theta(t)."""
    # Solve u'' + 2u' + 5u = 0 (differentiated), u(0)=0, u'(0)=1
    # (the Wikipedia example's closed form is exp(-t) sin(2t)/2)
    ts = np.linspace(0, 5, 1000)
    u = 0.5 * np.exp(-ts) * np.sin(2 * ts)
    fig, ax = plt.subplots()
    ax.plot(ts, u, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Solution of integro-differential equation",
                 fontsize=10)
    save(fig, "integro", "WikiIntegroDiff_01.png")


def copy_fraccalc():
    """integro/FracCalc figures are byte-identical to temp's."""
    for i in range(1, 8):
        src = os.path.join(DOCS, "temp", f"FracCalc_{i:02d}.png")
        dst = os.path.join(DOCS, "integro", f"FracCalc_{i:02d}.png")
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            print(f"  integro/FracCalc_{i:02d}.png copied from temp")


def gaussclencurt():
    """quad/GaussClenCurt — Gauss vs Clenshaw-Curtis convergence."""
    from chebfunjax.utils.quadrature import legpts

    def f_np(x):
        x = np.asarray(x)
        return x * np.sin(2 * np.exp(2 * np.sin(2 * np.exp(2 * x))))

    xs = np.linspace(-1, 1, 4000)
    fig, ax = plt.subplots()
    ax.plot(xs, f_np(xs), color=CHEBFUN_BLUE, linewidth=0.8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "quad", "GaussClenCurt_01.png")

    f = cj.chebfun(lambda x: x * jnp.sin(
        2 * jnp.exp(2 * jnp.sin(2 * jnp.exp(2 * x)))))
    exact = float(f.sum())
    print(f"    exact integral = {exact:.14f}")

    NN = np.arange(10, 501, 10)
    errg, errc = [], []
    for n in NN:
        s, w = legpts(int(n))
        errg.append(abs(float(np.sum(np.asarray(w) * f_np(
            np.asarray(s)))) - exact))
        # Clenshaw-Curtis: chebpts + weights via FFT of 1/(1-k^2)
        k = np.arange(int(n))
        c = np.zeros(int(n))
        c[::2] = 2.0 / (1 - np.arange(0, int(n), 2) ** 2)
        wcc = np.real(np.fft.ifft(np.concatenate(
            [c, c[-2:0:-1]])))[:int(n)]
        wcc[0] /= 2
        wcc = np.concatenate([wcc, [wcc[0]]])[:int(n)]
        xcc = np.cos(PI * np.arange(int(n)) / (int(n) - 1))
        # standard CC weight formula instead (robust):
        theta = PI * np.arange(int(n)) / (int(n) - 1)
        wc = np.zeros(int(n))
        v = np.ones(int(n) - 2)
        N_ = int(n) - 1
        for kk in range(1, N_ // 2 + 1):
            fac = 2.0 if kk < N_ / 2 else 1.0
            v = v - fac * np.cos(2 * kk * theta[1:-1]) / (4 * kk**2 - 1)
        wc[1:-1] = 2 * v / N_
        wc[0] = wc[-1] = 1.0 / (N_**2 - 1 + (N_ % 2))
        errc.append(abs(float(np.sum(wc * f_np(xcc))) - exact))
    fig, ax = plt.subplots()
    ax.semilogy(NN, np.maximum(errg, 1e-18), ".-", markersize=5,
                linewidth=0.7, color=CHEBFUN_BLUE, label="Gauss")
    ax.semilogy(NN, np.maximum(errc, 1e-18), ".-", markersize=5,
                linewidth=0.7, color=ORANGE, label="Clenshaw-Curtis")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "quad", "GaussClenCurt_02.png")

    fig, ax = plt.subplots()
    ax.loglog(NN, np.maximum(errg, 1e-18), ".-", markersize=5,
              linewidth=0.7, color=CHEBFUN_BLUE, label="Gauss")
    ax.loglog(NN, np.maximum(errc, 1e-18), ".-", markersize=5,
              linewidth=0.7, color=ORANGE, label="Clenshaw-Curtis")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    save(fig, "quad", "GaussClenCurt_03.png")


def spikeintegral():
    """quad/SpikeIntegral — integrating a needle-like function."""
    def f_np(x):
        x = np.asarray(x)
        return (1e-4 / ((x - 0.8) ** 2 + 1e-8)
                + np.exp(-((x - 0.5) ** 2) * 1e4) * 0 + 1) * 0 \
            + 1.0 / np.cosh(1e4 * (x - 0.8)) + 1e-2 / (
                (x - 0.4) ** 2 + 1e-4)

    # canonical spike: sech(10^4 (x-0.8)) + narrow Lorentzian
    xs = np.linspace(0, 1, 6000)
    fig, ax = plt.subplots()
    ax.plot(xs, f_np(xs), "b", linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Spike function", fontsize=10)
    save(fig, "quad", "SpikeIntegral_01.png")

    xz = np.linspace(0.795, 0.805, 2000)
    fig, ax = plt.subplots()
    ax.semilogy(xz, f_np(xz), "b", linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Zoom, on semilogy axes", fontsize=10)
    save(fig, "quad", "SpikeIntegral_02.png")

    # adaptive quadrature convergence
    from scipy.integrate import quad as sciquad

    exact, _ = sciquad(f_np, 0, 1, limit=400, epsabs=1e-14,
                       epsrel=1e-14, points=[0.4, 0.8])
    ns = 2 ** np.arange(4, 16)
    errs = []
    for n in ns:
        xg = np.linspace(0, 1, int(n) + 1)
        errs.append(abs(np.trapezoid(f_np(xg), xg) - exact))
    fig, ax = plt.subplots()
    ax.loglog(ns, np.maximum(errs, 1e-18), ".-", color=CHEBFUN_BLUE,
              markersize=6, linewidth=0.8)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    ax.set_title("trapezoid error vs n (spike resolution)", fontsize=9)
    save(fig, "quad", "SpikeIntegral_03.png")

    # piecewise-split chebfun integral (genuine)
    fcheb = cj.chebfun(
        lambda x: 1.0 / jnp.cosh(1e4 * (x - 0.8)) + 1e-2 / (
            (x - 0.4) ** 2 + 1e-4),
        domain=[0.0, 0.4 - 0.05, 0.4 + 0.05, 0.8 - 0.01, 0.8 + 0.01,
                1.0])
    I = float(fcheb.sum())
    print(f"    chebfun integral = {I:.12f}  (scipy {exact:.12f})")
    fig, ax = plt.subplots()
    ax.plot(xs, f_np(xs), "b", linewidth=0.9)
    for bp in (0.35, 0.45, 0.79, 0.81):
        ax.axvline(bp, color="r", linewidth=0.5, linestyle="--")
    ax.set_title(f"piecewise integration: I = {I:.8f}", fontsize=9)
    save(fig, "quad", "SpikeIntegral_04.png")


def quadratureconvergence():
    """quad/QuadratureConvergence — Gauss/CC on a nonsmooth f."""
    from chebfunjax.utils.quadrature import legpts

    def f_np(x):
        return np.abs(np.asarray(x) - 0.3)

    xs = np.linspace(-1, 1, 2000)
    fig, ax = plt.subplots()
    ax.plot(xs, f_np(xs), color=CHEBFUN_BLUE, linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "quad", "QuadratureConvergence_01.png")

    exact = ((1 - 0.3) ** 2 + (1 + 0.3) ** 2) / 2
    # capped at 2^12: legpts is O(n^2) Golub-Welsch (task #10)
    nn = np.unique(np.round(2.0 ** np.arange(1, 12.5, 0.5)).astype(int))
    errg, errc = [], []
    for n in nn:
        if n < 2:
            continue
        s, w = legpts(int(n))
        errg.append(abs(float(np.sum(np.asarray(w) * f_np(
            np.asarray(s)))) - exact))
        N_ = int(n) - 1
        if N_ < 2:
            errc.append(np.nan)
            continue
        theta = PI * np.arange(int(n)) / N_
        v = np.ones(int(n) - 2)
        for kk in range(1, N_ // 2 + 1):
            fac = 2.0 if kk < N_ / 2 else 1.0
            v = v - fac * np.cos(2 * kk * theta[1:-1]) / (4 * kk**2 - 1)
        wc = np.zeros(int(n))
        wc[1:-1] = 2 * v / N_
        wc[0] = wc[-1] = 1.0 / (N_**2 - 1 + (N_ % 2))
        xcc = np.cos(theta)
        errc.append(abs(float(np.sum(wc * f_np(xcc))) - exact))
    nn = nn[nn >= 2]
    fig, ax = plt.subplots()
    ax.loglog(nn, np.maximum(errg, 1e-18), ".-", color=CHEBFUN_BLUE,
              markersize=5, linewidth=0.6, label="Gauss")
    ax.loglog(nn, np.maximum(errc, 1e-18), ".-", color=ORANGE,
              markersize=5, linewidth=0.6, label="Clenshaw-Curtis")
    ax.loglog(nn, 0.1 * nn.astype(float) ** -2.0, "k--", linewidth=0.7)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    save(fig, "quad", "QuadratureConvergence_02.png")


def symbolicnumeric():
    """quad/SymbolicNumeric — integrals symbolic systems struggle with."""
    def f_np(x):
        x = np.asarray(x)
        return np.log(2 + x) ** 3 * np.log(3 + x) * x**3

    f = cj.chebfun(lambda x: jnp.log(2 + x) ** 3 * jnp.log(3 + x)
                   * x**3)
    xs = np.linspace(-1, 1, 2000)
    fig, ax = plt.subplots()
    ax.plot(xs, f_np(xs), color=CHEBFUN_BLUE, linewidth=2.0)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title(f"sum(f) = {float(f.sum()):.12f}", fontsize=9)
    save(fig, "quad", "SymbolicNumeric_01.png")

    g = cj.chebfun(lambda x: jnp.log(2 + x) ** 3
                   * jnp.log(3 + x) ** 2 * x**3)
    gi = g.cumsum()
    fig, ax = plt.subplots()
    ax.plot(xs, np.asarray(gi(jnp.asarray(xs))), color=(0.7, 0, 0.7),
            linewidth=2.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("indefinite integral of g", fontsize=9)
    save(fig, "quad", "SymbolicNumeric_02.png")


def hermitequad():
    """quad/HermiteQuad — Gauss-Hermite points on exp(-x^2) f."""
    n = 20
    s, w = np.polynomial.hermite.hermgauss(n)

    def g_np(x):
        x = np.asarray(x)
        return np.exp(-(x**2)) * np.cos(3 * x) * (1 + x / 4)

    xs = np.linspace(-8, 8, 3000)
    fig, ax = plt.subplots()
    ax.plot(xs, g_np(xs), color=CHEBFUN_BLUE, linewidth=2.0)
    ax.plot(s, g_np(s), ".r", markersize=10)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "quad", "HermiteQuad_01.png")
    Iq = float(np.sum(w * np.cos(3 * s) * (1 + s / 4)))
    print(f"    Hermite quadrature = {Iq:.12f} "
          f"(exact sqrt(pi) e^-9/4 cos-part = "
          f"{np.sqrt(PI) * np.exp(-9 / 4):.12f})")


PAGES = {
    "CondVandermonde": condvandermonde,
    "ResolventNorm": resolventnorm,
    "WikiIntegroDiff": wikiintegrodiff,
    "FracCalcCopy": copy_fraccalc,
    "GaussClenCurt": gaussclencurt,
    "SpikeIntegral": spikeintegral,
    "QuadratureConvergence": quadratureconvergence,
    "SymbolicNumeric": symbolicnumeric,
    "HermiteQuad": hermitequad,
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
