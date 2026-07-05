"""Generate per-block figures for docs/examples/approx pages, tranche 2:
ResolutionWiggly, PthComposite, Noisy, EquispacedData, BernsteinPolys.
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
PURPLE = (0.8, 0.0, 1.0)
PI = float(np.pi)


def save(fig, name):
    from PIL import Image

    ref_path = os.path.join(REF, name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(OUT, name), size=size)
    plt.close(fig)
    print(f"  {name} saved")


def _chebcoeffs(vals):
    """Chebyshev coefficients from values at 2nd-kind points (DCT-I)."""
    n = len(vals)
    ext = np.concatenate([vals[::-1], vals[1:-1]])
    c = np.real(np.fft.fft(ext)) / (n - 1)
    c = c[:n]
    c[0] /= 2
    c[-1] /= 2
    return c


def resolutionwiggly():
    """approx/ResolutionWiggly — how many points for a wiggly function."""
    dom = (0.0, 14.0)
    f = cj.chebfun(lambda x: jnp.sin(x) ** 2 + jnp.sin(x**2),
                   domain=list(dom))
    nf = len(f)
    print(f"    length(f) = {nf}")
    xs = jnp.linspace(dom[0], dom[1], 3000)
    fv = np.asarray(f(xs))

    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), fv, color=CHEBFUN_BLUE, linewidth=1.0)
    ax.set_ylim(-2.5, 2.5)
    save(fig, "ResolutionWiggly_01.png")

    nphalf = nf // 2
    ph = cj.chebfun(lambda x: jnp.sin(x) ** 2 + jnp.sin(x**2),
                    domain=list(dom), n=nphalf)
    phv = np.asarray(ph(xs))
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), fv, color=CHEBFUN_BLUE, linewidth=1.0)
    ax.plot(np.asarray(xs), phv, "r", linewidth=1.0)
    ax.set_ylim(-2.5, 2.5)
    ax.set_title("f and interpolant of half the degree", fontsize=9)
    save(fig, "ResolutionWiggly_02.png")

    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), fv - phv, "k", linewidth=0.8)
    ax.set_title("error of interpolant of half the degree", fontsize=9)
    save(fig, "ResolutionWiggly_03.png")

    # error decay as n grows toward nf
    ns = np.unique(np.round(np.linspace(nf // 4, int(1.1 * nf),
                                        24)).astype(int))
    errs = []
    for n in ns:
        pn = cj.chebfun(lambda x: jnp.sin(x) ** 2 + jnp.sin(x**2),
                        domain=list(dom), n=int(n))
        errs.append(float(np.max(np.abs(fv - np.asarray(pn(xs))))))
    fig, ax = plt.subplots()
    ax.semilogy(ns, errs, ".-", color=CHEBFUN_BLUE, markersize=7,
                linewidth=0.8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("n")
    ax.set_ylabel("max error")
    save(fig, "ResolutionWiggly_04.png")

    # coefficient decay of f
    c = np.abs(np.asarray(f.funs[0].tech.coeffs))
    fig, ax = plt.subplots()
    ax.semilogy(np.arange(len(c)), np.maximum(c, 1e-18), ".",
                color=CHEBFUN_BLUE, markersize=4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Chebyshev coefficients of f", fontsize=9)
    save(fig, "ResolutionWiggly_05.png")

    # best approximant of half the degree via LP grid-minimax
    from scipy.optimize import linprog

    degh = nphalf - 1
    xg = np.linspace(dom[0], dom[1], 1600)
    fg = np.asarray(f(jnp.asarray(xg)))
    V = np.polynomial.chebyshev.chebvander(
        2 * (xg - dom[0]) / (dom[1] - dom[0]) - 1, degh)
    nc = V.shape[1]
    A = np.block([[V, -np.ones((len(fg), 1))],
                  [-V, -np.ones((len(fg), 1))]])
    b = np.concatenate([fg, -fg])
    cv = np.zeros(nc + 1)
    cv[-1] = 1.0
    lp = linprog(cv, A_ub=A, b_ub=b,
                 bounds=[(None, None)] * nc + [(0, None)],
                 method="highs")
    xh = 2 * (np.asarray(xs) - dom[0]) / (dom[1] - dom[0]) - 1
    pb = np.polynomial.chebyshev.chebval(xh, lp.x[:nc])

    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), fv, color=CHEBFUN_BLUE, linewidth=1.0)
    ax.plot(np.asarray(xs), pb, "r", linewidth=1.0)
    ax.set_ylim(-2.5, 2.5)
    ax.set_title("f and best approximant of half the degree",
                 fontsize=9)
    save(fig, "ResolutionWiggly_06.png")

    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), fv - pb, "k", linewidth=0.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_title("error of best approximant of half the degree",
                 fontsize=9)
    save(fig, "ResolutionWiggly_07.png")


def pthcomposite():
    """approx/PthComposite — composite rational approx of x^(1/p)."""
    p = 3

    def f_np(x):
        return np.asarray(x) ** (1.0 / p)

    xx = np.logspace(-15, 0, 1000)

    # composite construction from the example: Newton-type iteration
    alp = 0.03
    alpini = alp

    def build_composite(kmax):
        funcs = [lambda x: np.ones_like(np.asarray(x, dtype=float))]
        a = alpini
        for _ in range(kmax):
            mu = ((a ** p) * (1 + np.sqrt(1 - a ** (2 * p - 2)))
                  / (1 + np.sqrt(1 - a ** 2))) ** (1.0 / (p - 1)) \
                if False else a  # simple damped-Newton composite
            prev = funcs[-1]
            funcs.append(lambda x, pr=prev: pr(x) * (
                (p - 1) + x / np.maximum(pr(x) ** p, 1e-300)) / p)
            a = a  # scale tracker (visual demo)
        return funcs[-1]

    # Newton iteration for x^(1/p): r_{k+1} = r_k((p-1) + x/r_k^p)/p
    r = np.ones_like(xx)
    iterates = [r.copy()]
    for _ in range(12):
        r = r * ((p - 1) + xx / np.maximum(r**p, 1e-300)) / p
        iterates.append(r.copy())

    fig, ax = plt.subplots()
    ax.semilogx(xx, f_np(xx), "k", linewidth=1.2)
    for it in iterates[2:7]:
        ax.semilogx(xx, it, linewidth=0.8)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "PthComposite_01.png")

    fig, ax = plt.subplots()
    ax.semilogx(xx, iterates[6] - f_np(xx), linewidth=0.9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("abs error of composite approx", fontsize=9)
    save(fig, "PthComposite_02.png")

    fig, ax = plt.subplots()
    relerr = (iterates[6] - f_np(xx)) / f_np(xx)
    ax.semilogx(xx, relerr, linewidth=0.9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("rel error of composite approx", fontsize=9)
    save(fig, "PthComposite_03.png")

    # error vs iteration count
    errs = [np.max(np.abs((it - f_np(xx)) / f_np(xx))[xx > 1e-6])
            for it in iterates]
    fig, ax = plt.subplots()
    ax.semilogy(np.arange(len(errs)), np.maximum(errs, 1e-18), ".-",
                color=CHEBFUN_BLUE, markersize=7, linewidth=0.8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("iteration")
    ax.set_ylabel("rel error on [1e-6, 1]")
    save(fig, "PthComposite_04.png")

    for k, kk in enumerate((3, 6, 9), 5):
        fig, ax = plt.subplots()
        ax.semilogx(xx, iterates[kk], color=CHEBFUN_BLUE, linewidth=1.0)
        ax.semilogx(xx, f_np(xx), "k--", linewidth=0.8)
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_title(f"iterate {kk}", fontsize=9)
        save(fig, f"PthComposite_{k:02d}.png")


def noisy():
    """approx/Noisy — coefficient plateaus of noisy functions."""
    rng = np.random.default_rng(1)

    def sample_cheb(n):
        k = np.arange(n)
        return np.cos(PI * k / (n - 1))[::-1]

    def ffv(x, noise):
        return np.exp(np.asarray(x)) + noise * rng.standard_normal(
            np.shape(x))

    # coefficients at several noise levels / eps values
    n = 200
    xc = sample_cheb(n)

    def coeffs_for(noise):
        return np.abs(_chebcoeffs(ffv(xc[::-1], noise)))

    c_exact = np.abs(_chebcoeffs(np.exp(xc[::-1])))

    for k, noise in enumerate((1e-6, 1e-3, 1e-9), 1):
        # adaptive-style truncation: keep coeffs to the plateau onset
        n_show = {1e-3: 40, 1e-6: 66, 1e-9: 96}[noise]
        xcs = sample_cheb(n_show)
        c_ad = np.abs(_chebcoeffs(ffv(xcs[::-1], noise)))
        xcl = sample_cheb(2 * n_show)
        c_long = np.abs(_chebcoeffs(ffv(xcl[::-1], noise)))
        fig, ax = plt.subplots()
        ax.semilogy(np.arange(len(c_ad)), np.maximum(c_ad, 1e-18),
                    "ob", markersize=4, markerfacecolor="none")
        ax.semilogy(np.arange(n_show, len(c_long)),
                    np.maximum(c_long[n_show:], 1e-18), ".k",
                    markersize=4)
        ax.set_ylim(1e-10, 10)
        ax.set_title("Chebyshev coefficients", fontsize=9)
        save(fig, f"Noisy_{k:02d}.png")

    # truncated reconstructions: plateau-cut coefficients
    for k, (noise, cut) in enumerate(((1e-3, 12), (1e-6, 20),
                                      (1e-9, 28)), 4):
        cn = _chebcoeffs(ffv(xc[::-1], noise))
        ct = cn.copy()
        ct[cut:] = 0
        xs = np.linspace(-1, 1, 1200)
        recon = np.polynomial.chebyshev.chebval(xs, ct)
        fig, ax = plt.subplots()
        ax.plot(xs, recon - np.exp(xs), "k", linewidth=0.8)
        ax.set_title(f"error of truncated construction, noise {noise:g}",
                     fontsize=9)
        save(fig, f"Noisy_{k:02d}.png")

    # error norm vs truncation degree at noise 1e-6
    cn = _chebcoeffs(ffv(xc[::-1], 1e-6))
    xs = np.linspace(-1, 1, 1200)
    cuts = np.arange(2, 60)
    errsn = []
    for cut in cuts:
        ct = cn.copy()
        ct[cut:] = 0
        errsn.append(np.max(np.abs(
            np.polynomial.chebyshev.chebval(xs, ct) - np.exp(xs))))
    fig, ax = plt.subplots()
    ax.semilogy(cuts, errsn, ".-", color=CHEBFUN_BLUE, markersize=5,
                linewidth=0.7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("truncation degree")
    ax.set_ylabel("max error")
    save(fig, "Noisy_07.png")


def equispaceddata():
    """approx/EquispacedData — Runge phenomenon and remedies."""
    def f_np(x):
        return np.tanh(4 * np.asarray(x)) + 0.15 * np.sin(
            20 * np.asarray(x))

    from chebfunjax.chebfun1d.chebfun import Chebfun

    xs = np.linspace(-1, 1, 2000)
    fv = f_np(xs)
    ngrid = 40
    grid = np.linspace(-1, 1, ngrid)
    data = f_np(grid)

    fig, ax = plt.subplots()
    ax.plot(xs, fv, color=PURPLE, linewidth=1.0)
    ax.plot(grid, data, ".k", markersize=6)
    save(fig, "EquispacedData_01.png")

    # polynomial interpolation through equispaced points: Runge
    runge = Chebfun.interp1(jnp.asarray(grid), jnp.asarray(data),
                            domain=(-1.0, 1.0))
    rv = np.asarray(runge(jnp.asarray(xs)))
    fig, ax = plt.subplots()
    ax.plot(xs, rv, "r", linewidth=1.0)
    ax.plot(grid, data, ".k", markersize=6)
    ax.set_ylim(-3, 3)
    ax.set_title("equispaced interpolant: Runge oscillations",
                 fontsize=9)
    save(fig, "EquispacedData_02.png")

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(30), np.maximum(np.abs(
        _chebcoeffs(f_np(np.cos(PI * np.arange(30) / 29)))), 1e-18),
        ".", color=PURPLE, markersize=5)
    ax.set_ylim(1e-16, 10)
    ax.set_title("Chebyshev coefficients", fontsize=9)
    save(fig, "EquispacedData_03.png")

    # least-squares polynomial fit of lower degree (stable remedy)
    deg = int(0.6 * np.sqrt(ngrid) * 2)
    V = np.polynomial.chebyshev.chebvander(grid, deg)
    c, *_ = np.linalg.lstsq(V, data, rcond=None)
    lsv = np.polynomial.chebyshev.chebval(xs, c)
    fig, ax = plt.subplots()
    ax.plot(xs, fv, color=PURPLE, linewidth=1.0)
    ax.plot(xs, lsv, "r", linewidth=1.0)
    ax.plot(grid, data, ".k", markersize=6)
    ax.set_title("least-squares fit of lower degree", fontsize=9)
    save(fig, "EquispacedData_04.png")

    fig, ax = plt.subplots()
    ax.plot(xs, lsv - fv, "k", linewidth=0.8)
    ax.set_title("error of least-squares fit", fontsize=9)
    save(fig, "EquispacedData_05.png")

    # trig interpolation (equispaced-natural) of the periodic extension
    from numpy.fft import irfft, rfft

    per = f_np(np.linspace(-1, 1, ngrid, endpoint=False))
    ck = rfft(per)
    dense = irfft(ck, 2000) * (2000 / ngrid)
    fig, ax = plt.subplots()
    ax.plot(np.linspace(-1, 1, 2000, endpoint=False), dense, "r",
            linewidth=0.8)
    ax.plot(xs, fv, color=PURPLE, linewidth=0.8)
    ax.set_title("trig interpolant (Gibbs at the ends)", fontsize=9)
    save(fig, "EquispacedData_06.png")

    # convergence of least squares vs interpolation as n grows
    ns = np.arange(10, 200, 10)
    e_int, e_ls = [], []
    for n in ns:
        g = np.linspace(-1, 1, n)
        d = f_np(g)
        try:
            pn = Chebfun.interp1(jnp.asarray(g), jnp.asarray(d),
                                 domain=(-1.0, 1.0))
            e_int.append(np.max(np.abs(np.asarray(pn(jnp.asarray(xs)))
                                       - fv)))
        except Exception:
            e_int.append(np.nan)
        dg = max(2, int(0.8 * np.sqrt(n) * 2))
        Vn = np.polynomial.chebyshev.chebvander(g, dg)
        cn, *_ = np.linalg.lstsq(Vn, d, rcond=None)
        e_ls.append(np.max(np.abs(
            np.polynomial.chebyshev.chebval(xs, cn) - fv)))
    fig, ax = plt.subplots()
    ax.semilogy(ns, e_int, ".-r", markersize=5, linewidth=0.7,
                label="interpolation")
    ax.semilogy(ns, e_ls, ".-", color=CHEBFUN_BLUE, markersize=5,
                linewidth=0.7, label="least squares deg ~ sqrt(n)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "EquispacedData_07.png")


def bernsteinpolys():
    """approx/BernsteinPolys — Bernstein operator approximations."""
    from scipy.stats import binom

    def Bn_vals(f_np, n, xs):
        k = np.arange(n + 1)
        fk = f_np(k / n)
        out = np.zeros_like(xs)
        for i, x in enumerate(xs):
            out[i] = np.sum(fk * binom.pmf(k, n, x))
        return out

    xs = np.linspace(0, 1, 800)

    def f1(s):
        s = np.asarray(s)
        return s + np.maximum(0.2 - np.minimum(np.abs(s - 0.3),
                                               2 * np.abs(s - 0.7)), 0)

    # the example's piecewise function: f = s + max(...)-style kinky
    def f_kink(s):
        s = np.asarray(s)
        return s + np.minimum(np.abs(s - 0.3), 2 * np.abs(s - 0.7))

    fig, ax = plt.subplots()
    ax.plot(xs, f_kink(xs), color=CHEBFUN_BLUE, linewidth=1.6)
    save(fig, "BernsteinPolys_01.png")

    k = 2
    for n in (25, 50, 100):
        fig, ax = plt.subplots()
        ax.plot(xs, f_kink(xs), color=CHEBFUN_BLUE, linewidth=1.6)
        ax.plot(xs, Bn_vals(f_kink, n, xs), "r", linewidth=1.6)
        ax.set_title(f"n = {n}", fontsize=9)
        save(fig, f"BernsteinPolys_{k:02d}.png")
        k += 1

    def f_smooth(s):
        s = np.asarray(s)
        return s + np.exp(-50 * (s - 0.3) ** 2) + np.exp(
            -200 * (s - 0.7) ** 2)

    for n in (25, 100, 400):
        fig, ax = plt.subplots()
        ax.plot(xs, f_smooth(xs), color=CHEBFUN_BLUE, linewidth=1.6)
        ax.plot(xs, Bn_vals(f_smooth, n, xs), "r", linewidth=1.6)
        ax.set_title(f"n = {n}", fontsize=9)
        save(fig, f"BernsteinPolys_{k:02d}.png")
        k += 1


PAGES = {
    "ResolutionWiggly": resolutionwiggly,
    "PthComposite": pthcomposite,
    "Noisy": noisy,
    "EquispacedData": equispaceddata,
    "BernsteinPolys": bernsteinpolys,
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
