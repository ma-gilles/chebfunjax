"""Generate per-block figures for the cheb, calc, and applics example
categories."""

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


def _chebcoeffs(vals):
    n = len(vals)
    ext = np.concatenate([vals[::-1], vals[1:-1]])
    c = np.real(np.fft.fft(ext)) / (n - 1)
    c = c[:n]
    c[0] /= 2
    c[-1] /= 2
    return c


# ----------------------------- cheb ---------------------------------

def chebpolyshigham():
    """cheb/ChebPolysHigham — 3D waterfalls of T_k and P_k."""
    ks = [0, 2, 4, 10, 20, 40, 60]
    xs = np.linspace(-1, 1, 1200)
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for name, basis in (("ChebPolysHigham_01.png", "cheb"),
                        ("ChebPolysHigham_02.png", "leg")):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for j, k in enumerate(ks):
            if basis == "cheb":
                ys = np.cos(k * np.arccos(xs))
            else:
                c = np.zeros(k + 1)
                c[k] = 1.0
                ys = np.polynomial.legendre.legval(xs, c)
            ax.plot3D(np.full_like(xs, j + 1), xs, ys,
                      color=cyc[j % len(cyc)], linewidth=1.4)
        ax.set_xlim(1, len(ks))
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.view_init(elev=25, azim=-100)
        save(fig, "cheb", name)


def convergence():
    """cheb/Convergence — algebraic convergence for nonsmooth f."""
    nn = 2 * np.round(2.0 ** np.arange(0, 7.5, 0.5)).astype(int)

    for name, f_np, slope in (
            ("Convergence_01.png",
             lambda x: np.abs(x) ** PI, PI),
            ("Convergence_02.png",
             lambda x: np.sin(np.abs(x) ** (x + 5.5)), None)):
        xs = np.linspace(-1, 1, 4000)
        fv = f_np(xs)
        ee = []
        for n in nn:
            xc = np.cos(PI * np.arange(int(n)) / (int(n) - 1))
            c = _chebcoeffs(f_np(xc[::-1])[::-1])
            ee.append(np.max(np.abs(
                fv - np.polynomial.chebyshev.chebval(xs, c))))
        fig, ax = plt.subplots()
        ax.loglog(nn, ee, ".", markersize=9, color=CHEBFUN_BLUE)
        if slope:
            ax.loglog(nn, 3.0 * nn.astype(float) ** (-slope), "r--",
                      linewidth=0.8)
        ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
        ax.set_xlabel("n")
        ax.set_ylabel("max error")
        save(fig, "cheb", name)


def doublelengthflag():
    """cheb/DoublelengthFlag — doubled-length coefficient plots."""
    def coeffs_of(f_np, dom, n):
        a, b = dom
        xc = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(
            PI * np.arange(n) / (n - 1))
        return np.abs(_chebcoeffs(f_np(xc[::-1])[::-1]))

    cases = [
        ("DoublelengthFlag_01.png", np.exp, (-1.0, 1.0), 15),
        ("DoublelengthFlag_02.png",
         lambda x: np.sin(x) + np.sin(np.asarray(x) ** 2), (0.0, 10.0),
         120),
        ("DoublelengthFlag_03.png",
         lambda x: 1.0 / (1 + 25 * np.asarray(x) ** 2), (-1.0, 1.0),
         182),
    ]
    for name, f_np, dom, n in cases:
        c1 = coeffs_of(f_np, dom, n)
        c2 = coeffs_of(f_np, dom, 2 * n)
        fig, ax = plt.subplots()
        ax.semilogy(np.arange(len(c2)), np.maximum(c2, 1e-40), ".",
                    markersize=4, color=CHEBFUN_BLUE)
        ax.semilogy(np.arange(len(c1)), np.maximum(c1, 1e-40), ".",
                    markersize=4, color=ORANGE)
        ax.grid(True, alpha=0.4, linewidth=0.4)
        ax.set_title("coefficients: standard vs doublelength",
                     fontsize=9)
        save(fig, "cheb", name)


def exactchebcoeffs():
    """cheb/ExactChebCoeffs — 1/(5-4x)-type exact coefficients."""
    f = cj.chebfun(lambda x: 1.0 / (5 - 4 * x))
    c = np.abs(np.asarray(f.funs[0].tech.coeffs))
    k = np.arange(1, len(c) + 1)
    exact = (1 / np.sqrt(6)) / (5 + np.sqrt(24)) ** (k - 1)
    fig, ax = plt.subplots()
    ax.semilogy(k - 1, np.maximum(c, 1e-18), "o", markersize=5,
                markerfacecolor="none", color=CHEBFUN_BLUE)
    ax.semilogy(k - 1, exact, ".r", markersize=4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("computed vs exact Chebyshev coefficients",
                 fontsize=9)
    save(fig, "cheb", "ExactChebCoeffs_01.png")


def turbo():
    """cheb/Turbo — extended-precision-style coefficient tails."""
    from scipy.special import iv

    f = cj.chebfun(lambda x: jnp.exp(x))
    n = len(f)
    c = np.abs(np.asarray(f.funs[0].tech.coeffs))

    # "turbo" tail: exact coefficients 2 I_k(1) continue below eps
    kk = np.arange(4 * n)
    cexact = 2 * iv(kk, 1.0)
    cexact[0] /= 2

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(n), np.maximum(c, 1e-45), ".", markersize=6,
                color=CHEBFUN_BLUE, label="standard")
    ax.semilogy(kk, cexact, ".", markersize=3, color=ORANGE,
                label="turbo/exact tail")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "cheb", "Turbo_01.png")

    fig, ax = plt.subplots()
    ax.semilogy(kk, cexact, ".", markersize=4, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("exact coefficients 2 I_k(1) to k = 4n", fontsize=9)
    save(fig, "cheb", "Turbo_02.png")


# ----------------------------- calc ---------------------------------

def forthebirds():
    """calc/ForTheBirds — energy-optimal flight path."""
    island = np.array([0.0, 5.0])
    nest = np.array([13.0, 0.0])

    def energy(x):
        # fly over water (cost 2/unit) to landfall (x, 0), then land
        over_water = np.hypot(x - island[0], island[1])
        over_land = np.abs(nest[0] - x)
        return 2.0 * over_water + 1.0 * over_land

    xs = np.linspace(0, 13, 800)
    ev = energy(xs)

    # geometry sketch
    fig, ax = plt.subplots()
    ax.plot([island[0]], [island[1]], "^k", markersize=10)
    ax.plot([nest[0]], [nest[1]], "sk", markersize=9)
    ax.plot([0, 13], [0, 0], "-", color=(0.6, 0.4, 0.2), linewidth=2)
    xstar = xs[np.argmin(ev)]
    ax.plot([island[0], xstar, nest[0]], [island[1], 0, 0], "--b",
            linewidth=1.2)
    ax.set_title("island, landfall, nest", fontsize=10)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "calc", "ForTheBirds_01.png")

    fig, ax = plt.subplots()
    ax.plot(xs, ev, linewidth=2.0, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlabel("landfall point x")
    ax.set_ylabel("total energy of flight")
    save(fig, "calc", "ForTheBirds_02.png")

    # derivative and its root
    f = cj.chebfun(lambda x: 2 * jnp.sqrt(x**2 + 25.0) + (13.0 - x),
                   domain=[0.0, 13.0])
    df = f.diff()
    r = np.asarray(df.roots())
    fig, ax = plt.subplots()
    ax.plot(xs, np.asarray(df(jnp.asarray(xs))), color=ORANGE,
            linewidth=1.4)
    ax.axhline(0, color="k", linewidth=0.6)
    if len(r):
        ax.plot(r, np.zeros_like(r), ".r", markersize=10)
        print(f"    optimal landfall x* = {float(r[0]):.6f} "
              f"(exact 5/sqrt(3) = {5/np.sqrt(3):.6f})")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("energy derivative and the optimum", fontsize=9)
    save(fig, "calc", "ForTheBirds_03.png")

    # varying water/land cost ratio
    fig, ax = plt.subplots()
    for ratio in (1.2, 1.5, 2.0, 3.0):
        ax.plot(xs, ratio * np.hypot(xs, 5.0) + (13 - xs),
                linewidth=1.1, label=f"ratio {ratio:g}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "calc", "ForTheBirds_04.png")


def integrals():
    """calc/Integrals — round(2cos x) and its antiderivative."""
    breaks = [0.0]
    # round(2cos x) jumps where 2cos x = +-0.5, +-1.5
    for lvl in (0.5, 1.5):
        for k in range(8):
            for s in (1, -1):
                v = np.arccos(s * lvl / 2)
                for root in (v + 2 * PI * k, 2 * PI - v + 2 * PI * k):
                    if 0 < root < 10:
                        breaks.append(float(root))
    breaks = sorted(set(breaks + [10.0]))
    f = cj.chebfun(lambda x: jnp.round(2 * jnp.cos(x)), domain=breaks)
    xs = jnp.linspace(0, 10, 3000)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(f(xs)), color=CHEBFUN_BLUE,
            linewidth=1.2)
    ax.set_ylim(-2.5, 2.5)
    save(fig, "calc", "Integrals_01.png")

    g = f.cumsum()
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(g(xs)), "m", linewidth=1.4)
    save(fig, "calc", "Integrals_02.png")

    print(f"    sum(f) = {float(f.sum()):.10f}")
    h = cj.chebfun(lambda x: jnp.exp(-(x**2)), domain=[0.0, 10.0])
    hi = h.cumsum()
    fig, ax = plt.subplots()
    ax.plot(np.asarray(xs), np.asarray(hi(xs)), color=CHEBFUN_BLUE,
            linewidth=1.4)
    ax.set_title("cumsum of exp(-x^2): erf shape", fontsize=9)
    save(fig, "calc", "Integrals_03.png")


def meanvaluetheorem():
    """calc/MeanValueTheorem — the MVT point and tangent."""
    a, b = 0.3, 1.8

    def f_np(x):
        return np.asarray(x) * np.sin(2 * np.asarray(x))

    f = cj.chebfun(lambda x: x * jnp.sin(2 * x), domain=[0.0, 2.0])
    slope = (f_np(b) - f_np(a)) / (b - a)
    df = f.diff()
    roots = np.asarray(
        (df - slope).roots() if hasattr(df - slope, "roots") else [])
    cands = [r for r in roots if a < r < b]
    c0 = cands[0] if cands else 0.5 * (a + b)
    xs = np.linspace(0, 2, 1000)
    fig, ax = plt.subplots()
    ax.plot(xs, f_np(xs), color=CHEBFUN_BLUE, linewidth=1.4)
    ax.plot([a, b], [f_np(a), f_np(b)], "--k", linewidth=1.0)
    ax.plot([c0], [f_np(c0)], ".r", markersize=12)
    L = 0.4
    ax.plot([c0 - L, c0 + L],
            [f_np(c0) - L * slope, f_np(c0) + L * slope], "r",
            linewidth=1.2)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "calc", "MeanValueTheorem_01.png")
    print(f"    MVT point c = {float(c0):.6f}")


def snellslaw():
    """calc/SnellsLaw — lifeguard/drowning-man refraction."""
    sMan = np.array([-5.0, 5.0])
    dMan = np.array([5.0, -5.0])

    def beach_axes():
        fig, ax = plt.subplots()
        ax.fill_between([-6, 6], -6, 0, color=(0.0, 0.8, 1.0))
        ax.axhline(0, color="k", linewidth=1.0)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        return fig, ax

    fig, ax = beach_axes()
    ax.set_title("Beach", fontsize=10)
    save(fig, "calc", "SnellsLaw_01.png")

    fig, ax = beach_axes()
    ax.plot(*sMan, ".k", markersize=16)
    ax.set_title("Beach, lifeguard", fontsize=10)
    save(fig, "calc", "SnellsLaw_02.png")

    fig, ax = beach_axes()
    ax.plot(*sMan, ".k", markersize=16)
    ax.plot(*dMan, ".r", markersize=16)
    ax.set_title("Beach, lifeguard, drowning man", fontsize=10)
    save(fig, "calc", "SnellsLaw_03.png")

    # time to reach as a function of entry point x (run 8, swim 2)
    v_run, v_swim = 8.0, 2.0

    def T(x):
        x = np.asarray(x, dtype=float)
        return (np.hypot(x - sMan[0], sMan[1]) / v_run
                + np.hypot(dMan[0] - x, dMan[1]) / v_swim)

    xs = np.linspace(-6, 6, 900)
    xstar = xs[np.argmin(T(xs))]
    fig, ax = plt.subplots()
    ax.plot(xs, T(xs), color=CHEBFUN_BLUE, linewidth=1.6)
    ax.plot([xstar], [T(xstar)], ".r", markersize=12)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("total time vs water-entry point", fontsize=10)
    save(fig, "calc", "SnellsLaw_04.png")
    print(f"    optimal entry x* = {xstar:.4f}")

    fig, ax = beach_axes()
    ax.plot(*sMan, ".k", markersize=12)
    ax.plot(*dMan, ".r", markersize=12)
    ax.plot([sMan[0], xstar], [sMan[1], 0], color=CHEBFUN_BLUE,
            linewidth=1.4)
    ax.plot([xstar, dMan[0]], [0, dMan[1]], "r", linewidth=1.4)
    ax.set_title("Beach, lifeguard, rescued man", fontsize=10)
    save(fig, "calc", "SnellsLaw_05.png")

    fig, ax = beach_axes()
    ax.plot(*sMan, ".k", markersize=12)
    ax.plot(*dMan, ".r", markersize=12)
    ax.plot([sMan[0], xstar], [sMan[1], 0], color=CHEBFUN_BLUE,
            linewidth=1.4)
    ax.plot([xstar, dMan[0]], [0, dMan[1]], "r", linewidth=1.4)
    ax.plot([xstar, xstar], [-4.5, 4.5], "--k", linewidth=1.0)
    th1 = np.arctan2(sMan[1], sMan[0] - xstar)
    arc = np.linspace(np.pi / 2, th1, 40)
    ax.plot(xstar + 1.6 * np.cos(arc), 1.6 * np.sin(arc), "b",
            linewidth=0.9)
    ax.plot([xstar - 0.35, xstar - 0.35, xstar],
            [-1.4, -1.05, -1.05], "r", linewidth=0.9)
    ax.set_title("Beach, lifeguard, rescued man", fontsize=10)
    save(fig, "calc", "SnellsLaw_06.png")


# --------------------------- applics --------------------------------

def bode2tf():
    """applics/Bode2tf — transfer-function identification via AAA."""
    from chebfunjax.utils.aaa import aaa

    def G(s):
        return 1.0 / ((s + 0.1) * (s**2 + s + 1))

    w = np.logspace(-4, 2, 3000)
    mag = np.abs(G(1j * w))
    ph = -np.angle(G(1j * w))

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.loglog(w, mag, color=CHEBFUN_BLUE, linewidth=1.0)
    ax1.set_title("magnitude", fontsize=8)
    ax2.semilogx(w, ph, color=ORANGE, linewidth=1.0)
    ax2.set_title("phase", fontsize=8)
    for a in (ax1, ax2):
        a.grid(True, which="both", alpha=0.4, linewidth=0.4)
        a.tick_params(labelsize=6)
    save(fig, "applics", "Bode2tf_01.png")

    # AAA fit of G on the imaginary axis (conjugate-symmetric data)
    wA = np.concatenate([-w[::-1], w])
    GA = np.concatenate([np.conj(G(1j * w))[::-1], G(1j * w)])
    r, pol, res, *_ = aaa(GA, 1j * wA, tol=1e-12, mmax=20)
    print("    identified poles:", np.round(np.sort_complex(pol)[:4], 4))

    fig, ax = plt.subplots()
    ax.plot(np.real(pol), np.imag(pol), "xr", markersize=9,
            markeredgewidth=2)
    ax.axvline(0, color="k", linewidth=0.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("poles recovered from Bode data", fontsize=9)
    save(fig, "applics", "Bode2tf_02.png")

    # reconstruction error along the axis
    fig, ax = plt.subplots()
    err = np.abs(G(1j * w) - r(1j * w))
    ax.loglog(w, np.maximum(err, 1e-18), color=CHEBFUN_BLUE,
              linewidth=0.9)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    ax.set_title("AAA reconstruction error", fontsize=9)
    save(fig, "applics", "Bode2tf_03.png")

    # impulse response from residues: g(t) = sum res exp(pol t)
    ts = np.linspace(0, 30, 800)
    gt = np.real(np.array([np.sum(res * np.exp(pol * t))
                           for t in ts]))
    fig, ax = plt.subplots()
    ax.plot(ts, gt, color=CHEBFUN_BLUE, linewidth=1.3)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("impulse response from identified poles/residues",
                 fontsize=9)
    save(fig, "applics", "Bode2tf_04.png")

    # step response
    st = np.array([np.sum(np.real(res / pol * (np.exp(pol * t) - 1)))
                   for t in ts])
    fig, ax = plt.subplots()
    ax.plot(ts, st, color=ORANGE, linewidth=1.3)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("step response", fontsize=9)
    save(fig, "applics", "Bode2tf_05.png")

    # magnitude fit overlay
    fig, ax = plt.subplots()
    ax.loglog(w, mag, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.loglog(w, np.abs(r(1j * w)), "--", color=ORANGE, linewidth=1.0)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    save(fig, "applics", "Bode2tf_06.png")


def gompertz():
    """applics/Gompertz — population growth models via Chebop."""
    import warnings

    from chebfunjax.operators.chebop import Chebop

    ts = jnp.linspace(0, 25, 600)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        N1 = Chebop(lambda t, P: P.diff() - 0.5 * P, domain=(0.0, 25.0),
                    lbc=0.2)
        exp_result = N1.solve(0.0)
    fig, ax = plt.subplots()
    ax.semilogy(np.asarray(ts), np.asarray(exp_result(ts)),
                color=CHEBFUN_BLUE, linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("exponential growth", fontsize=10)
    save(fig, "applics", "Gompertz_01.png")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        N2 = Chebop(lambda t, P: P.diff() - P * (0.5 * (1 - P / 5.0)),
                    domain=(0.0, 25.0), lbc=0.2)
        logi = N2.solve(0.0)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(ts), np.asarray(logi(ts)), color=CHEBFUN_BLUE,
            linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("logistic growth", fontsize=10)
    save(fig, "applics", "Gompertz_02.png")

    # Gompertz: P' = a P log(K/P) — closed form for the figure overlay
    a_, K = 0.5, 5.0
    t_np = np.asarray(ts)
    P0 = 0.2
    gomp = K * np.exp(np.log(P0 / K) * np.exp(-a_ * t_np))
    fig, ax = plt.subplots()
    ax.plot(t_np, gomp, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("Gompertz growth", fontsize=10)
    save(fig, "applics", "Gompertz_03.png")

    fig, ax = plt.subplots()
    ax.plot(t_np, np.asarray(logi(ts)), linewidth=1.2,
            label="logistic")
    ax.plot(t_np, gomp, linewidth=1.2, label="Gompertz")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "applics", "Gompertz_04.png")


def step2tf():
    """applics/Step2tf — system identification from a step response."""
    from chebfunjax.utils.aaa import aaa

    def G(s):
        return 1.0 / ((s + 0.5) * (s**2 + 0.4 * s + 1))

    w = np.logspace(-3, 2, 2000)

    def Stp(s):
        return 1.0 / s

    wA = np.concatenate([-w[::-1], w])
    GS = Stp(1j * w) * G(1j * w)
    GSA = np.concatenate([np.conj(GS)[::-1], GS])
    r, polG, resG, *_ = aaa(GSA, 1j * wA, tol=1e-10, mmax=24)
    polH = polG[np.real(polG) < -1e-8]
    print(f"    system poles found: {len(polH)}")

    ts = np.linspace(0, 40, 900)
    # true step response by residue expansion of G(s)/s

    step_vals = np.array([np.sum(np.real(
        resG * np.exp(polG * t))) for t in ts])
    fig, ax = plt.subplots()
    ax.plot(ts, step_vals, color=CHEBFUN_BLUE, linewidth=1.4)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("step response (identified)", fontsize=10)
    save(fig, "applics", "Step2tf_01.png")

    fig, ax = plt.subplots()
    ax.plot(np.real(polH), np.imag(polH), "xr", markersize=9,
            markeredgewidth=2)
    ax.axvline(0, color="k", linewidth=0.6)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("stable poles of the identified system", fontsize=9)
    save(fig, "applics", "Step2tf_02.png")

    # exponential-fit reconstruction of the step data (least squares)
    g_data = step_vals
    Q = np.exp(ts[:, None] * polH[None, :])
    resH, *_ = np.linalg.lstsq(Q, g_data.astype(complex), rcond=None)
    recon = np.real(Q @ resH)
    fig, ax = plt.subplots()
    ax.plot(ts, g_data, color=CHEBFUN_BLUE, linewidth=1.6,
            label="data")
    ax.plot(ts, recon, "--", color=ORANGE, linewidth=1.1,
            label="exp-sum fit")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "applics", "Step2tf_03.png")

    fig, ax = plt.subplots()
    ax.semilogy(ts, np.maximum(np.abs(g_data - recon), 1e-18),
                color=CHEBFUN_BLUE, linewidth=0.9)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_title("fit error", fontsize=10)
    save(fig, "applics", "Step2tf_04.png")

    # Bode magnitude of the identified H
    def H(s):
        s = np.atleast_1d(s)
        return (1.0 / (s[:, None] - polH[None, :])) @ resH

    fig, ax = plt.subplots()
    ax.loglog(w, np.abs(G(1j * w)), color=CHEBFUN_BLUE, linewidth=1.4,
              label="true G")
    ax.loglog(w, np.abs(H(1j * w) * 1j * w), "--", color=ORANGE,
              linewidth=1.0, label="identified")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.4, linewidth=0.4)
    save(fig, "applics", "Step2tf_05.png")


PAGES = {
    "ChebPolysHigham": chebpolyshigham,
    "Convergence": convergence,
    "DoublelengthFlag": doublelengthflag,
    "ExactChebCoeffs": exactchebcoeffs,
    "Turbo": turbo,
    "ForTheBirds": forthebirds,
    "Integrals": integrals,
    "MeanValueTheorem": meanvaluetheorem,
    "SnellsLaw": snellslaw,
    "Bode2tf": bode2tf,
    "Gompertz": gompertz,
    "Step2tf": step2tf,
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
