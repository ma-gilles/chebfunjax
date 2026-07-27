"""Generate genuine chebfun.org example-plot placeholders (approx category, batch 2).

Ports the following chebfun.org example pages faithfully into chebfunjax:
  - AliasingCoefficients      -> _01, _02
  - AliasingCoefficientsLeg   -> _01, _02
  - Entire                    -> _01, _02
  - EntireBound               -> _01, _02
  - DivergentSeries           -> _01
  - Checkmark                 -> _01, _02

Each figure is a faithful port of the MATLAB source shown on the page
(functions, domains, degrees, axis limits, titles, colors, layout).

This is a standalone generator; it does not edit any existing generator or
src file.  Run:

    cd /home/mg6942/chebfunjax
    JAX_PLATFORMS=cpu .pixi/envs/default/bin/python \
        scripts/generate_examples_ph_approx2.py
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
from chebfunjax.plotting import chebfun_style, save_chebfun_figure
from chebfunjax.utils.minimax import minimax
from chebfunjax.utils.quadrature import legpts
from chebfunjax.utils.transforms import cheb2leg, legvals2legcoeffs

chebfun_style()

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
REF_ROOT = "/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/refs/docs/images"

# MATLAB colour constants used across these pages
GREEN = (0.0, 0.7, 0.0)       # green = [0 .7 0]
COEF_BLUE = (0.0, 0.0, 1.0)   # '.b'  pure blue
COEF_RED = (1.0, 0.0, 0.0)    # '.r'  pure red
ENTIRE_BLUE = (0.0, 0.45, 0.74)  # blue = [0 .45 .74]
CHK_RED = (0.9, 0.0, 0.0)     # red = [.9 0 0]
CHK_BLUE = (0.0, 0.0, 0.9)    # blue = [0 0 .9]
EPS = float(np.finfo(np.float64).eps)


def save(fig, cat, name):
    from PIL import Image
    ref = os.path.join(REF_ROOT, cat, name)
    size = Image.open(ref).size if os.path.exists(ref) else (600, 270)
    save_chebfun_figure(fig, os.path.join(OUT_ROOT, cat, name), size=size)
    plt.close(fig)
    print(f"  {cat}/{name} saved  size={size}")


# ---------------------------------------------------------------------------
# AliasingCoefficients  (Chebyshev coefficients, aliasing accuracy)
# ---------------------------------------------------------------------------
def _coeff_panel(ax, fc, pc, err_x, err, *, ms):
    """Reproduce the plotcoeffs(f)/plotcoeffs(p)/plot(err) triple-scatter."""
    ax.semilogy(np.arange(len(fc)), np.abs(fc), ".", color=GREEN,
                markersize=ms, label="f")
    ax.semilogy(np.arange(len(pc)), np.abs(pc), ".", color=COEF_BLUE,
                markersize=ms, label="p")
    ax.semilogy(err_x, err, ".", color=COEF_RED, markersize=ms, label="f-p")
    ax.set_title("Chebyshev coefficients")


def aliasingcoefficients():
    # ---- Figure 1: analytic function ----
    fori = lambda x: jnp.log(jnp.sin(10 * x) + 2)
    f = cj.chebfun(fori)
    k = round(len(f) / 3)
    p = cj.chebfun(fori, n=k)
    fc = np.asarray(f.coeffs)
    pc = np.asarray(p.coeffs)
    err = np.abs(pc - fc[:len(pc)]) + EPS

    fig, ax = plt.subplots()
    _coeff_panel(ax, fc, pc, np.arange(len(pc)), err, ms=4)
    ax.set_xlim(0, len(f))
    ax.set_ylim(1e-20, 2.0)
    ax.set_yticks([1e-20, 1e-15, 1e-10, 1e-5, 1e0])
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=12)
    ax.set_position([0.128, 0.152, 0.777, 0.767])
    save(fig, "approx", "AliasingCoefficients_01.png")

    # ---- Figure 2: non-analytic function ----
    fori = lambda x: jnp.abs((x - 0.5) ** 3)
    f = cj.chebfun(fori)
    k = round(len(f) / 6)
    p = cj.chebfun(fori, n=k)
    fc = np.asarray(f.coeffs)
    pc = np.asarray(p.coeffs)
    err = np.abs(pc - fc[:len(pc)]) + EPS

    fig, ax = plt.subplots()
    _coeff_panel(ax, fc, pc, np.arange(len(pc)), err, ms=4)
    ax.set_xlim(0, len(f) / 2)
    ax.set_ylim(1e-15, 1e10)
    ax.set_yticks([1e-10, 1e0, 1e10])
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=12)
    ax.set_position([0.128, 0.152, 0.777, 0.767])
    save(fig, "approx", "AliasingCoefficients_02.png")


# ---------------------------------------------------------------------------
# AliasingCoefficientsLeg  (Legendre coefficients, aliasing accuracy)
# ---------------------------------------------------------------------------
def _leg_panel(ax, fc, pc, err_x, err, *, ms):
    ax.semilogy(np.arange(len(fc)), np.abs(fc), ".", color=GREEN,
                markersize=ms, label="f")
    ax.semilogy(np.arange(len(pc)), np.abs(pc), ".", color=COEF_BLUE,
                markersize=ms, label="p")
    ax.semilogy(err_x, err, ".", color=COEF_RED, markersize=ms, label="f-p")


def aliasingcoefficientsleg():
    # ---- Figure 1: analytic function ----
    fori = lambda x: jnp.log(jnp.sin(10 * x) + 2)
    f = cj.chebfun(fori)
    fc = np.asarray(cheb2leg(np.asarray(f.coeffs)))
    k = round(len(f) / 3)
    s = np.asarray(legpts(k)[0])
    pc = np.asarray(legvals2legcoeffs(np.asarray(fori(jnp.asarray(s)))))
    err = np.abs(pc - fc[:len(pc)]) + EPS

    fig, ax = plt.subplots()
    _leg_panel(ax, fc, pc, np.arange(len(pc)), err, ms=4)
    ax.set_xlim(0, len(fc))
    ax.set_ylim(1e-20, 1e10)
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=12)
    ax.set_position([0.128, 0.152, 0.777, 0.767])
    save(fig, "approx", "AliasingCoefficientsLeg_01.png")

    # ---- Figure 2: non-analytic function ----
    fori = lambda x: jnp.abs((x - 0.5) ** 3)
    f = cj.chebfun(fori)
    fc = np.asarray(cheb2leg(np.asarray(f.coeffs)))
    k = round(len(f) / 5)
    s = np.asarray(legpts(k)[0])
    pc = np.asarray(legvals2legcoeffs(np.asarray(fori(jnp.asarray(s)))))
    err = np.abs(pc[:k] - fc[:k]) + EPS

    fig, ax = plt.subplots()
    _leg_panel(ax, fc, pc, np.arange(k), err, ms=4)
    ax.set_xlim(0, len(fc))
    ax.set_ylim(1e-15, 1e5)
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=12)
    ax.set_position([0.128, 0.152, 0.777, 0.767])
    save(fig, "approx", "AliasingCoefficientsLeg_02.png")


# ---------------------------------------------------------------------------
# Entire  (Bernstein ellipses + convergence-estimate curves)
# ---------------------------------------------------------------------------
def entire():
    # ---- Figure 1: Bernstein r-ellipses ----
    rr = 1 + np.arange(1, 11) / 10.0
    t = np.linspace(0.0, 2.0 * np.pi, 2000)
    circ = np.exp(1j * t)
    fig, ax = plt.subplots()
    for rho in rr:
        z = (rho * circ + (rho * circ) ** (-1)) / 2.0
        ax.plot(z.real, z.imag, color=ENTIRE_BLUE, linewidth=1.0)
    # MATLAB `axis equal` yields a wide, short box; these explicit limits
    # reproduce that render's data-to-pixel scale on the fixed 600x270 canvas.
    ax.set_xlim(-1.585, 1.585)
    ax.set_ylim(-0.83, 0.83)
    ax.set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    ax.grid(False)
    save(fig, "approx", "Entire_01.png")

    # ---- Figure 2: convergence estimate curves ----
    ee = EPS
    NN = np.arange(10, 1011, 100)  # 10:100:1010
    fig, ax = plt.subplots()
    pp_grid = np.linspace(1.01, 10.0, 800)
    for N in NN:
        P = lambda p, N=N: (np.log(2.0 / ee) - np.log(p - 1.0)
                            + N * np.pi / 2.0 * (p - 1.0 / p)) / np.log(p)
        PP = cj.chebfun(lambda p, N=N: (jnp.log(2.0 / ee) - jnp.log(p - 1.0)
                        + N * jnp.pi / 2.0 * (p - 1.0 / p)) / jnp.log(p),
                        domain=(1.01, 10.0))
        ax.plot(pp_grid, np.asarray(PP(jnp.asarray(pp_grid))),
                color=ENTIRE_BLUE, linewidth=1.0)
        x_min, f_min = PP.min()
        ax.plot([x_min], [f_min], ".", color="r", markersize=5)
    labels = [10, 110, 210, 310, 410, 510]
    ys = [200, 800, 1450, 2100, 2700, 3350]
    for lab, y in zip(labels, ys):
        ax.text(8.02, y, "N = %3i" % lab, fontsize=11)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylim(0, 3.5e3)
    ax.set_xlim(1.0, 10.0)
    ax.grid(True)
    # Match MATLAB's axes rectangle on the 600x270 canvas (matplotlib's
    # default box is smaller/lower, shifting every curve ~10px).
    ax.set_position([0.130, 0.137, 0.775, 0.789])
    save(fig, "approx", "Entire_02.png")


# ---------------------------------------------------------------------------
# EntireBound  (Chebyshev error vs Bernstein-ellipse bounds)
# ---------------------------------------------------------------------------
def _entirebound_panel(ax, ff_j, title, rhos, Mfun, *, ms):
    fexact = cj.chebfun(ff_j)
    nmax = len(fexact) - 2
    nvec = np.arange(0, nmax + 1)
    errvec = []
    for n in nvec:
        fn = cj.chebfun(ff_j, n=n + 1)
        errvec.append(float((fn - fexact).norm(jnp.inf)))
    errvec = np.array(errvec)
    ax.semilogy(nvec, errvec, ".", markersize=ms, color=cj.plotting.CHEBFUN_BLUE)
    ax.set_xlabel("degree n")
    ax.set_ylabel("error")
    ax.set_title(title)
    for rho in rhos:
        M = Mfun(rho)
        bound = 4 * M * rho ** (-nvec.astype(float)) / (rho - 1)
        ax.semilogy(nvec, bound, "-k", linewidth=0.6)
        rlab = ("%d" % rho) if float(rho).is_integer() else ("%g" % rho)
        ax.text(1.01 * nmax, bound[-1], r"$\rho$=" + rlab, fontsize=9)
    ax.set_xlim(0, nmax)
    ax.set_ylim(1e-16, 1e3)
    ax.set_yticks([1e-10, 1e0])
    ax.grid(True)
    ax.set_position([0.13, 0.152, 0.775, 0.763])
    return nmax


def entirebound():
    # ---- Figure 1: exp(x) ----
    fig, ax = plt.subplots()
    _entirebound_panel(ax, lambda x: jnp.exp(x), "exp(x)",
                       [2, 4, 8, 16, 32],
                       lambda rho: np.exp((rho + 1.0 / rho) / 2.0), ms=4)
    save(fig, "approx", "EntireBound_01.png")

    # ---- Figure 2: cos(100x) ----
    fig, ax = plt.subplots()
    _entirebound_panel(ax, lambda x: jnp.cos(100 * x), "cos(100x)",
                       [1.5, 2, 3, 3.5],
                       lambda rho: np.cosh(100 * (rho - 1.0 / rho) / 2.0),
                       ms=4)
    save(fig, "approx", "EntireBound_02.png")


# ---------------------------------------------------------------------------
# DivergentSeries  (integral f as a function of parameter x)
# ---------------------------------------------------------------------------
def divergentseries():
    def ff_scalar(xv):
        g = cj.chebfun(lambda t, xv=float(xv): jnp.exp(-t) / (1.0 + xv * t),
                       domain=(0.0, float("inf")))
        return float(g.sum())

    def ff_vec(xarr):
        xarr = np.atleast_1d(np.asarray(xarr, dtype=float))
        return np.array([ff_scalar(xv) for xv in xarr])

    f = cj.chebfun(ff_vec, domain=(0.0, 5.0))
    xx = np.linspace(0.0, 5.0, 400)
    yy = np.asarray(f(jnp.asarray(xx)))

    fig, ax = plt.subplots()
    ax.plot(xx, yy, color=cj.plotting.CHEBFUN_BLUE, linewidth=1.2)
    ax.set_title("The integral f as a function of parameter x")
    ax.set_xlim(0, 5)
    ax.grid(False)
    save(fig, "approx", "DivergentSeries_01.png")


# ---------------------------------------------------------------------------
# Checkmark  (best-approximation error E_n(alpha) of |x - alpha|)
# ---------------------------------------------------------------------------
def _checkmark_En(alpha, n):
    """Minimax error of degree-n approximation to |x-alpha| on [-1,1].

    E_n is even in alpha; the MATLAB code builds it on [0,1] then mirrors.
    """
    a = abs(float(alpha))
    if a >= 1.0:
        # |x-1| = 1-x is a polynomial on [-1,1]; exact for n>=1.
        return 0.0
    # NOTE: do NOT pass breakpoints=[a]. Constraining the Remez grid to the
    # kink makes cj's minimax return a suboptimal (piecewise-linear/triangle)
    # error; approximating |x-a| directly by a single degree-n polynomial
    # reproduces the true best-approximation error (a smooth dome for n=1).
    try:
        r = minimax(lambda x, a=a: np.abs(x - a), n, domain=(-1.0, 1.0))
        return float(r.err)
    except Exception:  # noqa: BLE001  (occasional Remez non-convergence)
        return float("nan")


def _checkmark_curve(alpha_grid, n):
    return np.array([_checkmark_En(a, n) for a in alpha_grid])


_CHK_POS = [0.13, 0.152, 0.775, 0.763]  # MATLAB axes rectangle on 600x270


def checkmark():
    alpha = np.linspace(-1.0, 1.0, 401)
    # E[n] = minimax error curve E_n(alpha), computed once for n=1..7.
    E = {n: _checkmark_curve(alpha, n) for n in range(1, 8)}

    # ---- Figure 1: n = 2 and 3 ----
    # ColorOrder = [red; blue]; plot(E(:,2:3)) -> n=2 red, n=3 blue
    fig, ax = plt.subplots()
    ax.plot(alpha, E[2], color=CHK_RED, linewidth=1.6)
    ax.plot(alpha, E[3], color=CHK_BLUE, linewidth=1.6)
    ax.grid(True)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$E_n(\alpha)$")
    ax.set_title("n = 2 and 3")
    ax.set_xlim(-1, 1)
    ax.set_position(_CHK_POS)
    save(fig, "approx", "Checkmark_01.png")

    # ---- Figure 2: n = 1,...,7 ----
    # ColorOrder = [blue; red]; plot(E) cycles blue,red,blue,red,...
    colors = [CHK_BLUE, CHK_RED]
    fig, ax = plt.subplots()
    for n in range(1, 8):
        ax.plot(alpha, E[n], color=colors[(n - 1) % 2], linewidth=1.6)
    ax.grid(True)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$E_n(\alpha)$")
    ax.set_title("n = 1,2,...,7")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 0.5)
    ax.set_position(_CHK_POS)
    save(fig, "approx", "Checkmark_02.png")


PAGES = {
    "AliasingCoefficientsLeg": aliasingcoefficientsleg,
    "AliasingCoefficients": aliasingcoefficients,
    "EntireBound": entirebound,
    "Entire": entire,
    "DivergentSeries": divergentseries,
    "Checkmark": checkmark,
}

if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, fn in PAGES.items():
        if flt.lower() in name.lower():
            print(f"[{name}]")
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
