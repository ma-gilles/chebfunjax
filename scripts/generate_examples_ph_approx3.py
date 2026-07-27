"""Port of chebfun.org approx/ example plots (placeholder batch ph_approx3).

Pages ported here (chebfun.org/examples/approx/<Name>.html):
  Halphen              -> _01, _02   (semilogy convergence + Halphen constant)
  ScalingAndSquaring   -> _01, _02   (padeapprox of exp, complex-plane contourf)
  WeierstrassFunction  -> _01, _02   (sum of scaled cosines; nowhere-diff fn)
  WigglyApprox         -> _01, _02   (sum(chebpoly(m:n)); best poly via minimax)

BLOCKED (reported, PNG left untouched):
  RationalAbsx, Rationalxn -- both are equioscillating RATIONAL minimax error
  curves.  chebfunjax minimax(...,rational=True) raises NotImplementedError and
  there is no remez(f,m,n).  AAA is interpolatory (error has zeros at support
  points, does not equioscillate) so it cannot reproduce these figures.

Run:
  cd /home/mg6942/chebfunjax && JAX_PLATFORMS=cpu \
    .pixi/envs/default/bin/python scripts/generate_examples_ph_approx3.py
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.plotting import CHEBFUN_BLUE, chebfun_style, save_chebfun_figure
from chebfunjax.utils.minimax import minimax
from chebfunjax.utils.ratapprox import padeapprox

# MATLAB "parula" colormap (7-anchor); grayscale luminance matches the
# chebfun.org contourf renders better than the repo's 256-node table.
PARULA7 = LinearSegmentedColormap.from_list("parula7", [
    (0.2422, 0.1504, 0.6603),
    (0.2780, 0.3556, 0.9777),
    (0.1085, 0.5468, 0.8654),
    (0.0468, 0.6717, 0.7581),
    (0.3301, 0.7263, 0.5567),
    (0.7626, 0.7361, 0.3346),
    (0.9769, 0.9839, 0.0805),
])

_MLAB_FMT = FuncFormatter(lambda v, _: f"{v:g}")


def _mlab_ticks(ax):
    """Format tick labels like MATLAB (drop trailing '.0'/'.00')."""
    ax.xaxis.set_major_formatter(_MLAB_FMT)
    ax.yaxis.set_major_formatter(_MLAB_FMT)


chebfun_style()

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
REF_ROOT = "/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/refs/docs/images"

def save(fig, cat, name):
    from PIL import Image
    ref = os.path.join(REF_ROOT, cat, name)
    size = Image.open(ref).size if os.path.exists(ref) else (600, 270)
    save_chebfun_figure(fig, os.path.join(OUT_ROOT, cat, name), size=size)
    plt.close(fig)
    print(f"  {cat}/{name} saved  (size={size})")


# ----------------------------------------------------------------------------
# Halphen -- semilogy convergence + Halphen's constant
# ----------------------------------------------------------------------------
def _halphen_f(s):
    """f(s) = sum_{k>=1} k*s^k/(1-(-1)^k s^k)  (converges fast on [1/12,1/6])."""
    total = jnp.zeros_like(jnp.asarray(s, dtype=jnp.float64))
    for k in range(1, 26):
        sk = jnp.asarray(s) ** k
        total = total + k * sk / (1.0 - ((-1.0) ** k) * sk)
    return total


def halphen():
    # ---- compute Halphen's constant h via roots of f(s) - 1/8 ----
    fcheb = cj.chebfun(_halphen_f, domain=(1.0 / 12.0, 1.0 / 6.0))
    rts = np.asarray((fcheb - 0.125).roots())
    rts = rts[(rts >= 1.0 / 12.0) & (rts <= 1.0 / 6.0)]
    sstar = float(rts[0])
    h = 1.0 / sstar
    print(f"  Halphen constant h = {h:.13f}")

    # ---- Figure 1: semilogy model vs computed errors ----
    n = np.arange(0, 11)
    err = np.array([.5, .0668, 7.36e-3, 7.99e-4, 8.65e-5, 9.35e-6,
                    1.01e-6, 1.09e-7, 1.17e-8, 1.26e-9, 1.36e-10])
    model = 2.0 * h ** (-n - 0.5)
    fig, ax = plt.subplots()
    ax.semilogy(n, model, '-b', linewidth=1.4)
    ax.semilogy(n, err, '.k', markersize=14)
    ax.grid(True, which="both")
    ax.set_xlabel("n")
    ax.set_ylabel("error")
    ax.set_xlim(0, 10)
    ax.set_ylim(1e-10, 1e0)
    save(fig, "approx", "Halphen_01.png")

    # ---- Figure 2: 1/s vs f, with Halphen's constant marked ----
    ss = np.linspace(1.0 / 12.0, 1.0 / 6.0, 3000)
    fvals = np.asarray(_halphen_f(jnp.asarray(ss)))
    fig, ax = plt.subplots()
    ax.plot(1.0 / ss, fvals, color=CHEBFUN_BLUE, linewidth=1.5)
    ax.grid(True)
    ax.plot(h, 0.125, '.r', markersize=24)
    ax.text(h, 0.135, f"{h:16.13f}".strip(), fontsize=11)
    ax.set_title("Halphen's constant")
    ax.set_xlim(6, 12)
    ax.set_ylim(0.08, 0.22)
    save(fig, "approx", "Halphen_02.png")


# ----------------------------------------------------------------------------
# ScalingAndSquaring -- padeapprox of exp + complex-plane contourf
# ----------------------------------------------------------------------------
def scalingandsquaring():
    s = 2
    m = 8
    taylor = np.concatenate([[1.0], 1.0 / np.cumprod(np.arange(1, 51, dtype=float))])
    r, a, b, mu, nu, poles, res = padeapprox(taylor, m, m, 0.0)

    xgrid = np.linspace(-100, 100, 140)
    X, Y = np.meshgrid(xgrid, xgrid)
    Z = X + 1j * Y

    def rvec(zz):
        out = np.asarray(r(zz.ravel()), dtype=complex)
        return out.reshape(zz.shape)

    scaled = rvec(Z / (2 ** s)) ** (2 ** s)
    ez = np.exp(Z)
    eps = np.finfo(float).eps

    # ---- Figure 1: absolute error ----
    E1 = np.log10(np.abs(ez - scaled) + eps)
    lv1 = list(range(-16, 1, 2))
    fig, ax = plt.subplots()
    cf = ax.contourf(X, Y, E1, levels=lv1, cmap=PARULA7, extend="both")
    ax.contour(X, Y, E1, levels=lv1, colors="k", linewidths=0.8, linestyles="solid")
    fig.colorbar(cf, ax=ax, ticks=list(range(-15, 1, 5)))
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_yticks(list(range(-100, 101, 20)))
    save(fig, "approx", "ScalingAndSquaring_01.png")

    # ---- Figure 2: relative error ----
    E2 = np.log10(np.abs(ez - scaled) / np.abs(ez))
    lv2 = list(range(-16, 17, 2))
    fig, ax = plt.subplots()
    cf = ax.contourf(X, Y, E2, levels=lv2, cmap=PARULA7, extend="both")
    ax.contour(X, Y, E2, levels=lv2, colors="k", linewidths=0.8, linestyles="solid")
    fig.colorbar(cf, ax=ax, ticks=list(range(-16, 17, 4)))
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_yticks(list(range(-100, 101, 20)))
    save(fig, "approx", "ScalingAndSquaring_02.png")


# ----------------------------------------------------------------------------
# WeierstrassFunction -- sum_{k=0}^{8} 2^-k cos(pi/2 x 4^k)
# ----------------------------------------------------------------------------
def _weierstrass(x):
    x = np.asarray(x, dtype=float)
    total = np.zeros_like(x)
    for k in range(0, 9):
        total = total + 2.0 ** (-k) * np.cos(np.pi / 2.0 * x * 4.0 ** k)
    return total


def weierstrassfunction():
    # ---- Figure 1: full function ----
    xx = np.linspace(-1, 1, 300000)
    fig, ax = plt.subplots()
    ax.plot(xx, _weierstrass(xx), 'k', linewidth=0.5)
    ax.set_title("A pathological function of Weierstrass")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.5, 2)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    _mlab_ticks(ax)
    save(fig, "approx", "WeierstrassFunction_01.png")

    # ---- Figure 2: close-up on [0, 0.005] ----
    xx = np.linspace(0, 0.005, 8000)
    fig, ax = plt.subplots()
    ax.plot(xx, _weierstrass(xx), 'k', linewidth=1.5)
    ax.set_title("Close-up of Weierstrass approximant")
    ax.set_xlim(0, 0.005)
    ax.set_ylim(1.8, 2.0)
    save(fig, "approx", "WeierstrassFunction_02.png")


# ----------------------------------------------------------------------------
# WigglyApprox -- f = sum(chebpoly(m:n)); p = remez(f, m-1) via minimax
# ----------------------------------------------------------------------------
def _fmn(m, n):
    c = np.zeros(n + 1)
    c[m:n + 1] = 1.0
    return Chebfun.from_coeffs(jnp.asarray(c, dtype=jnp.float64))


def _wiggly_panel(m, n, name, closeup, ylims, cl_xticks, err_lw):
    fcheb = _fmn(m, n)
    res = minimax(fcheb, m - 1, domain=(-1.0, 1.0))
    pcoef = np.asarray(res.coeffs)

    def fval(x):
        return np.asarray(fcheb(jnp.asarray(x, dtype=jnp.float64)))

    def pval(x):
        return np.polynomial.chebyshev.chebval(np.asarray(x), pcoef)

    ca, cb = closeup
    yf, yfc, ye, yec = ylims
    xx_full = np.linspace(-1, 1, 20000)
    xx_cl = np.linspace(ca, cb, 4000)

    fig, axes = plt.subplots(2, 2)
    FS = 9

    ax = axes[0, 0]
    ax.plot(xx_full, fval(xx_full), color=CHEBFUN_BLUE, linewidth=1.0)
    ax.grid(True); ax.set_title(f"f({m},{n})", fontsize=FS)
    ax.set_xlim(-1, 1); ax.set_ylim(*yf)

    ax = axes[0, 1]
    ax.plot(xx_cl, fval(xx_cl), color=CHEBFUN_BLUE, linewidth=1.6)
    ax.grid(True); ax.set_title("closeup", fontsize=FS)
    ax.set_xlim(ca, cb); ax.set_ylim(*yfc)

    ax = axes[1, 0]
    ax.plot(xx_full, fval(xx_full) - pval(xx_full), 'r', linewidth=err_lw)
    ax.grid(True); ax.set_title("f - p", fontsize=FS)
    ax.set_xlim(-1, 1); ax.set_ylim(*ye)

    ax = axes[1, 1]
    ax.plot(xx_cl, fval(xx_cl) - pval(xx_cl), 'r', linewidth=1.6)
    ax.grid(True); ax.set_title("closeup", fontsize=FS)
    ax.set_xlim(ca, cb); ax.set_ylim(*yec)

    for a in (axes[0, 0], axes[1, 0]):
        a.set_xticks([-1, -0.5, 0, 0.5, 1])
    for a in (axes[0, 1], axes[1, 1]):
        a.set_xticks(cl_xticks)
    for a in axes.ravel():
        _mlab_ticks(a)
    save(fig, "approx", name)


def wigglyapprox():
    _wiggly_panel(30, 40, "WigglyApprox_01.png", (0.8, 1.0),
                  ((-20, 20), (-15, 15), (-10, 10), (-10, 10)),
                  [0.8, 0.85, 0.9, 0.95, 1.0], err_lw=1.2)
    _wiggly_panel(200, 220, "WigglyApprox_02.png", (0.995, 1.0),
                  ((-40, 40), (-40, 40), (-20, 20), (-20, 20)),
                  [0.996, 0.998, 1.0], err_lw=0.8)


PAGES = {
    "Halphen": halphen,
    "ScalingAndSquaring": scalingandsquaring,
    "WeierstrassFunction": weierstrassfunction,
    "WigglyApprox": wigglyapprox,
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
