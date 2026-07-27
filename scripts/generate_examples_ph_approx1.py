"""Generate placeholder example-plot figures for docs/examples/approx pages.

Pages ported (chebfun.org/examples/approx/<Name>.html):
  * AbsoluteValue  -> AbsoluteValue_01.png .. _05.png
  * BestApprox     -> BestApprox_01.png (poly panel only; rational panels
                       BLOCKED: chebfunjax has no rational Remez / remez(f,m,n))
  * OrthPolys      -> OrthPolys_01.png, OrthPolys_02.png

Standalone generator -- does NOT edit any existing generator or src file.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gc

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import chebfunjax as cj
from chebfunjax.plotting import CHEBFUN_BLUE, chebfun_style, save_chebfun_figure

chebfun_style()

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
REF_ROOT = "/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/refs/docs/images"


def save(fig, cat, name):
    from PIL import Image

    ref = os.path.join(REF_ROOT, cat, name)
    size = Image.open(ref).size if os.path.exists(ref) else (600, 270)
    save_chebfun_figure(fig, os.path.join(OUT_ROOT, cat, name), size=size)
    plt.close(fig)
    print(f"  {cat}/{name} saved")


def _clear():
    plt.close("all")
    jax.clear_caches()
    gc.collect()


# --------------------------------------------------------------------------
# AbsoluteValue
# --------------------------------------------------------------------------
LW = 1.6
FS = 12


def _plot_iter_panel(states, use_len_short):
    """3x2 grid of the rational-iteration approximants r to |x|.

    states: list of 6 dicts with keys x (numpy), y (numpy), err, length.
    """
    fig = plt.figure()
    # Axes positions measured from the chebfun.org reference render (600x270):
    # two columns, three rows of short panels with titles in the row gaps.
    col_lefts = [0.1292, 0.5692]
    row_bottoms = [0.7204, 0.4204, 0.1204]
    width, height = 0.335, 0.1852
    for i, st in enumerate(states):
        r_, c_ = divmod(i, 2)
        ax = fig.add_axes([col_lefts[c_], row_bottoms[r_], width, height])
        ax.plot(st["x"], st["y"], color=CHEBFUN_BLUE, linewidth=1.4)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "0.5", "1"], fontsize=7)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(["-1", "-0.5", "0", "0.5", "1"], fontsize=7)
        ax.tick_params(length=2, pad=1.5)
        ax.grid(True)
        if use_len_short:
            s = "error=%4.1e   len=%d" % (st["err"], st["length"])
        else:
            s = "error=%4.1e   length = %d" % (st["err"], st["length"])
        ax.set_title(s, fontsize=7.5, fontweight="bold", pad=2.5)
    return fig


def _plot_error_semilogy(xx, yy):
    fig, ax = plt.subplots()
    ax.semilogy(xx, yy, color=CHEBFUN_BLUE, linewidth=LW)
    ax.set_xlim(-1, 1)
    ax.set_ylim(1e-18, 10)
    ax.set_yticks([1e-15, 1e-10, 1e-5, 1e0])
    xt = np.round(np.arange(-1.0, 1.01, 0.2), 10)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{v:g}" for v in xt], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True)
    ax.set_xlabel("x", fontsize=9)
    ax.set_title("Error", fontsize=9, fontweight="bold")
    # MATLAB default axes position [0.13 0.11 0.775 0.815], as measured on
    # the chebfun.org reference renders (box x 77.5-542.5, y 19.5-239.5).
    fig.subplots_adjust(left=0.1292, right=0.9042, top=0.9278, bottom=0.113)
    return fig


def _dense(f, dom):
    """Evaluate chebfun f on a dense grid over its domain for plotting."""
    a, b = dom[0], dom[-1]
    xx = np.linspace(a, b, 3000)
    yy = np.asarray(f(jnp.asarray(xx)))
    return xx, np.asarray(yy)


def _iterate(domain, n_panels, r=None, x=None):
    """Run the Babylonian iteration r <- (r^2 + x^2)/(2 r), capturing panels.

    The sup-error norm(r - |x|, inf) is computed on a dense grid: building a
    chebfun for |x| on the breakpoint-free domain hits the 65537-point unhappy
    ceiling and makes .norm(inf) intractable, while the dense max reproduces
    the MATLAB title values to the 2 significant digits displayed.

    Returns (states, r, x) so the caller can continue iterating.
    """
    if x is None:
        x = cj.chebfun(lambda t: t, domain=domain)
        r = cj.chebfun(lambda t: jnp.ones_like(t), domain=domain)
    xe = np.linspace(domain[0], domain[-1], 100001)
    fe = np.abs(xe)
    states = []
    for _ in range(n_panels):
        err = float(np.max(np.abs(np.asarray(r(jnp.asarray(xe))) - fe)))
        xx, yy = _dense(r, domain)
        states.append({"x": xx, "y": yy, "err": err, "length": len(r)})
        r = (r ** 2 + x ** 2) / (2 * r)
    return states, r, x


def absolutevalue():
    cat = "approx"

    # Fig 1: no breakpoint, 6 panels, 'len=' short title
    states, _, _ = _iterate((-1.0, 1.0), 6)
    fig = _plot_iter_panel(states, use_len_short=True)
    save(fig, cat, "AbsoluteValue_01.png")
    _clear()

    # Fig 2: breakpoint domain (-1,0,1), 6 panels, 'length = ' title
    states, r, x = _iterate((-1.0, 0.0, 1.0), 6)
    fig = _plot_iter_panel(states, use_len_short=False)
    save(fig, cat, "AbsoluteValue_02.png")
    _clear()

    # Fig 3: semilogy of |r - |x|| after fig-2 iterations.
    # Evaluate r densely and take |r - |x|| in numpy: constructing a chebfun
    # of the non-smooth error near machine precision is not needed for plotting.
    xx = np.linspace(-1.0, 1.0, 4001)
    yy = np.abs(np.asarray(r(jnp.asarray(xx))) - np.abs(xx))
    fig = _plot_error_semilogy(xx, yy)
    save(fig, cat, "AbsoluteValue_03.png")
    _clear()

    # Fig 4: continue iterating (6 more panels) on breakpoint domain
    states, r, x = _iterate((-1.0, 0.0, 1.0), 6, r=r, x=x)
    fig = _plot_iter_panel(states, use_len_short=False)
    save(fig, cat, "AbsoluteValue_04.png")
    _clear()

    # Fig 5: final semilogy error. Denser sampling than fig 3: the reference
    # render shows a solid rounding-noise band (the final r has length ~5000),
    # which needs many samples to reproduce.
    xx = np.linspace(-1.0, 1.0, 50001)
    yy = np.abs(np.asarray(r(jnp.asarray(xx))) - np.abs(xx))
    fig = _plot_error_semilogy(xx, yy)
    save(fig, cat, "AbsoluteValue_05.png")
    _clear()


# --------------------------------------------------------------------------
# BestApprox  (poly panel only; rational panels BLOCKED)
# --------------------------------------------------------------------------
def bestapprox():
    cat = "approx"
    from chebfunjax.utils.minimax import minimax

    # f = |x - 0.5| on [-1,1], kink at 0.5
    f_np = lambda x: np.abs(np.asarray(x) - 0.5)
    # NB: passing breakpoints=[0.5] makes the Remez exchange stall for this
    # function (returns err=5.3e-3, actual sup-error 4.2e-2 -- not
    # equioscillating). The plain call converges to the true degree-16
    # minimax, err=1.61e-2, matching MATLAB remez(f,16) err=1.64e-2.
    r = minimax(f_np, 16, domain=(-1.0, 1.0))
    err = float(r.err)
    coeffs = np.asarray(r.coeffs)

    xx = np.linspace(-1.0, 1.0, 4000)
    # evaluate the minimax polynomial on r.domain (mapped to [-1,1])
    a, b = float(r.domain[0]), float(r.domain[-1])
    xh = 2 * (xx - a) / (b - a) - 1
    if coeffs.ndim == 1:
        p = np.polynomial.chebyshev.chebval(xh, coeffs)
    else:
        # piecewise coeffs (breakpoint): evaluate per segment
        p = _eval_piecewise(xx, r)
    ferr = f_np(xx) - p

    fig, ax = plt.subplots()
    ax.plot(xx, ferr, color=CHEBFUN_BLUE, linewidth=1.6)
    ax.plot([-1, 1], [err, err], "--k", linewidth=1.0)
    ax.plot([-1, 1], [-err, -err], "--k", linewidth=1.0)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.03, 0.03)
    ax.grid(False)
    ax.set_title("Degree 16 polynomial error curve", fontsize=14)
    save(fig, cat, "BestApprox_01.png")
    _clear()
    # Figs 02-05 are rational (8,8)/(16,16) remez -> BLOCKED (not implemented).


def _eval_piecewise(xx, r):
    """Evaluate a piecewise-minimax result on grid xx."""
    dom = np.asarray(r.domain)
    coeffs = r.coeffs
    out = np.empty_like(xx)
    for i in range(len(dom) - 1):
        a, b = dom[i], dom[i + 1]
        mask = (xx >= a) & (xx <= b)
        xh = 2 * (xx[mask] - a) / (b - a) - 1
        ci = np.asarray(coeffs[i]) if isinstance(coeffs, (list, tuple)) or (
            hasattr(coeffs, "ndim") and coeffs.ndim > 1) else np.asarray(coeffs)
        out[mask] = np.polynomial.chebyshev.chebval(xh, ci)
    return out


# --------------------------------------------------------------------------
# OrthPolys
# --------------------------------------------------------------------------
def orthpolys():
    cat = "approx"
    dom = (-1.0, 1.0)
    w = cj.chebfun(lambda t: jnp.exp(jnp.pi * t), domain=dom)
    x = cj.chebfun(lambda t: t, domain=dom)
    N = 5

    sum_w = float(w.sum())
    P = [cj.chebfun(lambda t: jnp.ones_like(t) / jnp.sqrt(sum_w), domain=dom)]
    for k in range(1, N + 1):
        xk = x * P[k - 1]
        pk1 = xk
        for j in range(k):
            C = float((w * xk * P[j]).sum())
            pk1 = pk1 - C * P[j]
        nrm = float((w * pk1 ** 2).sum())
        pk1 = pk1 * (1.0 / np.sqrt(nrm))
        P.append(pk1)

    # Fig 1: all orthonormal polys
    xx = np.linspace(-1.0, 1.0, 2000)
    fig, ax = plt.subplots()
    for p in P:
        yy = np.asarray(p(jnp.asarray(xx)))
        ax.plot(xx, yy, linewidth=1.6)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-10, 10)
    ax.grid(False)
    ax.set_title("Orthogonal polynomials on [-1,1] wrt w = exp(pi*x)",
                 fontsize=12)
    save(fig, cat, "OrthPolys_01.png")
    _clear()

    # Fig 2: least-squares approximation to |x|
    f = cj.chebfun(lambda t: jnp.abs(t), domain=dom)
    alpha = [float((w * P[k] * f).sum()) for k in range(N + 1)]
    P_star = P[0] * alpha[0]
    for k in range(1, N + 1):
        P_star = P_star + P[k] * alpha[k]

    xx = np.linspace(-1.0, 1.0, 2000)
    yf = np.abs(xx)
    yps = np.asarray(P_star(jnp.asarray(xx)))
    fig, ax = plt.subplots()
    ax.plot(xx, yf, "b", linewidth=1.6)
    ax.plot(xx, yps, "--r", linewidth=1.6)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.05, 1.2)
    ax.grid(False)
    ax.set_title("Least-squares approximation to |x| wrt w = exp(pi*x)",
                 fontsize=12)
    save(fig, cat, "OrthPolys_02.png")
    _clear()


PAGES = {
    "AbsoluteValue": absolutevalue,
    "BestApprox": bestapprox,
    "OrthPolys": orthpolys,
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
