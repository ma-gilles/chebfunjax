"""The gamma function and its poles.

Faithful port of approx/GammaFun.m by Nick Hale, December 2009 (revised
June 2019 by Nick Trefethen).  A chebfun with 'exps' [-1 ...] represents
gamma(x) on [-4, 4] with its simple poles at the non-positive integers;
reciprocal, absolute value, square root, local extrema, and integrals
all operate on the singular representation.

Original: https://www.chebfun.org/examples/approx/GammaFun.html
Copyright 2019 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): the three published integrals reproduce
exactly -- sum(gam) = NaN, sum(|gam|) = Inf, and sum(sqrt(|gam|)) =
14.043323986892393 (all digits).  The display tables' interval
endpoints, +/-Inf endpoint values, and endpoint exponents [-1 -1]
reproduce; the per-piece LENGTHS (MATLAB 20/25/24/20/35, ours
19/19/19/18/36) are adaptive-construction scheme values.  The page's
first construction uses 'blowup'+'splitting' automatic pole detection,
which chebfunjax does not yet reproduce (detection gap, ledgered); both
printed displays here use the 'exps' construction the page itself
recommends ("always a better idea").
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as scipy_gamma

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.fun.singfun import Singfun
from chebfunjax.plotting import chebfun_style

chebfun_style()

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def _fmt_end(v):
    if np.isposinf(v):
        return "Inf"
    if np.isneginf(v):
        return "-Inf"
    return f"{v:.2g}"


def _display(f, name):
    """MATLAB-style singular-chebfun display table."""
    n = len(f.funs)
    print(f"{name} =")
    print(f"   chebfun column ({n} smooth pieces)")
    print("       interval       length     endpoint values"
          "   endpoint exponents")
    total = 0
    for p in f.funs:
        a, b = p.interval
        lval, rval = p.endpoint_values
        exps = getattr(p.tech, "exponents", (0.0, 0.0))
        total += p.n
        print(f"[{a:8g},{b:8g}]  {p.n:7d}  {_fmt_end(lval):>8}  "
              f"{_fmt_end(rval):>7}         [{exps[0]:g}  {exps[1]:g}]  ")
    print(f"vertical scale = Inf    Total length = {total}")


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    gam = cj.chebfun(
        lambda x: jnp.asarray(scipy_gamma(np.asarray(x))),
        domain=[-4, -3, -2, -1, 0, 4],
        exps=[-1, -1, -1, -1, -1, 0],
    )
    # The page's first construction ('blowup','on','splitting','on')
    # auto-detects the same poles; chebfunjax lacks that detection, so
    # both displays show the 'exps' construction.
    _display(gam, "gam")
    _display(gam, "gam")

    gam_i = 1.0 / gam
    absgam = abs(gam)
    sqrtgam = (absgam ** 0.5).real()

    # Critical points via minandmax(..., 'local').
    _, r = gam.minandmax("local")
    _, ri = gam_i.minandmax("local")
    _, rs = sqrtgam.minandmax("local")

    xx = np.linspace(-4, 4, 3000)
    with np.errstate(all="ignore"):
        gxx = scipy_gamma(xx)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(xx, gxx, "b", lw=1.4, label=r"$\Gamma(x)$")
    ax.plot(xx, 1.0 / gxx, "r", lw=1.4, label=r"$1/\Gamma(x)$")
    ax.plot(xx, np.sqrt(np.abs(gxx)), "g", lw=1.4,
            label=r"$\sqrt{|\Gamma(x)|}$")
    for pts, f in ((r, gam), (ri, gam_i), (rs, sqrtgam)):
        pts = np.asarray(pts, dtype=float)
        pts = pts[np.isfinite(pts)]
        vals = np.asarray(f(jnp.asarray(pts)), dtype=float)
        ok = np.abs(vals) < 8
        ax.plot(pts[ok], vals[ok], ".k", ms=8)
    ax.set_ylim(-8, 8)
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Gamma function, related functions, critical points")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, "GammaFun.png"), dpi=150)
    plt.close(fig)

    def _print_ans(v):
        print("ans =")
        if np.isnan(v):
            print("   NaN")
        elif np.isposinf(v):
            print("   Inf")
        else:
            print(f"  {v:.15f}")

    _print_ans(float(gam.sum()))
    _print_ans(float(absgam.sum()))
    _print_ans(float(sqrtgam.sum()))

    assert isinstance(gam.funs[0].tech, Singfun)
    return True


if __name__ == '__main__':
    run()
