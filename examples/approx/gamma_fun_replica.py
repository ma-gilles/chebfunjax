"""The gamma function and its critical points.

Faithful replica of approx/GammaFun.m by Nick Trefethen (September
2010): the gamma function on [-4,4] via blowup+splitting and via an
'exps' construction, related functions 1/Gamma, sqrt|Gamma|, local
extrema, and the integrals NaN / Inf / 14.0433...

Original: https://www.chebfun.org/examples/approx/GammaFun.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as sp_gamma

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def _plot_gam(fun_list, labels, fname, title, dots=None):
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for g, lab in zip(fun_list, labels):
        for lo, hi in zip(list(g.domain.breakpoints)[:-1],
                          list(g.domain.breakpoints)[1:]):
            xs = np.linspace(float(lo) + 1e-6, float(hi) - 1e-6, 500)
            ys = np.asarray(g(jnp.asarray(xs)))
            ax.plot(xs, ys, lw=1.3,
                    color=f"C{labels.index(lab)}",
                    label=lab if lo == list(g.domain.breakpoints)[0]
                    else None)
    if dots is not None:
        for xs_d, ys_d in dots:
            ax.plot(xs_d, ys_d, '.k', ms=10)
    ax.set_ylim(-6, 6)
    ax.grid(True)
    ax.set_title(title, fontsize=12)
    if len(labels) > 1:
        ax.legend(loc="lower right")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    gam_op = lambda x: jnp.asarray(sp_gamma(np.asarray(x)))  # noqa: E731

    # Automatic pole detection with blowup + splitting
    gam = cj.chebfun(gam_op, domain=[-4.0, 4.0], blowup=True,
                     splitting=True)
    print("gam =")
    print(repr(gam))
    _plot_gam([gam], [r"$\Gamma(x)$"], "GammaFun_repl_01.png",
              "Gamma function")

    # Same, with breakpoints and exponents given explicitly
    gam = cj.chebfun(gam_op, domain=[-4.0, -3.0, -2.0, -1.0, 0.0, 4.0],
                     exps=[-1, -1, -1, -1, -1, -1, -1, -1, -1, 0])
    print("gam =")
    print(repr(gam))
    _plot_gam([gam], [r"$\Gamma(x)$"], "GammaFun_repl_02.png",
              "Gamma function again")

    # Related functions
    gam_i = 1.0 / gam
    absgam = gam.abs()
    sqrtgam = absgam.sqrt().real()
    _plot_gam([gam, gam_i, sqrtgam],
              [r"$\Gamma(x)$", r"$1/\Gamma(x)$",
               r"$\sqrt{|\Gamma(x)|}$"],
              "GammaFun_repl_03.png", "Various related functions")

    # Critical points
    dots = []
    for g in (gam, gam_i, sqrtgam):
        rr, _vv = g.minandmax('local')
        rr = np.atleast_1d(np.asarray(rr, dtype=np.float64))
        rr = rr[np.isfinite(rr)]
        vv = np.asarray(g(jnp.asarray(rr)))
        keep = np.abs(vv) < 6
        dots.append((rr[keep], vv[keep]))
    _plot_gam([gam, gam_i, sqrtgam],
              [r"$\Gamma(x)$", r"$1/\Gamma(x)$",
               r"$\sqrt{|\Gamma(x)|}$"],
              "GammaFun_repl_04.png", "Critical points", dots=dots)

    # Integrals
    for g in (gam, absgam, sqrtgam):
        v = float(g.sum())
        print("ans =")
        if np.isnan(v):
            print("   NaN")
        elif np.isinf(v):
            print("   Inf")
        else:
            print(f"  {v:.15f}")


if __name__ == "__main__":
    run()
