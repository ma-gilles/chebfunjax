"""Eight shades of rational approximation.

Faithful replica of approx/EightShades.m by Nick Trefethen (May 2016):
a 4x4 taxonomy of approximation methods — {polynomial, trigonometric}
x {interpolation, projection, minimax, CF} and their rational
analogues — applied to a Gaussian bump.

Original: https://www.chebfun.org/examples/approx/EightShades.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.cfpade import cf, chebpade
from chebfunjax.utils.minimax import minimax
from chebfunjax.utils.ratapprox import ratinterp
from chebfunjax.utils.trigrational import trigremez

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

XS = np.linspace(-1, 1, 2500)
YL = (-0.5, 1.2)


def _panel(ax, fv, pv, label, color='r'):
    ax.plot(XS, fv, 'k', lw=1.2)
    if pv is not None:
        ax.plot(XS, pv, color, lw=1.2)
    ax.set_ylim(*YL)
    ax.text(-0.93, 0.9, label, fontsize=10)


def _na(ax, label):
    ax.text(-0.93, 0.9, label, fontsize=10)
    ax.text(-0.5, 0.2, "(not yet available)", fontsize=10)
    ax.set_xlim(-1, 1)
    ax.set_ylim(*YL)
    ax.set_xticks([])
    ax.set_yticks([])


def run():
    os.makedirs(_IMG, exist_ok=True)
    fop = lambda x: jnp.exp(-50 * (x - 0.1) ** 2)  # noqa: E731
    f = cj.chebfun(fop, trig=True)
    fv = np.asarray(f(jnp.asarray(XS)))
    m = 8

    # 1. Polynomial approximations
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.0))
    p1 = cj.chebfun(fop, n=m + 1)
    _panel(axes[0, 0], fv, np.asarray(p1(jnp.asarray(XS))),
           "interpolation")
    fcheb = cj.chebfun(fop)
    p2 = cj.chebfun(jnp.asarray(np.asarray(fcheb.coeffs)[:m + 1]),
                    coeffs=True)
    _panel(axes[0, 1], fv, np.asarray(p2(jnp.asarray(XS))), "projection")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimax(fop, m)
    p3 = cj.chebfun(jnp.asarray(res.coeffs), coeffs=True)
    _panel(axes[1, 0], fv, np.asarray(p3(jnp.asarray(XS))), "minimax")
    pcf, qcf, rcf, _s = cf(fcheb, m)
    p4v = np.asarray(rcf(jnp.asarray(XS)))
    _panel(axes[1, 1], fv, p4v, "CF")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "EightShades_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    cferr = float(np.max(np.abs(np.asarray(p3(jnp.asarray(XS))) - p4v)))
    print("CFerror =")
    print(f"     {cferr:.15e}")

    # 2. Trigonometric approximations
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.0))
    t1 = cj.chebfun(fop, n=m + 1, trig=True)
    _panel(axes[0, 0], fv, np.asarray(t1(jnp.asarray(XS))),
           "interpolation", color='b')
    c = np.asarray(f.coeffs)
    k0 = len(c) // 2
    keep = np.zeros_like(c)
    lo, hi = k0 - m // 2, k0 + m // 2 + 1
    keep[lo:hi] = c[lo:hi]
    t2 = cj.chebfun(jnp.asarray(keep), coeffs=True, trig=True)
    _panel(axes[0, 1], fv, np.real(np.asarray(t2(jnp.asarray(XS)))),
           "projection", color='b')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t3, terr = trigremez(f, m // 2)[:2]
    _panel(axes[1, 0], fv, np.asarray(t3(jnp.asarray(XS))),
           "minimax", color='b')
    _na(axes[1, 1], "CF")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "EightShades_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Rational approximations, type (3,3)
    mr = nr = 3
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.0))
    rh, *_ = ratinterp(lambda x: fcheb(x), mr, nr)
    with np.errstate(divide="ignore", invalid="ignore"):
        _panel(axes[0, 0], fv, np.asarray(rh(XS)), "interpolation")
    try:
        pp, qq, rr_ = chebpade(fcheb, mr, nr)[:3]
        _panel(axes[0, 1], fv, np.asarray(rr_(jnp.asarray(XS))),
               "projection")
    except Exception:
        _na(axes[0, 1], "projection")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rres = minimax(fop, mr, rational=True, denom=nr)
    _panel(axes[1, 0], fv, np.asarray(rres.r(XS)), "minimax")
    p4, q4, r4, _s4 = cf(fcheb, mr, nr)
    _panel(axes[1, 1], fv, np.asarray(r4(jnp.asarray(XS))), "CF")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "EightShades_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Trigonometric rational: all four not yet available (as in the
    # published example)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.0))
    for ax, lab in zip(axes.ravel(),
                       ("interpolation", "projection", "minimax", "CF")):
        _na(ax, lab)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "EightShades_repl_04.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
