"""Analytic continuation via polynomials and rational functions.

Translation of complex/AnalyticContinuation.m by Nick Trefethen
(May 2011): how far off [-1,1] the chebfun polynomial of tanh(z)
continues accurately (its Chebfun ellipse), and how much further a
rational interpolant reaches, with its poles matching the true poles
of tanh.

Original: https://www.chebfun.org/examples/complex/AnalyticContinuation.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, plotregion
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.ratapprox import ratinterp

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'complex')

X = np.arange(-6, 6.05, 0.05)
XX, YY = np.meshgrid(X, X)
ZZ = XX + 1j * YY
LEV1 = np.arange(0.25, 2.01, 0.25)
LEV2 = 10.0 ** np.arange(1, 20, 2)


def _contours(ax, F):
    """contour(x,y,abs(F),lev1,'k'), hold on, contour(x,y,abs(F),lev2,'r')"""
    with np.errstate(all="ignore"):
        A = np.abs(np.asarray(F))
    ax.contour(X, X, A, levels=LEV1, colors='k', linewidths=0.6)
    ax.contour(X, X, A, levels=LEV2, colors='r', linewidths=0.6)


def _square(fig, ax, fname):
    """axis(6*[-1 1 -1 1]), axis square, then save."""
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, fname))
    plt.close(fig)


def _cplx(v):
    """One entry of MATLAB's format-long complex matrix display."""
    re = float(np.real(v)) + 0.0
    im = float(np.imag(v))
    return f"{re:19.15f} {'-' if im < 0 else '+'}{abs(im):18.15f}i"


def run():
    os.makedirs(_IMG, exist_ok=True)

    def f(z):
        return jnp.tanh(z)

    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    _contours(ax, f(jnp.asarray(ZZ)))
    _square(fig, ax, "AnalyticContinuation_01.png")

    z = cj.chebfun(lambda t: t)
    p = cj.tanh(z)   # p = f(z)
    print("ans =")
    print(f"    {len(p)}")

    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    plotregion(p, ax=ax)
    _square(fig, ax, "AnalyticContinuation_02.png")

    pp = p(jnp.asarray(ZZ))
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    _contours(ax, pp)
    plotregion(p, ax=ax)
    _square(fig, ax, "AnalyticContinuation_03.png")

    # [p,q,r,mu,nu,poles] = ratinterp(f,7,8); rr = r(zz)
    _r, a, b, mu, nu, _poles, _res = ratinterp(f, 7, 8)
    pn = cj.chebfun(jnp.asarray(a), coeffs=True)
    qn = cj.chebfun(jnp.asarray(b), coeffs=True)
    rr = pn(jnp.asarray(ZZ)) / qn(jnp.asarray(ZZ))
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    _contours(ax, rr)
    _square(fig, ax, "AnalyticContinuation_04.png")

    # poles = roots(q, 'all')
    poles = np.asarray(qn.roots(all_roots=True))
    exact = 0.5j * np.pi * np.arange(-7, 8, 2)

    def mat_sort(v):
        return sorted(v, key=lambda w: (abs(w), np.angle(w)))

    print("   Exact     rational approx")
    for j, col in enumerate((mat_sort(exact), mat_sort(poles)), start=1):
        print(f"  Column {j}")
        for v in col:
            print(_cplx(v))


if __name__ == "__main__":
    run()
