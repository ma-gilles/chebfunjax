"""Analytic continuation via polynomials and rational functions.

Faithful replica of complex/AnalyticContinuation.m by Nick Trefethen
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
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.ratapprox import ratinterp

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'complex')

X = np.arange(-6, 6.05, 0.05)
XX, YY = np.meshgrid(X, X)
ZZ = XX + 1j * YY
LEV1 = np.arange(0.25, 2.01, 0.25)
LEV2 = 10.0 ** np.arange(1, 20, 2)


def _contours(F, fname, ellipse_rho=None):
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    with np.errstate(all="ignore"):
        A = np.abs(F)
    ax.contour(X, X, A, levels=LEV1, colors='k', linewidths=0.8)
    ax.contour(X, X, A, levels=LEV2, colors='r', linewidths=0.8)
    if ellipse_rho is not None:
        t = np.linspace(0, 2 * np.pi, 400)
        w = ellipse_rho * np.exp(1j * t)
        e = (w + 1 / w) / 2
        ax.plot(e.real, e.imag, 'b', lw=1.6)
    ax.axis([-6, 6, -6, 6])
    ax.set_aspect("equal")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    f = lambda z: np.tanh(z)  # noqa: E731
    _contours(f(ZZ), "AnalyticContinuation_repl_01.png")

    p = cj.chebfun(lambda z: jnp.tanh(z))
    print("ans =")
    print(f"    {len(p)}")

    # Chebfun-ellipse parameter from the coefficient decay
    c = np.abs(np.asarray(p.coeffs))
    n = len(c)
    rho = np.exp(-np.polyfit(np.arange(n), np.log(c + 1e-300), 1)[0])

    pp = np.asarray(p(jnp.asarray(ZZ)))
    _contours(pp, "AnalyticContinuation_repl_02.png", ellipse_rho=rho)

    # rational interpolant reaches much further.  The returned handle
    # and pole list are real-line oriented, so evaluate p/q and find
    # the full complex pole set from the coefficient vectors directly.
    from numpy.polynomial import chebyshev as C
    rh, a, b, mu, nu, _poles_real, res = ratinterp(
        lambda x: jnp.tanh(x), 7, 8)
    a = np.asarray(a)
    b = np.asarray(b)
    with np.errstate(all="ignore"):
        rr = C.chebval(ZZ, a) / C.chebval(ZZ, b)
    _contours(rr, "AnalyticContinuation_repl_03.png")

    exact = 0.5j * np.pi * np.arange(-7, 8, 2)
    poles = np.asarray(C.chebroots(b), dtype=complex)
    print("   Exact     rational approx")
    for e, q in zip(sorted(exact, key=lambda v: (abs(v.imag), v.imag)),
                    sorted(poles, key=lambda v: (abs(v.imag), v.imag))):
        print(f"  {e.real:.15f} {'+' if e.imag>=0 else '-'} "
              f"{abs(e.imag):.15f}i   "
              f"{q.real:.15f} {'+' if q.imag>=0 else '-'} "
              f"{abs(q.imag):.15f}i")


if __name__ == "__main__":
    run()
