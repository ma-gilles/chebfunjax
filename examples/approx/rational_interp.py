"""Rational interpolation, robust and non-robust.

Translation of approx/RationalInterp.m by Nick Trefethen and
Ricardo Pachon (November 2011): rational interpolation in Chebyshev
points, spurious pole-zero (Froissart) artifacts, and the robust
SVD-based degree reduction.

Original: https://www.chebfun.org/examples/approx/RationalInterp.html
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
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.quadrature import chebpts
from chebfunjax.utils.ratapprox import ratinterp

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

XS = np.linspace(-1, 1, 3000)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # A tiny perturbation can produce a spurious pole-zero pair
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.0))
    for j, ep in enumerate((0.1, 0.001)):
        xs = XS[np.abs(XS - 1 / 3) > 2e-3]
        rv = 1 + (4 / 3) * ep * xs / (xs - 1 / 3)
        axes[j].plot(xs, rv, lw=1.6)
        axes[j].axis([-1, 1, 0, 3])
        axes[j].plot([-1, 0, 1], [1 + ep, 1, 1 + 2 * ep], '.k', ms=14)
        axes[j].text(-0.8, 2.3, f"ep = {ep:g}", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "RationalInterp_01.png"), size=(600, 270))
    plt.close(fig)

    # Error table for cos(e^x)
    f = cj.chebfun(lambda x: jnp.cos(jnp.exp(x)))
    print('    (n,n)       Error ')
    for n in range(1, 7):
        rh, a, b, mu, nu, poles, res = ratinterp(lambda x: f(x), n, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            err = np.max(np.abs(np.asarray(f(jnp.asarray(XS)))
                                - np.asarray(rh(XS))))
        inpoles = (poles[(np.real(poles) > -1) & (np.real(poles) < 1)]
                   if len(poles) else [])
        s = "    Inf" if len(inpoles) else f"{err:7.2e}"
        print(f"    ({n},{n})     {s}")

    # (3,3) interpolant in 7 points vs least-squares in 15 points
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.4))
    rh, *_ = ratinterp(lambda x: f(x), 3, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        axes[0].plot(XS, np.asarray(rh(XS)), lw=1.6)
    xx = np.asarray(chebpts(7))
    axes[0].plot(xx, np.asarray(f(jnp.asarray(xx))), '.k', ms=7)
    axes[0].set_title("Type (3,3) rational interpolant to cos(e^x) "
                      "in 7 Chebyshev points", fontsize=11)
    rh, *_ = ratinterp(lambda x: f(x), 3, 3, 15)
    axes[1].plot(XS, np.asarray(rh(XS)), color='#FF00FF', lw=1.6)
    xx = np.asarray(chebpts(16))
    axes[1].plot(xx, np.asarray(f(jnp.asarray(xx))), '.k', ms=7)
    axes[1].set_title("Type (3,3) least-squares approximant to cos(e^x) "
                      "in 16 Chebyshev points", fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "RationalInterp_02.png"), size=(600, 270))
    plt.close(fig)

    # exp(x): robust vs non-robust type (8,8)
    fe = cj.chebfun('x').exp()
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.0))
    rh_rob, a, b, mu, nu, *_ = ratinterp(fe, 8, 8)
    axes[0].plot(XS, np.asarray(rh_rob(XS)), color='#FF00FF', lw=1.6)
    xx = np.asarray(chebpts(17))
    axes[0].plot(xx, np.exp(xx), '.k', ms=7)
    rh0, a0, b0, *_ = ratinterp(fe, 8, 8, None, None, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        axes[1].plot(XS, np.asarray(rh0(XS)), lw=1.6)
    axes[1].plot(xx, np.exp(xx), '.k', ms=7)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "RationalInterp_03.png"), size=(600, 270))
    plt.close(fig)

    # format long; spurious_zeros = roots(p), spurious_poles = roots(q)
    p0 = cj.chebfun(jnp.asarray(a0), coeffs=True)
    q0 = cj.chebfun(jnp.asarray(b0), coeffs=True)
    sz = np.asarray(p0.roots())
    sp = np.asarray(q0.roots())
    _disp("spurious_zeros", sz)
    _disp("spurious_poles", sp)
    _disp("separation", sp - sz)

    # [p,q,rh,mu,nu] = ratinterp(f,8,8)
    p = cj.chebfun(jnp.asarray(a), coeffs=True)
    q = cj.chebfun(jnp.asarray(b), coeffs=True)
    _disp("degree_of_p", mu)
    _disp("spurious_zeros", np.asarray(p.roots()))
    _disp("degree_of_q", nu)
    _disp("spurious_poles", np.asarray(q.roots()))


def _disp(name, v):
    """MATLAB ``format long`` display of a scalar or real column vector."""
    print(f"{name} =")
    v = np.atleast_1d(np.asarray(v, dtype=float))
    if v.size == 0:
        print("   Empty matrix: 0-by-1")
        return
    if np.all(v == np.round(v)):
        w = len(str(int(np.max(np.abs(v))))) + (1 if np.any(v < 0) else 0)
        for x in v:
            print(f"{int(x):>{max(6, w + 3)}d}")
        return
    m = np.max(np.abs(v))
    if v.size == 1 and not 1e-3 <= m < 1e3:
        print(f"{v[0]:25.15e}")
        return
    e = 0 if 1e-3 <= m < 1e3 else int(np.floor(np.log10(m)))
    e += 1 if e < 0 else 0          # MATLAB: small entries scale to [0.1, 1)
    if e:
        print(f"   1.0e{e:+03d} *")
        v = v / 10.0**e
    for x in v:
        print("                   0" if x == 0 else f"{x:20.15f}")

if __name__ == "__main__":
    run()
