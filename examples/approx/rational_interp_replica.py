"""Rational interpolation, robust and non-robust.

Faithful replica of approx/RationalInterp.m by Nick Trefethen and
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
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
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
    fig.savefig(os.path.join(_IMG, "RationalInterp_repl_01.png"),
                dpi=150, bbox_inches="tight")
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
    axes[0].plot(xx, np.asarray(f(jnp.asarray(xx))), '.k', ms=12)
    axes[0].set_title("Type (3,3) rational interpolant to cos(e^x) "
                      "in 7 Chebyshev points", fontsize=11)
    rh, *_ = ratinterp(lambda x: f(x), 3, 3, 15)
    axes[1].plot(XS, np.asarray(rh(XS)), 'm', lw=1.6)
    xx = np.asarray(chebpts(16))
    axes[1].plot(xx, np.asarray(f(jnp.asarray(xx))), '.k', ms=12)
    axes[1].set_title("Type (3,3) least-squares approximant to cos(e^x) "
                      "in 16 Chebyshev points", fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "RationalInterp_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # exp(x): robust vs non-robust type (8,8)
    fe = lambda x: jnp.exp(jnp.asarray(x))  # noqa: E731
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.0))
    rh_rob, a, b, mu, nu, *_ = ratinterp(fe, 8, 8)
    axes[0].plot(XS, np.asarray(rh_rob(XS)), 'm', lw=1.6)
    xx = np.asarray(chebpts(17))
    axes[0].plot(xx, np.exp(xx), '.k', ms=12)
    axes[0].set_title("robust", fontsize=11)
    rh0, a0, b0, *_ = ratinterp(fe, 8, 8, None, None, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        axes[1].plot(XS, np.asarray(rh0(XS)), lw=1.6)
    axes[1].plot(xx, np.exp(xx), '.k', ms=12)
    axes[1].set_title("non-robust (tol = 0)", fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "RationalInterp_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    def _real_roots(c):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = np.polynomial.chebyshev.chebroots(np.asarray(c))
        r = r[np.abs(r.imag) < 1e-8].real
        return np.sort(r[(r > -1) & (r < 1)])

    sz = _real_roots(a0)
    sp = _real_roots(b0)
    print("spurious_zeros =")
    for v in sz:
        print(f"  {v:.15f}")
    print("spurious_poles =")
    for v in sp:
        print(f"  {v:.15f}")
    if len(sz) == len(sp):
        print("separation =")
        for v in (sp - sz):
            print(f"   {v:.3e}")

    print("degree_of_p =")
    print(f"     {mu}")
    print("spurious_zeros =")
    print("   " + (", ".join(f"{v:.15f}" for v in _real_roots(a))
                   or "Empty matrix: 0-by-1"))
    print("degree_of_q =")
    print(f"     {nu}")
    print("spurious_poles =")
    print("   " + (", ".join(f"{v:.15f}" for v in _real_roots(b))
                   or "Empty matrix: 0-by-1"))


if __name__ == "__main__":
    run()
