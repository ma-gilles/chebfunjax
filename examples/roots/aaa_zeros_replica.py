"""Finding zeros with AAA.

Faithful replica of roots/AAAZeros.m by Nick Trefethen
(September 2023): the zeros returned by the AAA algorithm as cheap,
accurate root estimates — for a Bessel function, random polynomials,
and analytic functions on the unit disk.

Original: https://www.chebfun.org/examples/roots/AAAZeros.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.aaa import aaa
from chebfunjax.utils.transforms import leg2cheb

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')


def _fmt_e(v):
    sign = "+" if v.imag >= 0 else "-"
    return f"{v.real:>24.15e} {sign} {abs(v.imag):.15e}i"


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Bessel roots from an AAA fit on 400 sample points
    J0 = cj.chebfun(lambda x: jnp.asarray(sp.j0(np.asarray(x))),
                    domain=(0.0, 100.0))
    X = np.linspace(0, 100, 400)
    t0 = time.time()
    j0, _pol, _res, zer, *_ = aaa(jnp.asarray(sp.j0(X)),
                                  jnp.asarray(X))
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    zer = np.asarray(zer)
    zer = np.sort(zer[(zer.imag == 0) | (np.abs(zer.imag) < 1e-13)]
                  .real)
    zer = zer[(zer >= 0) & (zer <= 100)]
    rts = np.asarray(J0.roots())
    print("max_diff =")
    print(f"     {np.max(np.abs(zer - rts)):.15e}")

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(X, np.asarray(J0(X)), 'c', lw=1.4)
    ax.plot(X, np.asarray(j0(jnp.asarray(X))).real, 'k:', lw=1.2)
    ax.grid(True)
    ax.set_title("Roots of Bessel function J_0 in [0,100]",
                 fontsize=11)
    ax.plot(zer, np.zeros_like(zer), 'r.', ms=10)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "AAAZeros_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # random degree-50 polynomials: AAA zeros vs chebfun roots
    X = np.linspace(-1, 1, 600)
    n = 50
    diffs = []
    rs = np.random.RandomState(1)
    t0 = time.time()
    for _k in range(10):
        cleg = rs.randn(n + 1)
        ccheb = np.asarray(leg2cheb(jnp.asarray(cleg),
                                    normalize=True))
        p = cj.Chebfun.from_coeffs(jnp.asarray(ccheb))
        pa, _pol, _res, zer, *_ = aaa(jnp.asarray(np.asarray(p(X))),
                                      jnp.asarray(X), tol=1e-9)
        zer = np.asarray(zer)
        zer = np.sort(zer[np.abs(zer.imag) < 1e-13].real)
        zer = zer[(zer >= -1) & (zer <= 1)]
        rts = np.asarray(p.roots())
        m = min(len(rts), len(zer))
        diffs.append(np.max(np.abs(rts[:m] - zer[:m]))
                     if m else np.nan)
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    print("diff =")
    for d in diffs:
        print(f"     {d:.15e}")

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(X, np.asarray(p(X)), 'c-', lw=2)
    ax.plot(X, np.asarray(pa(jnp.asarray(X))).real, 'k:', lw=1)
    ax.plot(zer, np.zeros_like(zer), 'r.', ms=10)
    ax.grid(True)
    ax.set_title("Roots in [-1,1] of random polynomial, degree = 50",
                 fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "AAAZeros_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # analytic functions on the unit disk
    Z = np.exp(1j * np.linspace(0, 2 * np.pi, 1000))
    cases = [
        lambda z: (z - 0.5j) * np.exp(z),
        lambda z: np.cosh(np.pi * z),
        lambda z: np.cosh(np.exp(z)) * (z - 0.3) * (1 + 4 * z**2),
        lambda z: (z**3 - 1 / 8) * np.exp((-1 - 2j) * z),
    ]
    handles, zsets = [], []
    t0 = time.time()
    for fcase in cases:
        r, _pol, _res, zer, *_ = aaa(jnp.asarray(fcase(Z)),
                                     jnp.asarray(Z))
        zer = np.asarray(zer)
        zin = zer[np.abs(zer) <= 1]
        handles.append(r)
        zsets.append(zin)
        print("zeros =")
        for v in zin:
            print(f"    {_fmt_e(v)}")
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.0))
    xs = np.linspace(-1.3, 1.3, 360)
    XX, YY = np.meshgrid(xs, xs)
    for ax, r, zin in zip(axes.ravel(), handles, zsets):
        with np.errstate(all="ignore"):
            V = np.asarray(r(jnp.asarray(XX + 1j * YY)))
        H = (np.angle(V) + np.pi) / (2 * np.pi)
        ax.imshow(plt.cm.hsv(H), origin="lower",
                  extent=(-1.3, 1.3, -1.3, 1.3))
        ax.plot(Z.real, Z.imag, 'k-', lw=1)
        ax.plot(zin.real, zin.imag, 'r.', ms=12)
        ax.set_aspect("equal")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "AAAZeros_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
