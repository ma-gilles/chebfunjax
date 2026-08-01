"""AAA rational approximation.

Faithful replica of approx/AAAApprox.m by Nick Trefethen (December
2016): the AAA algorithm on intervals, with restricted type, and on an
arbitrary point set in the complex plane (the moustache).

Original: https://www.chebfun.org/examples/approx/AAAApprox.html
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
from chebfunjax.utils.aaa import aaa
from chebfunjax.utils.minimax import minimax

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def _aaa_on_interval(f, dom=(-1.0, 1.0), **kw):
    """MATLAB aaa(handle): automated sample-set choice (aaa_autoZ)."""
    return aaa(f, dom=dom, **kw)


def _disp_polres(pol, res):
    order = np.argsort(np.asarray(pol).real)
    print('        poles             residues')
    for p, r in zip(np.asarray(pol)[order], np.asarray(res)[order]):
        print(f"  {p.real:8.4f} {'+' if p.imag >= 0 else '-'} "
              f"{abs(p.imag):.4f}i  {r.real:8.4f} "
              f"{'+' if r.imag >= 0 else '-'} {abs(r.imag):.4f}i")


def _ezplot(fun, dom, fname, ylim=None, title=""):
    xs = np.linspace(dom[0], dom[1], 3000)
    ys = np.asarray([float(np.real(fun(x))) for x in xs])
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.plot(xs, ys, lw=1.4)
    ax.grid(True)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    gam = lambda z: jnp.asarray(sp_gamma(np.asarray(z)))  # noqa: E731

    # Section 2: gamma on [-1,1]
    r, pol, res, *_ = _aaa_on_interval(gam)
    _ezplot(lambda x: r(np.asarray([x]))[0], (-3, 3),
            "AAAApprox_repl_01.png", ylim=(-8, 8))
    _disp_polres(pol, res)

    # gamma on [-2,2]: type (7,7)
    r, pol, res, *_ = _aaa_on_interval(gam, dom=(-2.0, 2.0))
    _disp_polres(pol, res)
    _ezplot(lambda x: r(np.asarray([x]))[0], (-3, 3),
            "AAAApprox_repl_02.png", ylim=(-8, 8))

    # chebfun input: f = sin(20x)/(1+25x^2) on [-1,2]
    x = cj.chebfun(lambda t: t, domain=(-1.0, 2.0))
    f = (20 * x).sin() / (1 + 25 * x**2)
    xs = np.linspace(-1, 2, 3000)
    fv = np.asarray(f(jnp.asarray(xs)))
    r, pol, res, *_ = aaa(jnp.asarray(fv), jnp.asarray(xs))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    axes[0].plot(xs, fv, lw=1.2)
    axes[0].grid(True)
    axes[0].set_ylim(-1, 1)
    axes[0].set_title("function f", fontsize=12)
    rv = np.real(np.asarray(r(np.asarray(xs))))
    axes[1].plot(xs, rv, lw=1.2)
    axes[1].grid(True)
    axes[1].set_ylim(-1, 1)
    axes[1].set_title("AAA approx r", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "AAAApprox_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("ans =")
    print(f"    {len(np.asarray(pol))}")
    inner = np.asarray(pol)[np.abs(np.asarray(pol)) < 1]
    print("ans =")
    for p in inner[np.argsort(-inner.imag)]:
        print(f"  {p.real:.15f} {'+' if p.imag >= 0 else '-'} "
              f"{abs(p.imag):.15f}i")

    # Section 3: full-precision AAA of exp, then type (3,3) vs best
    r, *_ = _aaa_on_interval(lambda z: jnp.exp(z))
    _ezplot(lambda x: np.exp(x) - np.real(r(np.asarray([x]))[0]),
            (-1, 1), "AAAApprox_repl_04.png",
            title="AAA approx of exp(x)")

    r3, *_ = _aaa_on_interval(lambda z: jnp.exp(z), mmax=4, lawson=0)
    res_mm = minimax(lambda t: jnp.exp(t), 3, rational=True, denom=3)
    rbest = res_mm.r
    xs = np.linspace(-1, 1, 3000)
    e_aaa = np.exp(xs) - np.real(r3(xs))
    e_best = np.exp(xs) - np.asarray(rbest(xs))
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.plot(xs, e_aaa, lw=1.4)
    ax.plot(xs, e_best, lw=1.4)
    ax.grid(True)
    ax.set_ylim(-1e-6, 1e-6)
    ax.set_title("AAA and best type (3,3) approximants to exp(x)",
                 fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "AAAApprox_repl_05.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # AAA of |x|
    r, *_ = _aaa_on_interval(lambda z: jnp.abs(z))
    _ezplot(lambda x: abs(x) - np.real(r(np.asarray([x]))[0]),
            (-1, 1), "AAAApprox_repl_06.png", ylim=(-5e-14, 5e-14))

    # Section 4: the moustache in the complex plane
    npts = 2000
    rs = np.random.RandomState(0)
    X = 8 * rs.random_sample(npts) - 4
    Y = 2 * rs.random_sample(npts) - 1 + X**3 / 16
    Z = X + 1j * Y
    ff = lambda z: np.sqrt(2 + z**2) / (z - 4)  # noqa: E731
    r, pol, *_ = aaa(jnp.asarray(ff(Z)), jnp.asarray(Z))
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(Z.real, Z.imag, '.k', ms=3)
    p = np.asarray(pol)
    ax.plot(p.real, p.imag, '.r', ms=10)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-5, 5)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "AAAApprox_repl_07.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    pts = [5 + 5j, 5 + 0j, 5 - 5j]
    print("  Column 1")
    for z in pts:
        v = ff(z)
        print(f"  {v.real:.15f} {'+' if v.imag >= 0 else '-'} "
              f"{abs(v.imag):.15f}i")
    print("  Column 2")
    for z in pts:
        v = complex(np.asarray(r(np.asarray([z])))[0])
        print(f"  {v.real:.15f} {'+' if v.imag >= 0 else '-'} "
              f"{abs(v.imag):.15f}i")


if __name__ == "__main__":
    run()
