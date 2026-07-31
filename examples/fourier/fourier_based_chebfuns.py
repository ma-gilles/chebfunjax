"""Fourier-based chebfuns.

Faithful replica of fourier/FourierBasedChebfuns.m by Grady Wright,
June 2014 -- every section, computation, and printed display of the
published page is reproduced in order.

Original: https://www.chebfun.org/examples/fourier/FourierBasedChebfuns.html
Copyright 2014 by The University of Oxford and The Chebfun Developers.

Output-parity note: all deterministic quantities (lengths, ratio,
max/min/roots, integral, heart area and its error) reproduce the
published values; the noisy-samples section uses numpy's RandomState
(MATLAB's rng(0) ziggurat stream is not reproducible) so the noise
realisation differs while the construction and mollification are
faithful.
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
from chebfunjax.utils.quadrature import trigpts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'fourier')


def _show(name, f):
    print(f"{name} =")
    print(repr(f))


def _save(fig, stem):
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, stem + ".png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    dom = [-np.pi, np.pi]

    # -- Construction and comparison --------------------------------
    f = cj.chebfun(lambda x: jnp.cos(8 * jnp.sin(x)), domain=dom,
                   trig=True)
    _show("f", f)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    xs = np.linspace(*dom, 800)
    ax.plot(xs, np.asarray(f(jnp.asarray(xs))), "b")
    _save(fig, "FourierBasedChebfuns_01")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    c = np.abs(np.asarray(f.funs[0].coeffs))
    n = len(c)
    k = np.arange(n) - (n - 1) // 2
    ax.semilogy(k, np.maximum(c, 1e-18), ".b", ms=6)
    ax.set_ylim(1e-18, 1)
    ax.set_xlabel("wave number")
    ax.set_ylabel("magnitude of coefficient")
    _save(fig, "FourierBasedChebfuns_02")

    f_cheby = cj.chebfun(lambda x: jnp.cos(8 * jnp.sin(x)), domain=dom)
    _show("f_cheby", f_cheby)

    print("ratio =")
    print(f"   {len(f_cheby) / len(f):.6f}")
    print("theoretical =")
    print(f"   {np.pi / 2:.6f}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_step = cj.chebfun(
            lambda x: 0.5 * (1.0 + jnp.sign(x)), domain=dom, trig=True)
    _show("f", f_step)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xs, np.asarray(f_step(jnp.asarray(xs))), "b", lw=0.7)
    _save(fig, "FourierBasedChebfuns_03")

    f_split = cj.chebfun(lambda x: 0.5 * (1.0 + jnp.sign(x)),
                         domain=dom, splitting=True)
    _show("f", f_split)

    # -- Basic operations -------------------------------------------
    f = cj.chebfun(
        lambda x: jnp.tanh(jnp.cos(1 + 2 * jnp.sin(x)) ** 2) - 0.5,
        domain=dom, trig=True)
    _show("f", f)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    fx = np.asarray(f(jnp.asarray(xs)))
    ax.plot(xs, fx, "b")
    _save(fig, "FourierBasedChebfuns_04")

    (xminf, minf), (xmaxf, maxf) = f.minandmax()
    rootsf = np.sort(np.asarray(f.roots()))
    print("maxf =")
    print(f"   {maxf:.6f}")
    print("minf =")
    print(f"  {minf:.6f}")
    print("rootsf =")
    for r in rootsf:
        print(f"  {r: .6f}")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xs, fx, "b", label="f")
    ax.plot([xmaxf], [maxf], "gs", label="max f")
    ax.plot([xminf], [minf], "md", label="min f")
    ax.plot(rootsf, 0 * rootsf, "ro", label="zeros f")
    ax.legend(loc="lower left", fontsize=9)
    _save(fig, "FourierBasedChebfuns_05")

    df = f.diff()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xs, np.asarray(df(jnp.asarray(xs))), "b")
    _save(fig, "FourierBasedChebfuns_06")

    print("intf =")
    print(f"  {float(f.sum()):.6f}")

    # -- Complex-valued trigfuns: the heart curve --------------------
    fh = cj.chebfun(
        lambda x: 1j * (13 * jnp.cos(x) - 5 * jnp.cos(2 * x)
                        - 2 * jnp.cos(3 * x) - jnp.cos(4 * x))
        + 16 * jnp.sin(x) ** 3, domain=dom, trig=True)
    _show("f", fh)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    vals = np.asarray(fh(jnp.asarray(xs)))
    ax.plot(np.real(vals), np.imag(vals), "b")
    ax.set_aspect("equal")
    _save(fig, "FourierBasedChebfuns_07")

    area_heart = abs(float((fh.real() * fh.imag().diff()).sum()))
    print("area_heart =")
    print(f"  {area_heart:.6f}")
    err = (area_heart - 180 * np.pi) / (180 * np.pi)
    print("err =")
    print(f"    {err:.15e}")

    # -- circconv + construction from values -------------------------
    rng = np.random.RandomState(0)
    n = 201
    x, _ = trigpts(n, tuple(dom))
    x = np.asarray(x)
    func_vals = np.exp(np.sin(2 * np.pi * x)) + 0.05 * rng.randn(n)
    fN = cj.Chebfun.from_trig_values(jnp.asarray(func_vals), tuple(dom)) \
        if hasattr(cj.Chebfun, "from_trig_values") else None
    if fN is None:
        from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
        from chebfunjax.domain import Domain
        from chebfunjax.tech.trigtech import Trigtech
        tech = Trigtech.from_values(jnp.asarray(func_vals))
        fN = Chebfun(funs=[_Piece(tech=tech, interval=tuple(dom))],
                     domain=Domain(tuple(dom)))
    _show("f", fN)

    sigma = 0.1
    gm = cj.chebfun(
        lambda t: 1 / (sigma * np.sqrt(2 * np.pi))
        * jnp.exp(-0.5 * (t / sigma) ** 2), domain=dom, trig=True)
    h = fN.circconv(gm)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xs, np.asarray(gm(jnp.asarray(xs))), "b", label="Mollifier g")
    ax.plot(xs, np.asarray(fN(jnp.asarray(xs))), "r",
            label="Noisy function f")
    ax.plot(xs, np.asarray(h(jnp.asarray(xs))), "k",
            label="Smoothed function h")
    ax.legend(fontsize=9)
    _save(fig, "FourierBasedChebfuns_08")

    return True


if __name__ == "__main__":
    run()
