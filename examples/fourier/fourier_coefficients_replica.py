"""Fourier coefficients.

Faithful replica of fourier/FourierCoefficients.m by Grady Wright,
June 2014 -- every section, computation, and printed display of the
published page reproduced in order.

Original: https://www.chebfun.org/examples/fourier/FourierCoefficients.html
Copyright 2014 by The University of Oxford and The Chebfun Developers.
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

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'fourier')


def _pc(vals):
    for v in np.atleast_1d(vals):
        v = complex(v)
        sign = "+" if v.imag >= 0 else "-"
        print(f"  {v.real: .15f} {sign} {abs(v.imag):.15f}i")


def _pr(vals):
    for v in np.atleast_1d(vals):
        print(f"   {float(np.real(v)): .15f}")


def _save(fig, stem):
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, stem + ".png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    dom = [-np.pi, np.pi]

    # -- Smooth periodic functions -----------------------------------
    u = cj.chebfun(lambda x: 1 - 4 * jnp.cos(x) + 6 * jnp.sin(2 * x),
                   domain=dom, trig=True)
    c = np.asarray(u.trigcoeffs())
    print('Fourier coeffs of 1 - 4*cos(x) + 6*sin(2*x):')
    print('c ='); _pc(c)

    a, b = u.trigcoeffs(form="cos_sin")
    print('Fourier cosine coeffs of 1 - 4*cos(x) + 6*sin(2*x)')
    print('a ='); _pr(a)
    print('Fourier sine coeffs of 1 - 4*cos(x) + 6*sin(2*x)')
    print('b ='); _pr(b)

    # -- Truncation: 3/(5-4cos x), c_k = 2^-|k| ----------------------
    numCoeffs = 11
    u = cj.chebfun(lambda x: 3.0 / (5 - 4 * jnp.cos(x)), domain=dom,
                   trig=True)
    c = np.asarray(u.trigcoeffs(numCoeffs))
    print('Fourier coeffs of 3/(5-4cos(x)):')
    print('c ='); _pc(c)

    # -- Finitely smooth: |sin(x)|^3 ---------------------------------
    numCoeffs = 17
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = cj.chebfun(lambda x: jnp.abs(jnp.sin(x)) ** 3, domain=dom,
                       trig=True)
    c = np.asarray(u.trigcoeffs(numCoeffs))[::-1]
    print('Fourier coeffs of |sin(x)|^3')
    print('c ='); _pc(c)
    print('ans ='); print(f"   {len(u)}")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    cc = np.abs(np.asarray(u.funs[0].tech.coeffs))
    n = len(cc)
    k = np.abs(np.arange(n) - (n - 1) // 2)
    pos = k > 0
    ax.loglog(k[pos], np.maximum(cc[pos], 1e-18), ".b", ms=4)
    ks = np.array([100.0, n / 2.0])
    ax.loglog(ks, 10 * ks ** -4.0, "k-", lw=1.6)
    ax.text(500, 50 * 500.0 ** -4, r"$O(k^{-4})$", fontsize=12)
    ax.set_ylim(1e-15, 1)
    ax.set_xlabel("wave number")
    ax.set_ylabel("magnitude of coefficient")
    _save(fig, "FourierCoefficients_repl_01")

    # -- Non-smooth: square wave via splitting -----------------------
    sq_wave = lambda x: jnp.sign(jnp.sin(x))
    u = cj.chebfun(sq_wave, domain=dom, splitting=True)
    numCoeffs = 15
    a, b = u.trigcoeffs(numCoeffs, form="cos_sin")
    print('Fourier sine coeffs of unit step function:')
    print('b ='); _pr(b)
    print('            k               pi/4*b_k')
    for kk in range(1, 8):
        print(f"   {kk:.4f}    {np.pi / 4 * float(np.real(np.asarray(b)[kk - 1])):.15f}")
    print('ans ='); print(f"   {float(np.max(np.abs(np.asarray(a)))):.15e}")

    # -- Truncated reconstruction and periodic extension -------------
    numModes = 15
    c = np.asarray(u.trigcoeffs(2 * numModes + 1))
    u_trunc = cj.chebfun(jnp.asarray(c), domain=dom, trig=True,
                         coeffs=True)
    xs = np.linspace(-np.pi, np.pi, 1200)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xs, np.asarray(u(jnp.asarray(xs))), "k-", lw=1.6)
    ax.plot(xs, np.real(np.asarray(u_trunc(jnp.asarray(xs)))), "b-",
            lw=1.6)
    _save(fig, "FourierCoefficients_repl_02")

    xw = np.linspace(-4 * np.pi, 4 * np.pi, 3000)
    uw = cj.chebfun(sq_wave, domain=[-4 * np.pi, 4 * np.pi],
                    splitting=True)
    # periodic extension of the truncated series
    period = 2 * np.pi
    xmap = ((xw + np.pi) % period) - np.pi
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xw, np.asarray(uw(jnp.asarray(xw))), "k-", lw=1.2)
    ax.plot(xw, np.real(np.asarray(u_trunc(jnp.asarray(xmap)))), "b-",
            lw=1.2)
    _save(fig, "FourierCoefficients_repl_03")

    # -- Sawtooth ----------------------------------------------------
    sawtooth = lambda x: jnp.mod(x + np.pi, 2 * np.pi) / (2 * np.pi)
    u = cj.chebfun(sawtooth, domain=dom, splitting=True)
    c = np.asarray(u.trigcoeffs(2 * numModes + 1))
    u_trunc = cj.chebfun(jnp.asarray(c), domain=dom, trig=True,
                         coeffs=True)
    uw = cj.chebfun(sawtooth, domain=[-4 * np.pi, 4 * np.pi],
                    splitting=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xw, np.asarray(uw(jnp.asarray(xw))), "k-", lw=1.2)
    ax.plot(xw, np.real(np.asarray(u_trunc(jnp.asarray(xmap)))), "b-",
            lw=1.2)
    _save(fig, "FourierCoefficients_repl_04")

    return True


if __name__ == "__main__":
    run()
