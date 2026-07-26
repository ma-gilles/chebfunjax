"""Accuracy tests for Spherefun.fast_sphere_eval (the fastSphereEval port).

The 2D-NUFFT evaluation of the doubled-up Fourier coefficient matrix reaches
~1e-15 per point on a band-limited function -- the sub-ulp gain over the
Horner scheme (:meth:`Spherefun.__call__`) that lets ``rotate`` reach MATLAB's
round-trip bound.

Provenance
----------
MATLAB source : @spherefun/fastSphereEval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun


def _cart(lam, th):
    return (np.cos(lam) * np.sin(th),
            np.sin(lam) * np.sin(th),
            np.cos(th))


def test_fast_sphere_eval_bandlimited_closed_form():
    # A band-limited (polynomial-in-Cartesian) function has an exact spherefun
    # representation, so fast_sphere_eval should match the closed form to ~1e-15.
    def fexact(lam, th):
        x, y, z = _cart(np.asarray(lam), np.asarray(th))
        return 1.0 + 2.0 * x * y + 3.0 * z * x - z * z + 0.5 * (x**2 - y**2)

    f = Spherefun.from_function(
        lambda lam, th: fexact(lam, th))

    rng = np.random.default_rng(0)
    lam = rng.uniform(-np.pi, np.pi, 4000)
    th = rng.uniform(1e-3, np.pi - 1e-3, 4000)

    vals = f.fast_sphere_eval(lam, th)
    ref = fexact(lam, th)
    assert np.max(np.abs(vals - ref)) < 5e-15
    assert np.max(np.abs(np.imag(vals))) < 1e-15


def test_fast_sphere_eval_matches_call():
    # fast_sphere_eval and __call__ evaluate the same spherefun and must agree
    # to a couple of ulps on a high-rank function.
    f = Spherefun.from_function(
        lambda lam, th: jnp.sin(
            jnp.cos(th) + jnp.cos(lam - 0.2) * jnp.sin(th)
            + jnp.sin(lam + 0.4) * jnp.sin(th)) ** 8)

    rng = np.random.default_rng(1)
    lam = rng.uniform(-np.pi, np.pi, 2000)
    th = rng.uniform(1e-2, np.pi - 1e-2, 2000)

    fse = f.fast_sphere_eval(lam, th)
    call = np.asarray(f(jnp.asarray(lam), jnp.asarray(th)))
    assert np.max(np.abs(fse - call)) < 1e-13


def test_fast_sphere_eval_shape_and_grid():
    # Shape is preserved and matches a broadcast meshgrid evaluation.
    f = Spherefun.from_function(
        lambda lam, th: jnp.cos(th) ** 2 + jnp.sin(lam) * jnp.sin(th))
    lam = np.linspace(-3.0, 3.0, 11)
    th = np.linspace(0.05, 3.09, 7)
    LL, TT = np.meshgrid(lam, th, indexing="ij")
    vals = f.fast_sphere_eval(LL, TT)
    assert vals.shape == LL.shape
    ref = np.asarray(f(jnp.asarray(LL), jnp.asarray(TT)))
    assert np.max(np.abs(vals - ref)) < 1e-13
