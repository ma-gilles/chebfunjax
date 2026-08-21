"""Port of MATLAB Chebfun tests/spherefun/test_subsref.m (Fable 5).

MATLAB's colon subsref becomes slice methods; evaluation calls map to
__call__.  The property-chain passes (f.rows(x) etc.) are MATLAB
indexing notation covered by direct attribute access elsewhere.

Provenance
----------
MATLAB source : tests/spherefun/test_subsref.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.spherefun.spherefun import Spherefun

jax.config.update("jax_enable_x64", True)

TOL = 1000 * 2.220446049250313e-16


def _sph(ff_cart):
    def f(lam, th):
        x = jnp.cos(lam) * jnp.sin(th)
        y = jnp.sin(lam) * jnp.sin(th)
        z = jnp.cos(th)
        return ff_cart(x, y, z)
    return Spherefun.from_function(f), f


def _maxdiff(g, fn):
    ts = jnp.linspace(-np.pi + 1e-9, np.pi - 1e-9, 41)
    return float(jnp.max(jnp.abs(jnp.asarray(g(ts))
                                 - jnp.asarray(fn(ts)))))


class TestSpherefunSubsref:
    def test_pointwise_evaluation(self):
        # pass(1): (lambda, theta) evaluation of a cartesian handle.
        f, fs = _sph(lambda x, y, z: z * jnp.sin(x * (y - 0.1)))
        v = 1 / np.sqrt(3)
        lam = np.arctan2(v, v)
        th = np.arccos(-v)
        got = float(f(jnp.asarray(lam), jnp.asarray(th)))
        want = float(-v * np.sin(v * (v - 0.1)))
        assert abs(got - want) < TOL

        # pass(2): spherical-handle evaluation.
        g = Spherefun.from_function(
            lambda lam, th: jnp.exp(jnp.cos(lam) * jnp.sin(th)
                                    * jnp.cos(th)))
        lam0, th0 = np.pi / 3, np.pi / 4
        want = np.exp(np.cos(lam0) * np.sin(th0) * np.cos(th0))
        assert abs(float(g(jnp.asarray(lam0), jnp.asarray(th0)))
                   - want) < TOL

    def test_theta_slices(self):
        # pass(3): f(:, th) as a trig chebfun in lambda.
        g = Spherefun.from_function(
            lambda lam, th: jnp.exp(jnp.cos(lam) * jnp.sin(th)
                                    * jnp.cos(th)))
        for th0 in (0.2, 0.7):
            sl = g.slice_theta(th0)
            ref = cj.chebfun(
                lambda lam, _t=th0: jnp.exp(
                    jnp.cos(lam) * jnp.sin(_t) * jnp.cos(_t)),
                domain=(-np.pi, np.pi), trig=True)
            assert _maxdiff(sl, ref) < TOL

    def test_lambda_slices(self):
        # pass(4): f(lam, :) as a trig chebfun in theta.
        g = Spherefun.from_function(
            lambda lam, th: jnp.exp(jnp.cos(lam) * jnp.sin(th)
                                    * jnp.cos(th)))
        for lam0 in (0.2, 0.7):
            sl = g.slice_lambda(lam0)
            xs = jnp.linspace(1e-6, np.pi - 1e-6, 31)
            want = jnp.exp(jnp.cos(lam0) * jnp.sin(xs)
                           * jnp.cos(xs))
            assert float(jnp.max(jnp.abs(
                jnp.asarray(sl(xs)) - want))) < TOL

    def test_z_slice(self):
        # pass(5): f(:, :, z), the circle of constant z.
        f, fs = _sph(lambda x, y, z: z * jnp.sin(x * (y - 0.1)))
        z0 = 0.1
        th0 = np.arccos(z0)
        sl = f.slice_z(z0)

        def ref(lam):
            x = jnp.cos(lam) * np.sin(th0)
            y = jnp.sin(lam) * np.sin(th0)
            return z0 * jnp.sin(x * (y - 0.1))

        assert _maxdiff(sl, ref) < TOL
