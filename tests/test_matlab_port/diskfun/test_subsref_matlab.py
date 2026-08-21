"""Port of MATLAB Chebfun tests/diskfun/test_subsref.m (Fable 5).

MATLAB's colon subsref becomes slice methods; the chebfunjax Diskfun
evaluates in polar (theta, r), so the cartesian passes convert
coordinates explicitly.

Provenance
----------
MATLAB source : tests/diskfun/test_subsref.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

jax.config.update("jax_enable_x64", True)

TOL = 1000 * 2.220446049250313e-16


def _maxdiff(g, fn, dom):
    ts = jnp.linspace(dom[0] + 1e-9, dom[1] - 1e-9, 41)
    return float(jnp.max(jnp.abs(jnp.asarray(g(ts))
                                 - jnp.asarray(fn(ts)))))


class TestDiskfunSubsref:
    def test_pointwise_evaluation(self):
        # pass(1)-(3): cartesian and polar evaluation.
        f = Diskfun.from_function(
            lambda t, r: jnp.sin((r * jnp.cos(t))
                                 * (r * jnp.sin(t) - 0.1)))
        x = y = 1 / np.sqrt(3)
        t0, r0 = float(np.arctan2(y, x)), float(np.hypot(x, y))
        want = np.sin(x * (y - 0.1))
        assert abs(float(f(jnp.asarray(t0), jnp.asarray(r0)))
                   - want) < TOL
        r1, t1 = 1 / 3, np.pi / 7
        want = np.sin((r1 * np.cos(t1)) * (r1 * np.sin(t1) - 0.1))
        assert abs(float(f(jnp.asarray(t1), jnp.asarray(r1)))
                   - want) < TOL

    def test_r_slices(self):
        # pass(4)/(5): f(th, :) as chebfuns in r.
        f = Diskfun.from_function(
            lambda t, r: jnp.exp(r ** 2 * jnp.sin(t) * jnp.cos(t)))
        for th0 in (0.2, 1.7):
            sl = f.slice_theta(th0)
            assert _maxdiff(
                sl, lambda r, _t=th0: jnp.exp(
                    r ** 2 * jnp.sin(_t) * jnp.cos(_t)),
                (0.0, 1.0)) < TOL

    def test_theta_slices(self):
        # pass(6): f(:, r) as trig chebfuns in theta.
        f = Diskfun.from_function(
            lambda t, r: jnp.exp(r ** 2 * jnp.sin(t) * jnp.cos(t)))
        for r0 in (0.2, 0.7):
            sl = f.slice_r(r0)
            assert _maxdiff(
                sl, lambda t, _r=r0: jnp.exp(
                    _r ** 2 * jnp.sin(t) * jnp.cos(t)),
                (-np.pi, np.pi)) < TOL

    def test_negative_r_slice(self):
        # pass(7): f(:, -r) equals f(:, r) for this pi-periodic f
        # (the BMC identity f(th, -r) = f(th + pi, r)).
        f = Diskfun.from_function(
            lambda t, r: jnp.exp(r ** 2 * jnp.sin(t) * jnp.cos(t)))
        a = f.slice_r(0.7)
        b = f.slice_r(-0.7)
        ts = jnp.linspace(-np.pi + 1e-9, np.pi - 1e-9, 41)
        assert float(jnp.max(jnp.abs(jnp.asarray(a(ts))
                                     - jnp.asarray(b(ts))))) < TOL

    def test_get_properties(self):
        # pass(8)-(10): rows/cols slice data and the domain.
        f = Diskfun.from_function(lambda t, r: r * jnp.cos(t))
        # rank-1: the row is +-cos(t), the col +-r (up to pivots).
        ts = jnp.linspace(-np.pi + 1e-9, np.pi - 1e-9, 21)
        row = jnp.asarray(f.rows[0]
                          (ts / np.pi * 1.0))  # trigtech on [-1,1]
        want = jnp.cos(ts)
        scale = float(jnp.max(jnp.abs(row)))
        assert (float(jnp.max(jnp.abs(row / scale - want))) < 1e-8
                or float(jnp.max(jnp.abs(row / scale + want))) < 1e-8)
        assert len(f.cols) == len(f.rows) == 1
