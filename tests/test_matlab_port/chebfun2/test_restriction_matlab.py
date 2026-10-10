"""Port of MATLAB Chebfun tests/chebfun2/test_restriction.m (Fable 5).

The test preserves the four native assertions and original bound. MATLAB's
explicit subsref struct is exercised through Chebfun2's documented
``g(x, ':')`` adapter.

Provenance
----------
MATLAB source : tests/chebfun2/test_restriction.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun2d.chebfun2 import Chebfun2

TOL = 1e5 * np.finfo(float).eps


def _native_function(x, y):
    return jnp.exp(-10 * (x ** 2 + y ** 2))


class TestChebfun2Restriction:
    def test_all_matlab_assertions(self):
        # Native pass(1): restrict to a point.
        g = Chebfun2.from_function(_native_function)
        value = g.restrict((0, 0, 0, 0))
        assert abs(value - _native_function(0.0, 0.0)) < TOL

        # Native pass(2): restrict to a line and evaluate at zero.
        g = Chebfun2.from_function(_native_function)
        line = g.restrict((0, 0, -0.9, 0.1))
        value = line(0.0)
        assert abs(value - _native_function(0.0, 0.0)) < TOL

        # Native pass(3): continuous Chebfun norm for the complex path.
        g = Chebfun2.from_function(_native_function)
        path = cj.chebfun(lambda t: t + 1e-18j)
        value = g.restrict(path)
        exact = cj.chebfun(
            lambda x: _native_function(x, 1e-18), domain=(-1.0, 1.0))
        assert (value - exact).norm() < TOL

        # Native pass(4): subsref(g, {pi/6, ':'}) continuous norm.
        g = Chebfun2.from_function(_native_function)
        value = g(np.pi / 6, ":")
        exact = cj.chebfun(
            lambda y: _native_function(np.pi / 6, y), domain=(-1.0, 1.0))
        assert (value - exact).norm() < TOL

    def test_nondegenerate_restriction_retains_native_factor_state(self):
        f = Chebfun2.from_function(
            lambda x, y: jnp.exp(x * y) + x * y,
            domain=(-1.0, 1.0, -1.0, 1.0))
        full = f.restrict(f.domain)
        assert full is not f
        assert full.domain == f.domain
        assert full.approx.techs == f.approx.techs
        assert full.approx.pivot_locations == f.approx.pivot_locations
        np.testing.assert_array_equal(full.approx.pivots, f.approx.pivots)
        np.testing.assert_allclose(
            full(jnp.asarray([-0.7, 0.0, 0.8]),
                 jnp.asarray([-0.6, 0.2, 0.9])),
            f(jnp.asarray([-0.7, 0.0, 0.8]),
              jnp.asarray([-0.6, 0.2, 0.9])), rtol=0, atol=2e-13)

        dom = (-0.7, 0.8, -0.6, 0.9)
        restricted = f.restrict(dom)
        assert restricted.domain == dom
        assert len(restricted.approx.cols) == len(f.approx.cols)
        assert len(restricted.approx.rows) == len(f.approx.rows)
        assert restricted.approx.techs == f.approx.techs
        assert restricted.approx.pivot_locations == f.approx.pivot_locations
        np.testing.assert_array_equal(restricted.approx.pivots, f.approx.pivots)
        np.testing.assert_array_equal(
            restricted.approx.pivot_values, f.approx.pivot_values)
        xs = jnp.asarray([-0.55, 0.1, 0.72])
        ys = jnp.asarray([-0.45, 0.2, 0.77])
        np.testing.assert_allclose(restricted(xs, ys), f(xs, ys), rtol=0, atol=2e-13)

    def test_point_and_line_restriction_do_not_reconstruct_2d(self, monkeypatch):
        f = Chebfun2.from_function(
            lambda x, y: jnp.exp(1j * (x + 2 * y)),
            domain=(-1.0, 1.0, -1.0, 1.0))
        point = f.restrict((0.2, 0.2, -0.3, -0.3))
        assert jnp.ndim(point) == 0
        assert jnp.iscomplexobj(point)
        assert jnp.abs(point - jnp.exp(1j * (0.2 - 0.6))) < 1e-12

        def no_adaptive_constructor(cls, *args, **kwargs):
            raise AssertionError("numeric restriction must reuse existing factors")

        monkeypatch.setattr(
            Chebfun2, "from_function", classmethod(no_adaptive_constructor))
        vertical = f.restrict((0.2, 0.2, -0.8, 0.7))
        horizontal = f.restrict((-0.8, 0.7, -0.3, -0.3))
        rectangle = f.restrict((-0.7, 0.8, -0.6, 0.9))
        points = jnp.asarray([-0.5, 0.0, 0.4])
        np.testing.assert_allclose(
            vertical(points), jnp.exp(1j * (0.2 + 2 * points)), rtol=0, atol=2e-12)
        np.testing.assert_allclose(
            horizontal(points), jnp.exp(1j * (points - 0.6)), rtol=0, atol=2e-12)
        np.testing.assert_allclose(
            rectangle(0.3, 0.4), jnp.exp(1j * (0.3 + 0.8)), rtol=0, atol=2e-12)

    def test_mixed_periodic_and_polynomial_axes_follow_native_cast(self):
        f = Chebfun2.from_function(
            lambda x, y: jnp.cos(jnp.pi * x) * jnp.exp(y),
            domain=(-1.0, 1.0, -1.0, 1.0), trigx=True)
        out = f.restrict((-0.8, 0.75, -0.6, 0.9))
        # @chebfun/restrict casts periodic Chebfuns to regular Chebfuns
        # before restricting; the resulting separable axes are Chebtechs.
        assert out.approx.techs == ("cheb", "cheb")
        assert all(type(t).__name__ != "Trigtech" for t in out.approx.rows)
        assert all(type(t).__name__ != "Trigtech" for t in out.approx.cols)
        xs = jnp.asarray([-0.7, 0.0, 0.7])
        ys = jnp.asarray([-0.3, 0.2, 0.7])
        np.testing.assert_allclose(out(xs, ys), f(xs, ys), rtol=0, atol=3e-12)

    def test_full_domain_restriction_casts_periodic_factor(self):
        f = Chebfun2.from_function(
            lambda x, y: jnp.cos(jnp.pi * x) * jnp.exp(y),
            domain=(-1.0, 1.0, -1.0, 1.0), trigx=True)
        out = f.restrict(f.domain)
        # @chebfun/restrict casts a periodic Chebfun factor before its
        # full-domain early return, so separableApprox receives Chebtech.
        assert out is not f
        assert out.domain == f.domain
        assert out.approx.techs == ("cheb", "cheb")
        assert all(type(t).__name__ != "Trigtech" for t in out.approx.rows)
        assert all(type(t).__name__ != "Trigtech" for t in out.approx.cols)
        xs = jnp.asarray([-0.8, -0.1, 0.6])
        ys = jnp.asarray([-0.7, 0.2, 0.9])
        np.testing.assert_allclose(out(xs, ys), f(xs, ys), rtol=0, atol=3e-12)

    def test_trigtech_restrict_preserves_periodicity_contract(self):
        from chebfunjax.tech.trigtech import Trigtech

        periodic = Trigtech.from_function(lambda x: jnp.cos(2 * jnp.pi * x))
        assert periodic.restrict(-1.0, 1.0) is periodic
        restricted = periodic.restrict(-0.5, 0.5)
        assert isinstance(restricted, Trigtech)
        assert restricted.ishappy
        points = jnp.asarray([-0.7, -0.2, 0.4, 0.8])
        np.testing.assert_allclose(
            restricted(points), jnp.cos(jnp.pi * points), rtol=0, atol=2e-12)

        nonperiodic = Trigtech.from_function(lambda x: jnp.cos(jnp.pi * x))
        assert nonperiodic.ishappy
        with np.testing.assert_raises_regex(
                ValueError, "CHEBFUN:TRIGTECH:restrict:notPeriodic"):
            nonperiodic.restrict(-0.5, 0.5)
