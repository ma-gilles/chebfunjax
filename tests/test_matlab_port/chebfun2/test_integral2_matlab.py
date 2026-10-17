"""Port of MATLAB Chebfun tests/chebfun2/test_integral2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_integral2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


class TestChebfun2Integral2:
    def test_odd_integrand_zero(self):
        f = Chebfun2.from_function(lambda x, y: x * y)
        assert abs(float(f.integral2())) < TOL

    def test_subdomain_integral(self):
        # Literal native predicate: retain the default-domain f and rectangle.
        f = Chebfun2.from_function(lambda x, y: x * y)
        assert abs(float(f.integral2([0.0, 1.0, 0.0, 1.0])) - 0.25) < TOL

    def test_stretched_domain(self):
        f = Chebfun2.from_function(lambda x, y: x ** 2 * jnp.cos(y),
                                   domain=(0.0, 3.0, -1.0, 1.0))
        assert abs(float(f.integral2()) - 9 * 2 * np.sin(1.0)) < TOL

    def test_rectangle_jax_array_and_degenerate_line(self):
        f = Chebfun2.from_function(lambda x, y: x * y)
        bounds = jnp.asarray([0.0, 1.0, 0.0, 1.0])
        assert abs(float(f.integral2(bounds)) - 0.25) < TOL

        one = Chebfun2.from_function(lambda x, y: jnp.ones_like(x + y))
        assert abs(float(one.integral2([0.0, 0.0, 0.0, 1.0])) - 1.0) < TOL
        with np.testing.assert_raises_regex(
                UnboundLocalError, "MATLAB:unassignedOutputs"):
            f.integral2([0.0, 0.0, 0.0, 0.0])
        empty = Chebfun2.empty()
        assert float(empty.integral2([0.0, 0.0, 0.0, 1.0])) == 0.0
        assert float(empty.sum2()) == 0.0

    def test_quad2d_matches_rectangle_integral2(self):
        f = Chebfun2.from_function(lambda x, y: x * y)
        expected = f.integral2([0.0, 1.0, 0.0, 1.0])
        assert abs(float(f.quad2d(0.0, 1.0, 0.0, 1.0)) - float(expected)) < TOL

        with np.testing.assert_raises_regex(
                TypeError, "First input argument must be a function handle"):
            f.quad2d(0.0, 0.0, 0.0, 1.0)
        with np.testing.assert_raises_regex(
                TypeError, "First input argument must be a function handle"):
            f.quad2d(0.0, 0.0, 0.0, 0.0)

    def test_quad2d_rejects_bounds_outside_domain(self):
        f = Chebfun2.from_function(lambda x, y: x * y)
        with np.testing.assert_raises_regex(ValueError, "within the SEPARABLEAPPROX's domain"):
            f.quad2d(-2.0, 0.0, 0.0, 1.0)

    def test_integral2_requires_four_rectangle_entries(self):
        f = Chebfun2.from_function(lambda x, y: x * y)
        with np.testing.assert_raises_regex(ValueError, "four corners"):
            f.integral2([0.0, 1.0, 0.0])
        with np.testing.assert_raises_regex(ValueError, "four corners"):
            f.integral2(jnp.asarray([[0.0, 1.0], [0.0, 1.0]]))
        from chebfunjax.chebfun1d.chebfun import chebfun

        c = chebfun(lambda t: t, domain=(0.0, 1.0))
        with np.testing.assert_raises_regex(
                ValueError, "Integration path must be complex-valued"):
            f.integral2(c)
