"""Port of MATLAB Chebfun tests/ballfun/test_plus.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_plus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS, X0, Y0, val


class TestBallfunPlus:
    def test_plus_functions_and_scalar(self):
        f = Ballfun.from_function(lambda x, y, z: x)
        g = Ballfun.from_function(lambda x, y, z: y * y)
        assert abs(val(f + g) - (X0 + Y0 ** 2)) < 1e3 * EPS
        assert abs(val(f + 2.0) - (X0 + 2)) < 1e3 * EPS
        assert abs(val(2.0 + f) - (X0 + 2)) < 1e3 * EPS


class TestNativePlusPredicates:
    def test_numeric_values(self):
        f = Ballfun.from_values(jnp.ones((21, 20, 22)))
        v1 = Ballfun.coeffs2vals(f.coeffs)
        v2 = Ballfun.coeffs2vals((f + f).coeffs)
        assert jnp.max(jnp.abs(v2.reshape(-1) - 2*v1.reshape(-1))) < 1e4*EPS

    def test_scalar_right(self):
        f = Ballfun.from_function(lambda x, y, z: x**2*jnp.cos(y)-1)
        exact = Ballfun.from_function(lambda x, y, z: x**2*jnp.cos(y))
        assert (f + 1 - exact).norm() < 1e4*EPS

    def test_scalar_left(self):
        f = Ballfun.from_function(lambda x, y, z: y*jnp.sin(z))
        exact = Ballfun.from_function(lambda x, y, z: y*jnp.sin(z)+3)
        assert (3 + f - exact).norm() < 1e4*EPS

    def test_scalar_both_sides(self):
        f = Ballfun.from_function(lambda x, y, z: x*jnp.sin(z)**2*jnp.cos(y))
        exact = Ballfun.from_function(lambda x, y, z: x*jnp.sin(z)**2*jnp.cos(y)+5)
        assert (3 + f + 2 - exact).norm() < 1e4*EPS

    def test_constant_callable(self):
        f = Ballfun.from_function(lambda r, lam, th: 1)
        exact = Ballfun.from_function(lambda r, lam, th: 2)
        assert (f + f - exact).norm() < 1e4*EPS

    def test_zero_plus_scalar(self):
        f = Ballfun.from_function(lambda x, y, z: 0) + 1
        exact = Ballfun.from_function(lambda x, y, z: 1)
        assert (f - exact).norm() < 1e4*EPS


class TestNativePlusEdgeContracts:
    # Direct source contracts from @ballfun/plus.m and constructor.m.
    def test_exact_maximum_shape(self):
        a = jnp.zeros((2, 3, 3), dtype=jnp.complex128).at[0, 1, 1].set(1)
        b = jnp.zeros((1, 1, 1), dtype=jnp.complex128).at[0, 0, 0].set(2)
        result = Ballfun.from_coeffs(a) + Ballfun.from_coeffs(b)
        assert result.shape == (2, 3, 3)
        assert jnp.array_equal(result.coeffs, a.at[0, 1, 1].add(2))

    def test_complex_scalar(self):
        f = Ballfun.from_coeffs(jnp.ones((1, 1, 1)))
        result = f + 2j
        assert not result.is_real
        assert abs(result.feval(0., 0., 0.) - (1+2j)) < 1e4*EPS

    def test_complex_cancellation(self):
        f = Ballfun.from_coeffs(jnp.full((1, 1, 1), 1+2j), is_real=False)
        g = Ballfun.from_coeffs(jnp.full((1, 1, 1), 2-2j), is_real=False)
        assert (f + g).is_real
        assert (f - 2j).is_real

    def test_empty_identity(self):
        e = Ballfun.empty()
        f = Ballfun.from_coeffs(jnp.ones((1, 1, 1)))
        assert e + f is f
        assert f + e is f
        assert (e + e).isempty()
        assert abs((e + 2j).feval(0., 0., 0.) - 2j) < 1e4*EPS
        assert abs((3 + e).feval(0., 0., 0.) - 3) < 1e4*EPS
        small = e + 2e-10j
        assert not small.is_real
        assert abs(small.feval(0., 0., 0.) - 2e-10j) < 1e4*EPS


    def test_compiled_complex_evaluation(self):
        import jax
        def evaluate(c):
            f = Ballfun.from_coeffs(c)
            return (f + 2j).feval(0., 0., 0.)
        c = jnp.ones((1, 1, 1), dtype=jnp.complex128)
        assert abs(jax.jit(evaluate)(c) - (1+2j)) < 1e4*EPS
