"""Port of MATLAB Chebfun tests/ballfun/test_times.m (Fable 5).

FIXED (Fable 5): Ballfun*Ballfun product exercised by the port.

Provenance
----------
MATLAB source : tests/ballfun/test_times.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS, X0, Y0, val

TOL = 1e4 * EPS


class TestBallfunTimes:
    def test_all_matlab_assertions(self):
        # Constants: 2 * 3 = 6.
        f = Ballfun.from_values(2*jnp.ones((21, 12, 22)))
        g = Ballfun.from_values(3*jnp.ones((23, 20, 24)))
        h = Ballfun.from_values(6*jnp.ones((25, 20, 10)))
        assert (f * g - h).norm() < TOL
        assert (g * f - h).norm() < TOL

    def test_pointwise_product(self):
        f = Ballfun.from_function(lambda x, y, z: x)
        g = Ballfun.from_function(lambda x, y, z: y)
        assert abs(val(f * g) - X0 * Y0) < 1e3 * EPS

    def test_scalar_times(self):
        f = Ballfun.from_function(lambda x, y, z: x)
        assert abs(val(3 * f) - 3 * X0) < 1e3 * EPS
        assert abs(val(f * 3) - 3 * X0) < 1e3 * EPS


class TestNativeTimesContracts:
    def test_exact_sum_dimensions(self):
        f = Ballfun.from_coeffs(jnp.ones((1, 1, 1)))
        g = Ballfun.from_coeffs(2*jnp.ones((1, 1, 1)))
        h = f*g
        assert h.shape == (2, 2, 2)
        assert abs(h.feval(.2, .1, .3)-2) < TOL

    def test_complex_product_realness(self):
        f = Ballfun.from_coeffs(1j*jnp.ones((1, 1, 1)), is_real=False)
        assert (f*f).is_real
        assert (f*1j).is_real
        assert abs((f*f).feval(.2, .1, .3)+1) < TOL
        assert abs((f*1j).feval(.2, .1, .3)+1) < TOL

    def test_even_nyquist_prolongation(self):
        # Native trigtech.alias splits an even-grid Nyquist coefficient.
        c = jnp.zeros((1, 2, 1), dtype=jnp.complex128).at[0, 0, 0].set(1)
        f = Ballfun.from_coeffs(c, is_real=False)
        lam = jnp.array([.2, .7, 1.2])
        got = (f*f).feval(jnp.full_like(lam, .5), lam,
                          jnp.full_like(lam, .8), coord='spherical')
        assert jnp.max(jnp.abs(got-jnp.cos(lam)**2)) < TOL

    def test_compiled_product(self):
        import jax
        def product(c):
            f = Ballfun.from_coeffs(c)
            return (f*f).feval(.2, .1, .3)
        assert abs(jax.jit(product)(jnp.ones((1, 1, 1)))-1) < TOL


    def test_even_theta_nyquist_evaluation(self):
        import jax
        c = jnp.zeros((1, 1, 2), dtype=jnp.complex128).at[0, 0, 0].set(1j)
        f = Ballfun.from_coeffs(c, is_real=False)
        th = jnp.array([.2, .7, 1.2])
        evaluate = lambda t: f.fevalm(jnp.array([.5]), jnp.array([.3]), t)[0, 0]
        assert jnp.max(jnp.abs(evaluate(th) - 1j*jnp.cos(th))) < TOL
        assert jnp.max(jnp.abs(jax.jit(evaluate)(th) - 1j*jnp.cos(th))) < TOL
