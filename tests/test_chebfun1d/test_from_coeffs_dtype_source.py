"""Coefficient-input dtype and column controls.

Provenance
----------
MATLAB @chebfun/chebfun.m coefficient polynomial contract and parseInputs;
@chebtech/populate.m discrete coefficient assignment.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Independent degree-two polynomials; no numerical oracle or tolerance migration.
"""

import jax
import jax.numpy as jnp
import pytest

from chebfunjax import Chebfun, chebfun


@pytest.mark.parametrize("disable_jit", [False, True])
@pytest.mark.parametrize("shape", ["vector", "singleton", "column", "columns"])
@pytest.mark.parametrize("complex_input", [False, True])
def test_coefficient_polynomial_dtype_shape_and_action(disable_jit, shape, complex_input):
    with jax.disable_jit(disable_jit):
        c = jnp.asarray([1., -.5, .25])
        if complex_input:
            c = c + 1j * jnp.asarray([.5, .25, -.125])
        if shape == "singleton":
            c = c[:1]
        elif shape == "column":
            c = c[:, None]
        elif shape == "columns":
            c = jnp.stack((c, -2 * c), axis=1)
        f = Chebfun.from_coeffs(c, domain=(2., 6.))
        public = chebfun(c, domain=(2., 6.), coeffs=True)
        assert f.coeffs.shape == c.shape
        assert f.coeffs.dtype == (jnp.complex128 if complex_input else jnp.float64)
        assert bool(jnp.array_equal(f.coeffs, c))
        assert bool(jnp.array_equal(f.coeffs, public.coeffs))
        x = jnp.asarray([2., 2.5, 4., 5.5, 6.])
        t = (x - 4.) / 2.
        if c.ndim == 2:
            t = t[:, None]
        expected = jnp.ones_like(t) * c[0]
        if c.shape[0] > 1:
            expected = expected + c[1] * t + c[2] * (2 * t**2 - 1)
        values = f(x)
        assert values.shape == expected.shape
        assert bool(jnp.allclose(values, expected, rtol=0., atol=1e-14))
        assert bool(jnp.allclose(public(x), expected, rtol=0., atol=1e-14))


def test_integer_coefficients_still_promote_to_float64():
    f = Chebfun.from_coeffs([1, 2, 3])
    assert f.coeffs.dtype == jnp.float64
    assert bool(jnp.array_equal(f.coeffs, jnp.asarray([1., 2., 3.])))
