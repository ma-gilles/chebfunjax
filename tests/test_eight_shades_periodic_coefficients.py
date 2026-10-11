"""Public periodic coefficient conversion controls, native Chebfun7574c77."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import jv

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.tech.trigtech import Trigtech


def test_periodic_chebcoeffs_independent_bessel_series():
    f = cj.chebfun(lambda x: jnp.cos(jnp.pi * x), trig=True)
    expected = np.zeros(12)
    expected[0] = jv(0, np.pi)
    for k in range(1, 6):
        expected[2 * k] = 2 * (-1) ** k * jv(2 * k, np.pi)
    np.testing.assert_allclose(f.chebcoeffs(12), expected, rtol=0, atol=2e-14)
    np.testing.assert_array_equal(f.chebcoeffs(12), f.funs[0].tech.chebcoeffs(12))
    with pytest.raises(ValueError, match="CHEBFUN:TRIGTECH:chebcoeffs:input"):
        f.chebcoeffs()
    np.testing.assert_array_equal(f.chebcoeffs(0), np.array([]))
    np.testing.assert_array_equal(f.chebcoeffs(12, jnp.array([])), f.chebcoeffs(12))
    with pytest.raises(ValueError, match="CHEBFUN:CHEBTECH:chebcoeffs:badKind"):
        f.chebcoeffs(12, 3)


def test_scalar_array_complex_and_second_kind_shapes():
    f = cj.chebfun(
        lambda x: jnp.stack((jnp.cos(jnp.pi * x), 1j * jnp.sin(jnp.pi * x)), axis=-1), trig=True
    )
    first = f.chebcoeffs(12)
    second = f.chebcoeffs(10, 2)
    assert first.shape == (12, 2) and second.shape == (10, 2)
    expected = np.asarray(first[:10]).copy()
    expected[1:] /= 2
    expected -= np.asarray(first[2:12]) / 2
    np.testing.assert_allclose(second, expected, rtol=0, atol=2e-14)


def test_cf_periodic_uses_converted_chebyshev_series():
    f = cj.chebfun(lambda x: 2 + jnp.cos(jnp.pi * x), trig=True)
    converted = Chebfun.from_coeffs(f.chebcoeffs(len(f)), (-1.0, 1.0))
    p, q, _, _ = cj.cf(f, 1)
    expected, expectedq, _, _ = cj.cf(converted, 1)
    assert p.isequal(expected) and q.isequal(expectedq)
    assert isinstance(f.funs[0].tech, Trigtech)
