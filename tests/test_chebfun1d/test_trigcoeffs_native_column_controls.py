"""Native Fourier column and even Nyquist contracts.

MATLAB source: @trigtech/trigcoeffs.m, @chebfun/trigcoeffs.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj


@pytest.mark.parametrize("domain", [(-1.0, 1.0), (0.0, 2.0)])
def test_even_nyquist_columns_and_phase(domain):
    a, b = domain
    f = cj.chebfun(
        lambda x: jnp.stack(
            [jnp.exp(1j * jnp.pi * (x - a - 1)), jnp.exp(2j * jnp.pi * (x - a - 1))], axis=-1
        ),
        domain=domain,
        trig=True,
    )
    out = np.asarray(f.trigcoeffs(4))
    expected = np.zeros((4, 2), complex)
    modes = np.arange(-2, 2)
    expected[3, 0] = 1
    expected[0, 1] = 1
    expected *= np.exp(-1j * modes * jnp.pi * (a + 1))[:, None]
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out, expected, rtol=0, atol=10 * np.finfo(float).eps)


def test_even_nyquist_scalar():
    f = cj.chebfun(lambda x: jnp.exp(2j * jnp.pi * x), trig=True)
    expected = np.array([1, 0, 0, 0], complex)
    np.testing.assert_allclose(
        np.asarray(f.trigcoeffs(4)), expected, rtol=0, atol=10 * np.finfo(float).eps
    )


@pytest.mark.parametrize("n", [0, -1])
def test_native_nonpositive_length(n):
    from chebfunjax.tech.trigtech import Trigtech

    tech = Trigtech(coeffs=jnp.zeros((3, 2), dtype=jnp.complex128), is_real=False)
    assert tech.trigcoeffs(n).shape == (0,)


@pytest.mark.parametrize("n", [4, 8])
def test_literal_helper_jit_static_length(n):
    import jax

    from chebfunjax.tech.trigtech import _trigcoeffs_trigtech

    coeffs = jnp.asarray([[1, 2], [3, 4], [5, 6]], dtype=jnp.complex128)
    actual = jax.jit(lambda c: _trigcoeffs_trigtech(c, n))(coeffs)
    expected = np.zeros((n, 2), complex)
    center = n // 2
    expected[center - 1 : center + 2] = np.asarray(coeffs)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_zero_empty_coefficients():
    from chebfunjax.tech.trigtech import Trigtech

    empty = Trigtech(coeffs=jnp.empty((0,), dtype=jnp.complex128), is_real=False)
    np.testing.assert_array_equal(np.asarray(empty.trigcoeffs(3)), np.zeros(3, complex))
