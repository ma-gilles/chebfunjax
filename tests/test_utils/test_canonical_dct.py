"""Canonical DCT numerical adapter controls, Chebfun commit: 7574c77."""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils.transforms import vals2coeffs


@pytest.mark.parametrize('kind', ['real', 'imag', 'zero', 'mixed'])
def test_transverse_jvp(kind):
    x = jnp.arange(17, dtype=jnp.float64)
    real = jnp.cos(x / 9)
    primal = {'real': real + 0j, 'imag': 1j * real,
              'zero': jnp.zeros(17, dtype=jnp.complex128),
              'mixed': real + 1j * jnp.sin(x / 3)}[kind]
    tangent = jnp.exp(x / 17) + 1j * jnp.sin(x / 7)
    _, actual = jax.jvp(vals2coeffs, (primal,), (tangent,))
    expected = vals2coeffs(tangent)
    assert jnp.max(jnp.abs(actual - expected)) < 100 * jnp.finfo(jnp.float64).eps


def test_independent_columns_and_conjugate_pairs():
    x = jnp.arange(33, dtype=jnp.float64)
    v = jnp.cos(x / 3) + 1j * jnp.sin(x / 7)
    values = jnp.stack([v, jnp.conj(v), v.real + 0j, 1j * v.real,
                        jnp.zeros_like(v)], axis=1)
    actual = vals2coeffs(values)
    separate = jnp.stack([vals2coeffs(values[:, i]) for i in range(5)], axis=1)
    assert jnp.array_equal(actual, separate)
    assert jnp.array_equal(actual[:, 0], jnp.conj(actual[:, 1]))
    assert jnp.all(actual[:, 2].imag == 0)
    assert jnp.all(actual[:, 3].real == 0)


def test_nonfinite_not_hidden():
    for value in [jnp.nan, jnp.inf]:
        v = jnp.ones(8, dtype=jnp.complex128).at[2].set(value + 1j)
        assert jnp.any(~jnp.isfinite(vals2coeffs(v)))
