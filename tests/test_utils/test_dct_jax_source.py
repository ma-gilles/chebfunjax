"""JAX-only Chebtech2 transform contracts, Chebfun7574c77 DCT-I formulas."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.transforms import coeffs2vals, vals2coeffs


@pytest.mark.parametrize('n', [0, 1, 2, 17])
@pytest.mark.parametrize('kind', ['real', 'imag', 'complex'])
def test_components_and_conjugacy_without_host_fft(monkeypatch, n, kind):
    def forbidden(*args, **kwargs):
        raise AssertionError('host FFT used')
    monkeypatch.setattr(np.fft, 'fft', forbidden)
    monkeypatch.setattr(np.fft, 'ifft', forbidden)
    base = jnp.arange(n*2, dtype=jnp.float64).reshape(n, 2)/13
    value = base if kind == 'real' else 1j*base if kind == 'imag' else base+1j*base[:, ::-1]
    for fn in [vals2coeffs, coeffs2vals]:
        result = fn(value)
        assert result.shape == value.shape
        assert bool(jnp.array_equal(jnp.conj(result), fn(jnp.conj(value))))
        if kind == 'real':
            assert bool(jnp.all(jnp.imag(result) == 0))
        if kind == 'imag':
            assert bool(jnp.all(jnp.real(result) == 0))


@pytest.mark.parametrize('n', [2, 8, 17])
def test_dense_cosine_reference_and_jvp(n):
    coefficients = jnp.sin(jnp.arange(n, dtype=jnp.float64)) + 1j*jnp.cos(jnp.arange(n, dtype=jnp.float64))
    angles = jnp.arange(n-1, -1, -1)*jnp.pi/(n-1)
    reference = jnp.cos(angles[:, None]*jnp.arange(n)[None, :]) @ coefficients
    values = coeffs2vals(coefficients)
    bound = 100*jnp.finfo(jnp.float64).eps*jnp.max(jnp.abs(reference))
    assert float(jnp.max(jnp.abs(values-reference))) < float(bound)
    back = vals2coeffs(values)
    assert float(jnp.max(jnp.abs(back-coefficients))) < 100*jnp.finfo(jnp.float64).eps
    tangent = jnp.exp(jnp.arange(n, dtype=jnp.float64)/n)
    _, derivative = jax.jvp(lambda c: vals2coeffs(coeffs2vals(c)), (coefficients,), (tangent.astype(jnp.complex128),))
    assert float(jnp.max(jnp.abs(derivative-tangent))) < 100*jnp.finfo(jnp.float64).eps
    batched = jax.vmap(coeffs2vals)(jnp.stack([coefficients, 2*coefficients]))
    assert float(jnp.max(jnp.abs(batched-jnp.stack([values, 2*values])))) < float(bound)
