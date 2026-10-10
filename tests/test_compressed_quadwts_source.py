"""Source-stage observations; tagged provider tests do not qualify FFT accuracy."""
import struct
from fractions import Fraction

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils.quadrature import (
    _clenshaw_curtis_weights,
    _clenshaw_curtis_weights_core,
)


@pytest.mark.parametrize('n', [2, 3, 4, 5])
def test_source_stages(n, monkeypatch):
    records = []
    expected_moments = {
        2: [Fraction(2)],
        3: [Fraction(2), Fraction(-2, 3)],
        4: [Fraction(2), Fraction(-2, 3), Fraction(-2, 3)],
        5: [Fraction(2), Fraction(-2, 3), Fraction(-2, 15), Fraction(-2, 3)],
    }

    def tagged_fft(values):
        records.append(values)
        return jnp.asarray([complex((n-1)*(8+3*k), (n-1)*(2+k)) for k in range(n-1)],
                           dtype=jnp.complex128)

    monkeypatch.setattr(jnp.fft, 'fft', tagged_fft)
    # Call the source body directly so the tagged provider is observable while
    # the division helper's lax loop keeps its ordinary compiled semantics.
    result = _clenshaw_curtis_weights_core.__wrapped__(
        jnp.arange(2, n, 2, dtype=jnp.float64), n)
    assert len(records) == 1
    expected = jnp.asarray([float(q) for q in expected_moments[n]], dtype=jnp.float64)
    wanted = jnp.asarray([4]+[8+3*k for k in range(1, n-1)]+[4], dtype=jnp.float64)
    assert records[0].shape == (n-1,)
    assert result.shape == (n,)
    assert result.dtype == jnp.float64
    assert jax.device_get(records[0]).tobytes() == jax.device_get(expected).tobytes()
    assert jax.device_get(result).tobytes() == jax.device_get(wanted).tobytes()


@pytest.mark.parametrize('n', [5, 10])
def test_exact_normalization(n):
    # Independently form native compressed moments with exact rational arithmetic.
    # Runtime forward FFT supplies represented values only: this checks scaling,
    # not the accuracy or MATLAB bit parity of the FFT provider itself.
    moments = [Fraction(2)] + [Fraction(2, 1-k*k) for k in range(2, n, 2)]
    mirrored = moments + moments[n//2-1:0:-1]
    values = jnp.asarray([float(q) for q in mirrored], dtype=jnp.float64)
    transformed = jnp.conj(jnp.fft.fft(jnp.conj(values)))
    real_bits = jax.device_get(jnp.real(transformed)).tobytes()
    real = struct.unpack('='+str(n-1)+'d', real_bits)
    normalized = [float(Fraction.from_float(x)/(n-1)) for x in real]
    expected = jnp.asarray([normalized[0]/2]+normalized[1:]+[normalized[0]/2],
                           dtype=jnp.float64)
    even = jnp.arange(2, n, 2, dtype=jnp.float64)
    # Eager outer function; internal lax control-flow retains ordinary JAX semantics.
    eager = _clenshaw_curtis_weights_core.__wrapped__(even, n)
    compiled = _clenshaw_curtis_weights(n)
    for value in (eager, compiled):
        assert value.shape == expected.shape and value.dtype == expected.dtype
        assert jax.device_get(value).tobytes() == jax.device_get(expected).tobytes()
