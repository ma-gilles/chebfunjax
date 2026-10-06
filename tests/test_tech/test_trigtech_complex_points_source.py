"""Independent analytic complex-point controls for source Fourier evaluation.

Provenance: @trigtech/feval.m and @trigtech/horner.m, Chebfun7574c77.
These additional controls use independently stated odd/even Fourier identities;
they do not replace or relax any original MATLAB assertion.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize('series', ['odd-real', 'odd-complex', 'even-real', 'even-complex'])
@pytest.mark.parametrize('compiled', [False, True], ids=['eager', 'jit'])
@pytest.mark.parametrize('vector', [False, True], ids=['scalar', 'vector'])
def test_complex_coordinates_follow_independent_source_horner_identity(series, compiled, vector):
    if series == 'odd-real':
        coeffs = [0.5, 0.0, 0.5]
        real = True
        def exact(z):
            return jnp.cos(jnp.pi*z)
    elif series == 'odd-complex':
        coeffs = [0.0, 1j, 1.0]
        real = False
        def exact(z):
            return 1j+jnp.exp(1j*jnp.pi*z)
    elif series == 'even-real':
        # The source even-N Nyquist coefficient is a cosine, not a
        # one-sided negative exponential.
        coeffs = [1.25, 0.0, 0.0, 0.0]
        real = True
        def exact(z):
            return 1.25*jnp.cos(2*jnp.pi*z)
    else:
        coeffs = [0.2+0.3j, 0.0, 1j, 1.0]
        real = False
        def exact(z):
            return (0.2+0.3j)*jnp.cos(2*jnp.pi*z)+1j+jnp.exp(1j*jnp.pi*z)

    f = Trigtech.from_coeffs(jnp.asarray(coeffs, dtype=jnp.complex128), is_real=real)
    z = jnp.asarray([0.2+0.3j, -0.4-0.2j] if vector else 0.2+0.3j, dtype=jnp.complex128)
    actual = jax.jit(lambda point: f(point))(z) if compiled else f(z)
    expected = exact(z)
    assert actual.shape == expected.shape
    assert jnp.issubdtype(actual.dtype, jnp.complexfloating)
    bound = 200*jnp.finfo(jnp.float64).eps*jnp.maximum(1.0, jnp.abs(expected))
    assert bool(jnp.all(jnp.abs(actual-expected) <= bound))
