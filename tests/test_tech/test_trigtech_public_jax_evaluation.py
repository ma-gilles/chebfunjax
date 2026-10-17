"""Analytic/no-host controls for public JAX trigonometric evaluation.

Provenance: Chebfun7574c77 @trigtech/feval.m, @trigtech/horner.m.
These controls supplement original MATLAB predicates, without relaxing them.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech import trigtech
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize('n', [3, 4, 17, 18, 1025])
@pytest.mark.parametrize('real', [False, True])
@pytest.mark.parametrize('matrix', [False, True])
def test_public_real_coordinates_use_jax_with_source_odd_even_series(monkeypatch, n, real, matrix):
    def forbidden(*args, **kwargs):
        pytest.fail('public evaluation used the legacy NumPy Horner adapter')
    monkeypatch.setattr(trigtech, '_trig_eval_np', forbidden)
    x = jnp.array([[-1., -.3], [.17, 1.]]) if matrix else jnp.array([-.8, 0., .6])
    mid = n//2
    coeffs = jnp.zeros(n, dtype=jnp.complex128)
    if real:
        coeffs = coeffs.at[mid].set(1.25).at[mid-1].set(.125+.25j).at[mid+1].set(.125-.25j)
        expected = 1.25+.25*jnp.cos(jnp.pi*x)+.5*jnp.sin(jnp.pi*x)
    else:
        coeffs = coeffs.at[mid].set(1.25+.5j).at[mid-1].set(.2-.1j).at[mid+1].set(.125+.25j)
        expected = (1.25+.5j+(.2-.1j)*jnp.exp(-1j*jnp.pi*x)
                    +(.125+.25j)*jnp.exp(1j*jnp.pi*x))
    if n % 2 == 0:
        amplitude = .0625 if real else .125-.0625j
        coeffs = coeffs.at[0].set(amplitude)
        expected = expected+amplitude*jnp.cos((n//2)*jnp.pi*x)
    f = Trigtech.from_coeffs(coeffs, is_real=real)
    actual = f(x)
    assert actual.shape == expected.shape
    assert jnp.iscomplexobj(actual) == (not real)
    # These are additional analytic controls, not an original MATLAB bound.
    # Long complex Horner chains accumulate O(n*eps) roundoff: each complex
    # multiply/add uses <=8 real roundings, plus the degree-dependent phase
    # and final prefactor. Budget32n real roundings conservatively; do not
    # impose the low-degree64eps bound on an arbitrarily long sparse series.
    # The unchanged baseline also violates that initial unscaled bound for
    # n=1025 (saved diagnostic); no original source assertion is changed.
    eps = np.finfo(float).eps
    if not real:
        gamma = (32*n*eps)/(1-32*n*eps)
        bound = gamma*float(jnp.sum(jnp.abs(coeffs)))
    else:
        bound = 64*eps
    np.testing.assert_allclose(actual, expected, rtol=0, atol=bound)


@pytest.mark.parametrize('n', [1, 4, 5])
def test_public_array_coefficients_scalar_jit_vmap_and_derivative(monkeypatch, n):
    def forbidden(*args, **kwargs):
        pytest.fail('public evaluation used NumPy')
    monkeypatch.setattr(trigtech, '_trig_eval_np', forbidden)
    c = jnp.zeros((n, 2), dtype=jnp.complex128).at[n//2].set(jnp.array([2., 3.]))
    scale = jnp.array([1., -.25])
    if n > 1:
        c = c.at[n//2-1].set(.5*scale).at[n//2+1].set(.5*scale)
    f = Trigtech.from_coeffs(c, is_real=True)
    def exact(x):
        return jnp.array([2., 3.])+(scale*jnp.cos(jnp.pi*x) if n>1 else 0.)
    x = jnp.array([-.4, .2, .8])
    np.testing.assert_allclose(f(.2), exact(.2), rtol=0, atol=16*np.finfo(float).eps)
    expected = jax.vmap(exact)(x)
    np.testing.assert_allclose(jax.jit(lambda t: f(t))(x), expected, rtol=0, atol=16*np.finfo(float).eps)
    np.testing.assert_allclose(jax.vmap(lambda t: f(t))(x), expected, rtol=0, atol=16*np.finfo(float).eps)
    actual_d = jax.jvp(lambda t: f(t), (x,), (jnp.ones_like(x),))[1]
    expected_d = -jnp.pi*jnp.sin(jnp.pi*x[:, None])*scale if n>1 else jnp.zeros_like(expected)
    np.testing.assert_allclose(actual_d, expected_d, rtol=0, atol=32*np.finfo(float).eps)
