"""Analytical Horner/zero-predicate controls for persistent staging only.

Provenance
----------
MATLAB source : @trigtech/horner.m, @separableApprox/iszero.m
Chebfun commit: 7574c77
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun._plus import _iszero, _make, _real_factor_values, _tech


@pytest.mark.parametrize('disabled', [False, True])
@pytest.mark.parametrize('n', [17, 18])
def test_real_horner_analytic_values_and_derivative(disabled, n):
    c = jnp.zeros(n, dtype=jnp.complex128).at[n//2].set(.25)
    c = c.at[n//2-1].set(.5).at[n//2+1].set(.5)
    x = jnp.asarray([-1., -.41, 0., .27, 1.])
    with jax.disable_jit(disabled):
        y = _real_factor_values(c, x)
        derivative = jax.jvp(lambda z: _real_factor_values(c, z),
                             (x,), (jnp.ones_like(x),))[1]
    np.testing.assert_allclose(y, .25+jnp.cos(jnp.pi*x), rtol=0, atol=50*np.finfo(float).eps)
    np.testing.assert_allclose(derivative, -jnp.pi*jnp.sin(jnp.pi*x),
                               rtol=0, atol=50*np.finfo(float).eps)


@pytest.mark.parametrize('disabled', [False, True])
def test_source_exact_zero_predicate_retains_small_nonzero(disabled):
    one = _tech(jnp.ones(1))
    with jax.disable_jit(disabled):
        zero = _make([_tech(jnp.zeros(1))], [one], [1.], 1)
        small = _make([_tech(jnp.asarray([2.**-32]))], [one], [1.], 1)
        assert _iszero(zero)
        assert not _iszero(small)


@pytest.mark.parametrize('disabled', [False, True])
def test_nonconstant_cancellation_preserves_source_structural_fallback(disabled):
    c = jnp.zeros(17, dtype=jnp.complex128).at[7].set(.5).at[9].set(.5)
    factor = _tech(c)
    one = _tech(jnp.ones(1))
    x = jnp.linspace(-1., 1., 10)
    with jax.disable_jit(disabled):
        a = _real_factor_values(c, x)
        b = _real_factor_values(c, x)
        np.testing.assert_array_equal(a-b, jnp.zeros_like(a))
        cancelled = _make([factor, factor], [one, one], [1., -1.], 2)
        perturbed = _make([factor, _tech(c.at[8].set(2.**-32))],
                          [one, one], [1., -1.], 2)
        # Source sampled zero falls through to structural factor checks.
        # Neither factor collection is identically zero, so both returnFalse.
        assert not _iszero(cancelled)
        assert not _iszero(perturbed)
