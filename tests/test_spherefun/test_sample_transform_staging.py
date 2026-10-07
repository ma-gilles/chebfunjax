"""Source transform staging controls; no altered sampling or truncation.

Provenance
----------
MATLAB source : @trigtech/coeffs2vals.m, @spherefun/sample.m
Chebfun commit: 7574c77
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.spherefun import _plus
from chebfunjax.tech.trigtech import Trigtech, _trig_coeffs2vals_impl


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("n", [17, 18])
def test_real_transform_modes_and_linear_jvp(disabled, n):
    with jax.disable_jit(disabled):
        c = jnp.zeros((n, 2), dtype=jnp.complex128)
        c = c.at[n // 2].set(jnp.array([0.25, -0.5]))
        c = c.at[n // 2 - 1].set(jnp.array([0.125, 0.25j]))
        c = c.at[n // 2 + 1].set(jnp.array([0.125, -0.25j]))
        x = -1 + 2 * jnp.arange(n) / n
        expected = jnp.stack((0.25 + 0.25 * jnp.cos(jnp.pi * x),
                              -0.5 + 0.5 * jnp.sin(jnp.pi * x)), axis=1)
        actual, tangent = jax.jvp(_plus._real_sample_values, (c,), (2 * c,))
        assert float(jnp.max(jnp.abs(actual - expected))) < 50 * jnp.finfo(jnp.float64).eps
        assert float(jnp.max(jnp.abs(tangent - 2 * expected))) < 100 * jnp.finfo(jnp.float64).eps
        raw = jnp.real(_trig_coeffs2vals_impl(c))
        assert float(jnp.max(jnp.abs(actual - raw))) < 20 * jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("n,m", [(9, 5), (10, 6), (9, 12), (10, 13)])
def test_sample_unchanged_aliasing_and_padding(monkeypatch, disabled, n, m):
    with jax.disable_jit(disabled):
        c = (jnp.arange(n) + 1) / 128 + 1j * jnp.arange(n)[::-1] / 256
        techs = [Trigtech(coeffs=c, is_real=False, ishappy=True),
                 Trigtech(coeffs=-2 * c, is_real=False, ishappy=True)]
        actual = _plus._sample(techs, m)
        monkeypatch.setattr(_plus, "_real_sample_values",
                            lambda z: jnp.real(_trig_coeffs2vals_impl(z)))
        raw = _plus._sample(techs, m)
        assert actual.shape == (m, 2)
        assert float(jnp.max(jnp.abs(actual - raw))) < 100 * jnp.finfo(jnp.float64).eps
