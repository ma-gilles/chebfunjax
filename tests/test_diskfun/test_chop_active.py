"""Preserve existing source standardChop at buffer and plateau boundaries."""

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.diskfun._chop_active import _standard_chop_active
from chebfunjax.utils.misc import _standard_chop_kernel

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu" or not jax.config.x64_enabled,
    reason="Disk cutoff reuse qualified on CPU binary64",
)


@pytest.mark.parametrize("n", [17, 32, 33, 65, 129])
def test_active_length_preserves_source_cutoff(n):
    k = jnp.arange(n, dtype=jnp.float64)
    eps = jnp.finfo(jnp.float64).eps
    cases = [
        (jnp.zeros(n), eps),
        (jnp.ones(n), eps),
        (jnp.where(k < n // 2, 0.7**k, eps * (1 + k / n)), eps),
        (jnp.exp(-k), jnp.finfo(jnp.float64).tiny),
    ]
    capacity = 1 << (n - 1).bit_length()
    for values, tol in cases:
        expected = _standard_chop_kernel(values, jnp.asarray(tol))
        padded = jnp.pad(values, (0, capacity - n))
        actual = _standard_chop_active(padded, jnp.asarray(tol), jnp.asarray(n))
        assert int(actual) == int(expected)
