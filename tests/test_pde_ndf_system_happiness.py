"""Native weighted check can differ from checking every component."""

import jax.numpy as jnp

from chebfunjax.chebfun1d._pde_ndf.adaptive import scalar_happiness, system_happiness
from chebfunjax.utils.quadrature import chebpts


def test_single_column_is_identical_to_qualified_scalar():
    x = chebpts(33)
    values = jnp.exp(x)
    assert system_happiness(values[:, None], 1e-6) == scalar_happiness(values, 1e-6)


def test_source_weighted_cancellation_not_all_column_happiness():
    x = chebpts(17)
    mode = jnp.cos(16 * jnp.arccos(x))
    c = 1 + jnp.sin(jnp.arange(1, 3, dtype=jnp.float64))
    values = jnp.column_stack((1 + mode, 1 - c[0] / c[1] * mode))
    assert not scalar_happiness(values[:, 0], 1e-6)[0]
    assert system_happiness(values, 1e-6)[0]
