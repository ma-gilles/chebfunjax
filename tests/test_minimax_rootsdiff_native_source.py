"""Independent panel extrema controls for native minimax.m rootsdiff."""

import jax.numpy as jnp
import numpy as np

from chebfunjax.utils._minimax_rootsdiff_source import rootsdiff


def test_native_keeps_endpoint_derivative_roots():
    x = jnp.cos(jnp.pi * jnp.arange(8, -1, -1) / 8)
    result = rootsdiff(x**3 - 3 * x, (-1.0, 1.0), lambda z: z**3 - 3 * z)
    np.testing.assert_allclose(jnp.sort(result), [-1.0, 1.0], rtol=0, atol=4e-15)


def test_native_T_tail_resampling_and_seven_exact_extrema():
    x = jnp.cos(jnp.pi * jnp.arange(8, -1, -1) / 8)
    seen = []

    def error(z):
        seen.append(z.size)
        return 1 + jnp.cos(8 * jnp.arccos(z))

    result = rootsdiff(1 + jnp.cos(8 * jnp.arccos(x)), (-1.0, 1.0), error)
    assert seen == [17]
    np.testing.assert_allclose(
        jnp.sort(result), jnp.sort(jnp.cos(jnp.pi * jnp.arange(1, 8) / 8)), rtol=0, atol=2e-14
    )


def test_native_constant_derivative_has_no_candidate_extrema():
    x = jnp.cos(jnp.pi * jnp.arange(8, -1, -1) / 8)
    assert rootsdiff(1 + x, (-1.0, 1.0), lambda z: 1 + z).size == 0
