"""Preserve source-rounded grid arguments across CPU compiler reassociation.

Chebfun 7574c77, @chebtech1/chebpts.m and @chebtech2/chebpts.m.
Copyright 2017 by The University of Oxford and The Chebfun Developers.
Native4000 capture independently verifies the binary64 argument ordering.
Sine backend bit identity with MATLAB is deliberately not asserted.
"""
import math
import struct

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils._binary64 import _divide_binary64_by_positive_integer
from chebfunjax.utils.quadrature import chebpts


def _bits(x):
    return b"".join(struct.pack("=d", v) for v in jax.device_get(x).tolist())


@pytest.mark.parametrize("kind", [1, 2])
@pytest.mark.parametrize("n", [2, 3, 17, 64, 129, 4000])
def test_source_rounded_grid_arguments(n, kind):
    """Scalar binary64 multiplication then division, with unchanged JAX sine."""
    m = n if kind == 1 else n - 1
    start = -n + 1 if kind == 1 else -m
    stop = n if kind == 1 else m + 1
    expected = jnp.asarray([math.pi * k / (2 * m) for k in range(start, stop, 2)])

    @jax.jit
    def instrument():
        k = jnp.arange(start, stop, 2, dtype=jnp.float64)
        angles = _divide_binary64_by_positive_integer(jnp.pi * k, 2 * m)
        return angles, jnp.sin(angles)

    angles, instrumented_nodes = instrument()
    assert _bits(angles) == _bits(expected)
    actual = chebpts(n, kind)
    assert _bits(instrumented_nodes) == _bits(actual)
    assert _bits(actual) == _bits(jnp.sin(expected))
