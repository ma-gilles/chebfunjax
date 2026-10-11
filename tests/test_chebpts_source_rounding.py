"""Preserve source-rounded grid arguments across CPU compiler reassociation.

Chebfun 7574c77, @chebtech1/chebpts.m and @chebtech2/chebpts.m.
Copyright 2017 by The University of Oxford and The Chebfun Developers.
First-kind source divides before multiplying; second-kind multiplies first.
The earlier Native4000 capture covered the second-kind argument ordering.
This checks argument rounding independently of the selected sine backend.
Second-kind nodes use the fdlibm adapter introduced for native POWER parity;
first-kind nodes retain JAX sine. MATLAB sine parity is qualified separately.
"""
import math
import struct

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils._binary64 import _divide_binary64_by_positive_integer
from chebfunjax.utils._fdlibm_sine import _sin_cheb_angles
from chebfunjax.utils.quadrature import chebpts


def _bits(x):
    return b"".join(struct.pack("=d", v) for v in jax.device_get(x).tolist())


@pytest.mark.parametrize("kind", [1, 2])
@pytest.mark.parametrize("n", [2, 3, 17, 64, 129, 4000])
def test_source_rounded_grid_arguments(n, kind):
    """Literal source binary64 operation order with the selected sine backend."""
    m = n if kind == 1 else n - 1
    start = -n + 1 if kind == 1 else -m
    stop = n if kind == 1 else m + 1
    expected = jnp.asarray([math.pi * (k / (2 * m)) if kind == 1
                            else math.pi * k / (2 * m)
                            for k in range(start, stop, 2)])

    @jax.jit
    def instrument():
        k = jnp.arange(start, stop, 2, dtype=jnp.float64)
        angles = (jnp.pi * _divide_binary64_by_positive_integer(k, 2 * m)
                  if kind == 1 else
                  _divide_binary64_by_positive_integer(jnp.pi * k, 2 * m))
        return angles

    angles = instrument()
    assert _bits(angles) == _bits(expected)
    actual = chebpts(n, kind)
    sine = jnp.sin if kind == 1 else _sin_cheb_angles
    assert _bits(actual) == _bits(sine(expected))
