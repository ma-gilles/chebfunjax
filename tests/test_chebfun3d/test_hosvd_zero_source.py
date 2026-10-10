"""Native zero-core QR/HOSVD branch, independent of adaptive construction."""

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.tech.chebtech import Chebtech2


def test_zero_core_hosvd():
    # @chebfun3/hosvd.m and @chebfun/qr.m, Chebfun7574c77.
    factor = Chebtech2.from_coeffs(jnp.asarray([0.]))
    f = Chebfun3([factor], [factor], [factor], jnp.zeros((1, 1, 1)),
                 (-1., 1., -1., 1., -1., 1.))
    values, g = f.hosvd()
    for mode in values:
        assert jnp.array_equal(mode, jnp.zeros((1,)))
    x = jnp.asarray([-.9, 0., .8])
    assert jnp.array_equal(g(x, x, x), jnp.zeros_like(x))
