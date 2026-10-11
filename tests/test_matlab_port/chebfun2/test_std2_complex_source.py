"""Analytic controls for @separableApprox/std2.m, Chebfun commit: 7574c77.

No native tests/chebfun2/test_std2.m exists in the pinned reference tree.
The complex controls exercise the literal h*conj(h) source contract.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2


@pytest.mark.parametrize(
    "fn,variance",
    [
        (lambda x, y: x + y, 2 / 3),
        (lambda x, y: x + 1j * y, 2 / 3),
        (lambda x, y: x + 1j * y + 2 + 3j, 2 / 3),
        (lambda x, y: 1j * x + 3j, 1 / 3),
    ],
)
def test_analytic_variance(fn, variance):
    f = Chebfun2.from_function(fn)
    # Supplemental analytic bound; native mean/std bounds are unchanged.
    assert jnp.abs(f.std2() - jnp.sqrt(variance)) < 1000 * jnp.finfo(jnp.float64).eps
