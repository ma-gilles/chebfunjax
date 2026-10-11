"""All six original Chebfun2 rank predicates, Chebfun pin7574c77.

Provenance
----------
MATLAB source: tests/chebfun2/test_rank.m.
The source requires rank <= min of both factor lengths, including the Airy case.
"""

import gc

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.utils.airy_general import airy_all

CASES = [
    lambda x, y: 1 / (1 + 100 * (x**2 - y**2) ** 2),
    lambda x, y: 1 / (1 + 100 * (0.5 - x**2 - y**2) ** 2),
    lambda x, y: (
        1 / (1 + 1000 * ((x - 0.5) ** 2 * (y + 0.5) ** 2 * (x + 0.5) ** 2 * (y - 0.5) ** 2))
    ),
    lambda x, y: jnp.cos(10 * (x**2 + y)) * jnp.sin(10 * (x + y**2)),
    lambda x, y: jnp.real(airy_all(5 * (x + y**2))[0]) * jnp.real(airy_all(-5 * (x**2 + y**2))[0]),
    lambda x, y: jnp.tanh(10 * x) * jnp.tanh(10 * y) / jnp.tanh(10.0) ** 2 + jnp.cos(5 * x),
]


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()
    gc.collect()


@pytest.mark.parametrize("clause", range(1, 7))
def test_native_rank_at_most_both_factor_lengths(clause):
    f = Chebfun2.from_function(CASES[clause - 1])
    m, n = f.length()
    assert f.rank <= min(m, n)
